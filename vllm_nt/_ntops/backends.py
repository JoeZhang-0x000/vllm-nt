from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch

from vllm_nt._ntops.config import config_for

LOG = logging.getLogger("vllm_nt")
_DISABLE_OPS_ENV = "VLLM_NT_DISABLE_OPS"


@dataclass
class BackendStats:
    attempts: int = 0
    hits: int = 0
    failures: int = 0
    fallbacks: int = 0


_DISABLED: set[str] = set()
_STATS: dict[str, dict[str, BackendStats]] = {}


def disabled_ops() -> set[str]:
    value = os.environ.get(_DISABLE_OPS_ENV, "")
    env_disabled = {item.strip() for item in value.split(",") if item.strip()}
    return _DISABLED | env_disabled


def configured_backend(op_name: str) -> str:
    return config_for(op_name).backend


def fallback_backend(op_name: str) -> str:
    return config_for(op_name).fallback_backend


def active_backend(op_name: str) -> str:
    cfg = config_for(op_name)
    return cfg.fallback_backend if op_name in disabled_ops() else cfg.backend


def backend_enabled(op_name: str) -> bool:
    return active_backend(op_name) != "original"


def _stats(op_name: str, backend: str) -> BackendStats:
    return _STATS.setdefault(op_name, {}).setdefault(backend, BackendStats())


def record_attempt(op_name: str, backend: str) -> None:
    _stats(op_name, backend).attempts += 1


def record_hit(op_name: str, backend: str | None = None) -> None:
    _stats(op_name, backend or active_backend(op_name)).hits += 1


def record_failure(op_name: str, backend: str | None = None) -> None:
    _stats(op_name, backend or active_backend(op_name)).failures += 1


def record_fallback(op_name: str, backend: str | None = None) -> None:
    _stats(op_name, backend or fallback_backend(op_name)).fallbacks += 1


def backend_stats(op_name: str) -> dict[str, BackendStats]:
    return _STATS.get(op_name, {})


def reset_backend_state() -> None:
    _DISABLED.clear()
    _STATS.clear()


def _call_backend(
    op_name: str,
    backend: str,
    call_original: Callable[[], Any],
    call_infinicore: Callable[[], Any] | None,
    call_ninetoothed: Callable[[], Any] | None,
) -> Any:
    if backend == "original":
        record_fallback(op_name, backend)
        return call_original()

    fn = call_infinicore if backend == "infinicore" else call_ninetoothed
    if fn is None:
        raise RuntimeError(f"{op_name} has no adapter for backend {backend}")
    record_attempt(op_name, backend)
    result = fn()
    if backend == "infinicore":
        record_hit(op_name, backend)
    return result


def route(
    op_name: str,
    call_original: Callable[[], Any],
    *,
    call_infinicore: Callable[[], Any] | None = None,
    call_ninetoothed: Callable[[], Any] | None = None,
) -> Any:
    cfg = config_for(op_name)
    backend = active_backend(op_name)
    try:
        return _call_backend(
            op_name, backend, call_original, call_infinicore, call_ninetoothed
        )
    except Exception as exc:
        record_failure(op_name, backend)
        if cfg.disable_backend_on_first_failure and backend != "original":
            _DISABLED.add(op_name)
            LOG.warning(
                "vllm-nt: %s backend %s disabled after failure; falling back to %s: %s",
                op_name,
                backend,
                cfg.fallback_backend,
                exc,
            )
        if cfg.fallback_backend == backend:
            raise
        return _call_backend(
            op_name,
            cfg.fallback_backend,
            call_original,
            call_infinicore,
            call_ninetoothed,
        )


def as_infini(tensor: torch.Tensor):
    import infinicore

    return infinicore.from_torch(tensor.contiguous() if not tensor.is_contiguous() else tensor)


def as_infini_strided(tensor: torch.Tensor):
    import infinicore
    from infinicore.tensor import to_infinicore_dtype

    device_index = tensor.device.index if tensor.device.index is not None else 0
    return infinicore.strided_from_blob(
        tensor.data_ptr(),
        list(tensor.shape),
        list(tensor.stride()),
        dtype=to_infinicore_dtype(tensor.dtype),
        device=infinicore.device(tensor.device.type, device_index),
    )


def rms_norm_infinicore(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    import torch
    import infinicore.nn.functional as F

    out = torch.empty_like(x)
    F.rms_norm(as_infini(x), list(weight.shape), as_infini(weight), eps, out=as_infini(out))
    return out


def linear_infinicore(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    import torch
    import infinicore.nn.functional as F

    out = torch.empty(x.shape[:-1] + (weight.shape[0],), dtype=x.dtype, device=x.device)
    F.linear(
        as_infini(x),
        as_infini(weight),
        None if bias is None else as_infini(bias),
        out=as_infini(out),
    )
    return out


def pa_cache_views(kv_cache: torch.Tensor, key: torch.Tensor):
    key_cache, value_cache = kv_cache.unbind(0)
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise RuntimeError(
            f"expected 4D paged KV cache tensors, got {key_cache.shape}/{value_cache.shape}"
        )
    num_kv_heads = key.shape[1]
    if key_cache.shape[1] == num_kv_heads:
        return as_infini_strided(key_cache), as_infini_strided(value_cache)
    if key_cache.shape[2] == num_kv_heads:
        return as_infini_strided(key_cache.permute(0, 2, 1, 3)), as_infini_strided(
            value_cache.permute(0, 2, 1, 3)
        )
    raise RuntimeError(f"cannot infer KV cache layout from key={key.shape}, cache={key_cache.shape}")


def prefill_total_lens(attn_metadata) -> torch.Tensor:
    cu_prefix_kv_lens = getattr(attn_metadata, "cu_prefix_kv_lens", None)
    if cu_prefix_kv_lens is not None:
        return (cu_prefix_kv_lens[1:] - cu_prefix_kv_lens[:-1]).to(torch.int64)

    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None:
        raise RuntimeError("missing seq_lens for paged attention prefill")
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
    num_prefills = int(getattr(attn_metadata, "num_prefills", 0))
    if seq_lens.shape[0] >= num_decodes + num_prefills:
        return seq_lens[num_decodes : num_decodes + num_prefills]
    if seq_lens.shape[0] == num_prefills:
        return seq_lens
    raise RuntimeError(
        f"cannot derive prefill total lengths from seq_lens={seq_lens.shape}, "
        f"num_decodes={num_decodes}, num_prefills={num_prefills}"
    )


def store_kv_cache_infinicore(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    import infinicore

    key_cache, value_cache = pa_cache_views(kv_cache, key)
    infinicore.paged_caching(
        key_cache,
        value_cache,
        as_infini(key),
        as_infini(value),
        as_infini(slot_mapping.flatten()),
    )


def paged_attention_prefill_infinicore(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> None:
    import infinicore

    key_cache, value_cache = pa_cache_views(kv_cache, key)
    num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    q = query[num_decode_tokens:num_actual_tokens]
    if q.numel() == 0:
        return
    out = output[num_decode_tokens:num_actual_tokens].view(q.shape)
    infinicore.paged_attention_prefill(
        as_infini(q),
        key_cache,
        value_cache,
        as_infini(attn_metadata.prefill_block_table),
        as_infini(prefill_total_lens(attn_metadata)),
        as_infini(attn_metadata.prefill_query_start_loc),
        as_infini(self.alibi_slopes) if self.alibi_slopes is not None else None,
        self.scale,
        out=as_infini(out),
    )


def paged_attention_decode_infinicore(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> None:
    import infinicore

    num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
    num_decodes = int(getattr(attn_metadata, "num_decodes", 0))
    if num_decode_tokens == 0:
        return
    if num_decode_tokens != num_decodes:
        raise RuntimeError(
            "speculative decode is not supported by InfiniCore PA adapter: "
            f"tokens={num_decode_tokens}, decodes={num_decodes}"
        )

    key_cache, value_cache = pa_cache_views(kv_cache, key)
    q = query[:num_decode_tokens]
    out = output[:num_decode_tokens].view(q.shape)
    infinicore.paged_attention(
        as_infini(q),
        key_cache,
        value_cache,
        as_infini(attn_metadata.decode_block_table),
        as_infini(attn_metadata.decode_seq_lens),
        as_infini(self.alibi_slopes) if self.alibi_slopes is not None else None,
        self.scale,
        out=as_infini(out),
    )
