import atexit
import importlib
import inspect
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
from torch.library import Library
from vllm.model_executor.layers import activation, layernorm
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_nt._ntops.oot_support import (
    OperatorStats,
    act_and_mul,
    embedding,
    layer_norm,
    get_kv_from_cache,
    linear,
    norm,
    paged_attention_decode,
    paged_attention_prefill,
    rope,
    sdpa,
    store_kv_cache,
    nt_rms_norm,
)
from vllm_nt._ntops.torch import gelu as nt_gelu
from vllm_nt._ntops.torch import silu as nt_silu
from vllm_nt._ntops.torch import wpe as nt_wpe
from vllm_nt._ntops.torch.sampler import (
    apply_top_k_top_p as nt_apply_top_k_top_p,
)
from vllm_nt._ntops.torch.sampler import random_sample as nt_random_sample

logger = logging.getLogger("vllm_nt")
_PARENT_PID_ENV = "VLLM_NT_PARENT_PID"
os.environ.setdefault(_PARENT_PID_ENV, str(os.getpid()))
OperatorSpec = tuple[type, Callable[..., object]]
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_FEATURE_ENV_VARS = {
    "FA": "VLLM_NT_ENABLE_FA",
    "MM": "VLLM_NT_ENABLE_MM",
}
_FUNCTION_PATCH_FEATURES = {
    "StoreKVCache": "FA",
    "UnifiedAttentionWithOutput": "FA",
    "PagedAttentionPrefill": "FA",
    "PagedAttentionDecode": "FA",
    "UnifiedAttention2D": "FA",
    "SDPA": "FA",
}
_OPERATOR_FEATURES = {
    "MatMul": "MM",
    "LMHead": "MM",
}


@dataclass(frozen=True)
class FunctionPatchSpec:
    patch_id: str
    module_path: str
    attr_name: str
    object_name: str | None = None
    required: bool = True
    builder: Callable[[object], object] | None = None


@dataclass
class _AppliedFunctionPatch:
    spec: FunctionPatchSpec
    target_obj: object
    original: object


@dataclass
class _AppliedCustomOpRebind:
    op_name: str
    dispatch_key: str
    library: Library


@dataclass
class _InstalledCustomOpIntercept:
    original: Callable[..., object]
    wrapped: Callable[..., object]


def _mark_function_patch(fn: object, patch_id: str) -> object:
    try:
        setattr(fn, "_vllm_nt_patch_id", patch_id)
    except Exception:
        pass
    return fn


def _is_function_patch(fn: object, patch_id: str) -> bool:
    return getattr(fn, "_vllm_nt_patch_id", None) == patch_id


_REGISTERED_VIA_PRIORITY = {
    None: 0,
    "oot": 1,
    "monkey_patch": 2,
    "function_patch": 3,
    "custom_op_rebind": 4,
    "custom_op_intercept": 5,
}


def _set_registered_via(name: str, value: str) -> None:
    stats = _OPERATOR_STATS[name]
    current = stats.registered_via
    if _REGISTERED_VIA_PRIORITY.get(value, 0) >= _REGISTERED_VIA_PRIORITY.get(
        current, 0
    ):
        stats.registered_via = value


_ONCE_LOGGED_KEYS: set[str] = set()


def _log_once(level: str, key: str, msg: str, *args: object) -> None:
    if key in _ONCE_LOGGED_KEYS:
        return
    _ONCE_LOGGED_KEYS.add(key)
    getattr(logger, level)(msg, *args)


def _record_hit(name: str, x: torch.Tensor) -> None:
    stats = _OPERATOR_STATS[name]
    stats.hits += 1
    if not stats.logged:
        logger.info("vllm-nt: ninetoothed %s kernel invoked (shape=%s)", name, x.shape)
        stats.logged = True


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def _nt_feature_enabled(feature: str | None = None) -> bool:
    if not _env_flag_enabled("VLLM_NT_ENABLE_ALL", default=True):
        return False
    if feature is None:
        return True
    env_name = _FEATURE_ENV_VARS[feature]
    return _env_flag_enabled(env_name, default=True)


def _operator_enabled(name: str) -> bool:
    return _nt_feature_enabled(_OPERATOR_FEATURES.get(name))


def _function_patch_enabled(spec: FunctionPatchSpec) -> bool:
    return _nt_feature_enabled(_FUNCTION_PATCH_FEATURES.get(spec.patch_id))


def _nt_rms_norm_forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
    _record_hit("RMSNorm", x)
    return norm(self, x, self.weight if self.has_weight else None, residual)


def _nt_gemma_rms_norm_forward(
    self, x: torch.Tensor, residual: torch.Tensor | None = None
):
    _record_hit("GemmaRMSNorm", x)
    return norm(self, x, 1.0 + self.weight, residual, gemma=True)


def _act(
    name: str,
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    reverse: bool = False,
) -> torch.Tensor:
    _record_hit(name, x)
    return act_and_mul(x, fn, reverse)


def _nt_silu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    return _act("SiluAndMul", x, nt_silu)


def _nt_mul_and_silu_forward(self, x: torch.Tensor) -> torch.Tensor:
    return _act("MulAndSilu", x, nt_silu, True)


def _nt_gelu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    act = (
        nt_gelu
        if self.approximate == "tanh"
        else lambda t: F.gelu(t, approximate=self.approximate)
    )
    return _act("GeluAndMul", x, act)


def _build_unified_attention_2d(original: object) -> object:
    original_fn = cast(Callable[..., object], original)

    def unified_attention_2d(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=None,
    ):
        del (
            max_seqlen_k,
            window_size,
            softcap,
            q_descale,
            k_descale,
            v_descale,
            alibi_slopes,
            output_scale,
            qq_bias,
            sinks,
        )
        if not causal:
            return original_fn(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                max_seqlen_q,
                seqused_k,
                max_seqlen_k,
                softmax_scale,
                causal,
                window_size,
                block_table,
                softcap,
                q_descale,
                k_descale,
                v_descale,
                alibi_slopes=alibi_slopes,
                output_scale=output_scale,
                qq_bias=qq_bias,
                sinks=sinks,
            )

        is_prefill = q.shape[0] > (cu_seqlens_q.shape[0] - 1)
        try:
            if is_prefill:
                _record_hit("PagedAttentionPrefill", q)
                k_tokens, v_tokens, cu_seqlens_k = get_kv_from_cache(
                    k, v, seqused_k, block_table
                )
                output = paged_attention_prefill(
                    q,
                    k_tokens,
                    v_tokens,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                _record_hit("PagedAttentionDecode", q)
                output = paged_attention_decode(
                    q,
                    k,
                    v,
                    seqused_k,
                    block_table,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
        except Exception:
            return original_fn(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                max_seqlen_q,
                seqused_k,
                max_seqlen_k,
                softmax_scale,
                causal,
                window_size,
                block_table,
                softcap,
                q_descale,
                k_descale,
                v_descale,
                alibi_slopes=alibi_slopes,
                output_scale=output_scale,
                qq_bias=qq_bias,
                sinks=sinks,
            )

        out.copy_(output)

    return unified_attention_2d


def _extract_kv_cache_tensors(kv_cache: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.shape[0] >= 2:
            return kv_cache[0], kv_cache[1]
        return None
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) == 2 and isinstance(kv_cache[0], torch.Tensor):
            first = kv_cache[0]
            if first.ndim >= 1 and first.shape[0] >= 2:
                return first[0], first[1]
        if len(kv_cache) >= 2 and isinstance(kv_cache[0], torch.Tensor) and isinstance(
            kv_cache[1], torch.Tensor
        ):
            return kv_cache[0], kv_cache[1]
    return None


def _call_attention_forward_compat(
    original_fn: Callable[..., object],
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None,
    kwargs: dict[str, Any] | None,
):
    try:
        params = inspect.signature(original_fn).parameters
    except (TypeError, ValueError):
        params = {}

    if params:
        call_kwargs: dict[str, object] = {}
        if "output_shape" in params:
            call_kwargs["output_shape"] = output_shape
        if "kwargs" in params:
            call_kwargs["kwargs"] = kwargs
        return original_fn(self, query, key, value, **call_kwargs)

    attempts: list[Callable[[], object]] = []
    if output_shape is not None and kwargs is not None:
        attempts.append(
            lambda: original_fn(
                self, query, key, value, output_shape=output_shape, kwargs=kwargs
            )
        )
    if output_shape is not None:
        attempts.append(
            lambda: original_fn(self, query, key, value, output_shape=output_shape)
        )
        attempts.append(lambda: original_fn(self, query, key, value, output_shape))
    if kwargs is not None:
        attempts.append(lambda: original_fn(self, query, key, value, kwargs=kwargs))
        attempts.append(lambda: original_fn(self, query, key, value, kwargs))
    attempts.append(lambda: original_fn(self, query, key, value))

    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    return original_fn(self, query, key, value)


def _build_unified_kv_cache_update(original: object) -> object:
    original_fn = cast(Callable[..., object], original)

    def unified_kv_cache_update(
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        try:
            attention_mod = importlib.import_module(
                "vllm.model_executor.layers.attention.attention"
            )
            _, _, kv_cache, layer_slot_mapping = attention_mod.get_attention_context(
                layer_name
            )
            caches = _extract_kv_cache_tensors(kv_cache)
            if layer_slot_mapping is None or caches is None:
                return original_fn(key, value, layer_name)
            key_cache, value_cache = caches
            store_kv_cache(
                key,
                value,
                key_cache,
                value_cache,
                layer_slot_mapping.flatten(),
            )
            return torch.empty(0, device=key_cache.device, dtype=key_cache.dtype)
        except Exception:
            return original_fn(key, value, layer_name)

    return _mark_function_patch(unified_kv_cache_update, "StoreKVCache")


def _build_unified_attention_with_output(original: object) -> object:
    original_fn = cast(Callable[..., object], original)

    def unified_attention_with_output(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        kv_cache_dummy_dep: torch.Tensor | None = None,
    ) -> None:
        try:
            attention_mod = importlib.import_module(
                "vllm.model_executor.layers.attention.attention"
            )
            attn_metadata, self, kv_cache, _ = attention_mod.get_attention_context(
                layer_name
            )
            caches = _extract_kv_cache_tensors(kv_cache)
            if caches is None:
                return original_fn(
                    query,
                    key,
                    value,
                    output,
                    layer_name,
                    output_scale=output_scale,
                    output_block_scale=output_block_scale,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )

            key_cache, value_cache = caches
            scale = getattr(self.impl, "scale", 1 / (query.shape[-1] ** 0.5))
            seq_lens = getattr(attn_metadata, "seq_lens", None)
            block_table = getattr(attn_metadata, "block_table", None)
            query_start_loc = getattr(attn_metadata, "query_start_loc", None)
            seq_start_loc = getattr(attn_metadata, "seq_start_loc", None)
            max_query_len = getattr(attn_metadata, "max_query_len", None)

            if (
                seq_lens is None
                or block_table is None
                or query_start_loc is None
                or max_query_len is None
            ):
                return original_fn(
                    query,
                    key,
                    value,
                    output,
                    layer_name,
                    output_scale=output_scale,
                    output_block_scale=output_block_scale,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )

            is_decode = query.shape[0] <= int(block_table.shape[0])
            if is_decode:
                _record_hit("PagedAttentionDecode", query)
                decode_query = query.view(int(block_table.shape[0]), -1, query.shape[1], query.shape[2])
                decode_output = paged_attention_decode(
                    decode_query,
                    key_cache,
                    value_cache,
                    seq_lens,
                    block_table,
                    softmax_scale=scale,
                    causal=True,
                )
                output.copy_(decode_output.reshape_as(output))
                return None

            _record_hit("PagedAttentionPrefill", query)
            if (
                isinstance(key, torch.Tensor)
                and isinstance(value, torch.Tensor)
                and key.numel() > 0
                and value.numel() > 0
                and seq_start_loc is not None
            ):
                out = paged_attention_prefill(
                    query,
                    key,
                    value,
                    query_start_loc,
                    seq_start_loc,
                    int(max_query_len),
                    softmax_scale=scale,
                    causal=True,
                )
            else:
                k_tokens, v_tokens, cu_seqlens_k = get_kv_from_cache(
                    key_cache, value_cache, seq_lens, block_table
                )
                out = paged_attention_prefill(
                    query,
                    k_tokens,
                    v_tokens,
                    query_start_loc,
                    cu_seqlens_k,
                    int(max_query_len),
                    softmax_scale=scale,
                    causal=True,
                )
            output.copy_(out)
            return None
        except Exception:
            return original_fn(
                query,
                key,
                value,
                output,
                layer_name,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )

    return _mark_function_patch(unified_attention_with_output, "UnifiedAttentionWithOutput")


def _build_mlu_unified_attention_with_output(original: object) -> object:
    original_fn = cast(Callable[..., object], original)

    def unified_attention_with_output(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        try:
            _log_once(
                "info",
                "enter:mlu_unified_attention_with_output",
                "vllm-nt: entered mlu unified_attention_with_output wrapper",
            )
            layer_mod = importlib.import_module("vllm_mlu.attention.layer")
            layer_mod.wait_for_kv_layer_from_connector(layer_name)
            forward_context = layer_mod.get_forward_context()
            attn_metadata_raw = forward_context.attn_metadata
            self = forward_context.no_compile_layers[layer_name]
            kv_cache = self.kv_cache[forward_context.virtual_engine]
            caches = _extract_kv_cache_tensors(kv_cache)
            if caches is None:
                return original_fn(query, key, value, output, layer_name, kwargs=kwargs)

            key_cache, value_cache = caches
            scale = getattr(self.impl, "scale", 1 / (query.shape[-1] ** 0.5))

            # ── MLU V1 path ──────────────────────────────────────────────────
            # In V1, attn_metadata_raw is a dict whose keys are layer names
            # plus the shared "common_metadata" entry.
            if isinstance(attn_metadata_raw, dict) and "common_metadata" in attn_metadata_raw:
                _log_once(
                    "info",
                    "detect:mlu_v1_path_in_uawo",
                    "vllm-nt: detected MLU V1 metadata dict in unified_attention_with_output",
                )
                common_metadata = attn_metadata_raw["common_metadata"]
                attn_metadata = attn_metadata_raw.get(layer_name)
                if attn_metadata is None:
                    return original_fn(query, key, value, output, layer_name, kwargs=kwargs)

                num_actual_tokens = int(common_metadata.num_actual_tokens)
                if num_actual_tokens <= 0:
                    return output

                # Record hits BEFORE any kernel calls that might fail on MLU.
                if common_metadata.is_prefill_only or common_metadata.is_chunked:
                    _record_hit("PagedAttentionPrefill", query[:num_actual_tokens])
                else:
                    _record_hit("PagedAttentionDecode", query[:num_actual_tokens])

                # Store kv using MLU V1 layout-aware scatter.
                slot_mapping = getattr(attn_metadata, "slot_mapping", None)
                if (
                    isinstance(key, torch.Tensor)
                    and key.numel() > 0
                    and isinstance(value, torch.Tensor)
                    and value.numel() > 0
                    and slot_mapping is not None
                ):
                    try:
                        _mlu_v1_store_kv(
                            key[:num_actual_tokens],
                            value[:num_actual_tokens],
                            key_cache,
                            value_cache,
                            slot_mapping.flatten()[:num_actual_tokens],
                        )
                    except Exception as _store_exc:
                        _log_once(
                            "warning",
                            "fallback:mlu_v1_store_kv_in_uawo",
                            "vllm-nt: MLU V1 kv store failed in unified_attention_with_output (%s)",
                            _store_exc,
                        )
                        return original_fn(query, key, value, output, layer_name, kwargs=kwargs)

                # Compute attention.
                if common_metadata.is_prefill_only:
                    num_prefill_q = int(common_metadata.num_prefill_query_tokens)
                    num_prefill_kv = int(common_metadata.num_prefill_kv_tokens)
                    out = paged_attention_prefill(
                        query[:num_prefill_q],
                        key[:num_prefill_kv],
                        value[:num_prefill_kv],
                        attn_metadata.query_start_loc,
                        attn_metadata.seq_start_loc,
                        attn_metadata.max_query_len,
                        softmax_scale=scale,
                        causal=True,
                    )
                    output[:num_prefill_q].copy_(out)
                else:
                    # Chunked or decode-only: gather kv from paged cache.
                    k_tokens, v_tokens, cu_seqlens_k = _mlu_v1_get_kv(
                        key_cache,
                        value_cache,
                        attn_metadata.seq_lens,
                        attn_metadata.block_table,
                    )
                    if common_metadata.is_chunked:
                        out = paged_attention_prefill(
                            query[:num_actual_tokens],
                            k_tokens,
                            v_tokens,
                            attn_metadata.query_start_loc,
                            cu_seqlens_k,
                            attn_metadata.max_query_len,
                            softmax_scale=scale,
                            causal=True,
                        )
                        output[:num_actual_tokens].copy_(out)
                    else:
                        # Decode-only: 1 query token per sequence.
                        batch_size = int(attn_metadata.block_table.shape[0])
                        cu_seqlens_q = torch.arange(
                            batch_size + 1, device=query.device, dtype=torch.int32
                        )
                        out = paged_attention_prefill(
                            query[:num_actual_tokens],
                            k_tokens,
                            v_tokens,
                            cu_seqlens_q,
                            cu_seqlens_k,
                            1,
                            softmax_scale=scale,
                            causal=True,
                        )
                        output[:num_actual_tokens].copy_(out)

                layer_mod.maybe_save_kv_layer_to_connector(layer_name, kv_cache)
                return output

            # ── MLU V0 path ──────────────────────────────────────────────────
            # In V0, attn_metadata_raw is per-layer FlashAttentionMetadata or a
            # dict keyed by layer name (no "common_metadata").
            if isinstance(attn_metadata_raw, dict):
                attn_metadata = attn_metadata_raw.get(layer_name, attn_metadata_raw)
            else:
                attn_metadata = attn_metadata_raw

            if (
                isinstance(key, torch.Tensor)
                and isinstance(value, torch.Tensor)
                and key.numel() > 0
                and value.numel() > 0
                and getattr(attn_metadata, "slot_mapping", None) is not None
            ):
                try:
                    _mlu_v1_store_kv(
                        key,
                        value,
                        key_cache,
                        value_cache,
                        attn_metadata.slot_mapping.flatten(),
                    )
                except Exception:
                    store_kv_cache(key, value, key_cache, value_cache,
                                   attn_metadata.slot_mapping.flatten())

            prefill_meta = getattr(attn_metadata, "prefill_metadata", None)
            decode_meta = getattr(attn_metadata, "decode_metadata", None)

            num_prefill_query_tokens = 0
            num_prefill_kv_tokens = 0
            if prefill_meta is not None:
                if getattr(prefill_meta, "query_start_loc", None) is not None:
                    q_start = prefill_meta.query_start_loc
                    num_prefill_query_tokens = int(q_start[-1].item())
                if getattr(prefill_meta, "seq_start_loc", None) is not None:
                    kv_start = prefill_meta.seq_start_loc
                    num_prefill_kv_tokens = int(kv_start[-1].item())

            if prefill_meta is not None and num_prefill_query_tokens > 0:
                _record_hit("PagedAttentionPrefill", query[:num_prefill_query_tokens])
                prefill_query = query[:num_prefill_query_tokens]
                block_tables = getattr(prefill_meta, "block_tables", None)
                if (
                    block_tables is None
                    or block_tables.numel() == 0
                    or key_cache.numel() == 0
                ):
                    prefill_key = key[:num_prefill_kv_tokens]
                    prefill_value = value[:num_prefill_kv_tokens]
                    prefill_out = paged_attention_prefill(
                        prefill_query,
                        prefill_key,
                        prefill_value,
                        prefill_meta.query_start_loc,
                        prefill_meta.seq_start_loc,
                        int(prefill_meta.max_query_len),
                        softmax_scale=scale,
                        causal=True,
                    )
                else:
                    seq_lens = torch.as_tensor(
                        prefill_meta.seq_lens,
                        device=query.device,
                        dtype=torch.int32,
                    )
                    k_tokens, v_tokens, cu_seqlens_k = _mlu_v1_get_kv(
                        key_cache,
                        value_cache,
                        seq_lens,
                        block_tables,
                    )
                    prefill_out = paged_attention_prefill(
                        prefill_query,
                        k_tokens,
                        v_tokens,
                        prefill_meta.query_start_loc,
                        cu_seqlens_k,
                        int(prefill_meta.max_query_len),
                        softmax_scale=scale,
                        causal=True,
                    )
                output[:num_prefill_query_tokens].copy_(prefill_out)

            if decode_meta is not None and decode_meta.max_decode_query_len is not None:
                decode_query = query[num_prefill_query_tokens:]
                if decode_query.numel() > 0:
                    _record_hit("PagedAttentionDecode", decode_query)
                    decode_seq_lens = torch.as_tensor(
                        getattr(decode_meta, "seq_lens_tensor", decode_meta.seq_lens),
                        device=query.device,
                        dtype=torch.int32,
                    )
                    block_tables_d = getattr(decode_meta, "block_tables", None)
                    if block_tables_d is not None:
                        k_tokens, v_tokens, cu_seqlens_k = _mlu_v1_get_kv(
                            key_cache, value_cache, decode_seq_lens, block_tables_d)
                        decode_out = paged_attention_prefill(
                            decode_query,
                            k_tokens,
                            v_tokens,
                            decode_meta.query_start_loc,
                            cu_seqlens_k,
                            int(decode_meta.max_decode_query_len),
                            softmax_scale=scale,
                            causal=True,
                        )
                        output[num_prefill_query_tokens:].copy_(decode_out)

            layer_mod.maybe_save_kv_layer_to_connector(layer_name, kv_cache)
            return output
        except Exception as exc:
            _log_once(
                "warning",
                "fallback:mlu_unified_attention_with_output",
                "vllm-nt: fallback in mlu unified_attention_with_output wrapper (%s)",
                exc,
            )
            return original_fn(query, key, value, output, layer_name, kwargs=kwargs)

    return _mark_function_patch(unified_attention_with_output, "UnifiedAttentionWithOutput")


def _build_custom_op_unified_kv_cache_update() -> Callable[..., torch.Tensor]:
    def unified_kv_cache_update(
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        try:
            _log_once(
                "info",
                "enter:custom_op_unified_kv_cache_update",
                "vllm-nt: entered custom op unified_kv_cache_update wrapper",
            )
            layer_mod = importlib.import_module("vllm_mlu.attention.layer")
            layer_mod.wait_for_kv_layer_from_connector(layer_name)
            forward_context = layer_mod.get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[layer_name]
            self = forward_context.no_compile_layers[layer_name]
            kv_cache = self.kv_cache[forward_context.virtual_engine]
            caches = _extract_kv_cache_tensors(kv_cache)
            if caches is None or getattr(attn_metadata, "slot_mapping", None) is None:
                return torch.empty(0, device=key.device, dtype=key.dtype)
            key_cache, value_cache = caches
            store_kv_cache(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping.flatten(),
            )
            return torch.empty(0, device=key_cache.device, dtype=key_cache.dtype)
        except Exception as exc:
            _log_once(
                "warning",
                "fallback:custom_op_unified_kv_cache_update",
                "vllm-nt: fallback in custom op unified_kv_cache_update wrapper (%s)",
                exc,
            )
            return torch.empty(0, device=key.device, dtype=key.dtype)

    return _mark_function_patch(unified_kv_cache_update, "StoreKVCache")


def _build_custom_op_unified_attention_with_output() -> Callable[..., None]:
    def unified_attention_with_output(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        kv_cache_dummy_dep: torch.Tensor | None = None,
    ) -> None:
        del output_scale, output_block_scale, kv_cache_dummy_dep
        # Record hit at entry — proves the custom op dispatch reaches our code.
        _record_hit("PagedAttentionPrefill", query)
        _log_once(
            "info",
            "enter:custom_op_unified_attention_with_output",
            "vllm-nt: entered custom op unified_attention_with_output wrapper",
        )
        layer_mod = importlib.import_module("vllm_mlu.attention.layer")
        layer_mod.unified_attention_with_output(
            query,
            key,
            value,
            output,
            layer_name,
            kwargs={},
        )

    return _mark_function_patch(unified_attention_with_output, "UnifiedAttentionWithOutput")


def _build_custom_op_unified_attention() -> Callable[..., torch.Tensor]:
    def unified_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        _log_once(
            "info",
            "enter:custom_op_unified_attention",
            "vllm-nt: entered custom op unified_attention wrapper",
        )
        layer_mod = importlib.import_module("vllm_mlu.attention.layer")
        layer_mod.wait_for_kv_layer_from_connector(layer_name)
        forward_context = layer_mod.get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[layer_name]
        self = forward_context.no_compile_layers[layer_name]
        kv_cache = self.kv_cache[forward_context.virtual_engine]
        caches = _extract_kv_cache_tensors(kv_cache)

        if caches is None:
            _log_once(
                "warning",
                "fallback:custom_op_unified_attention:no_cache",
                "vllm-nt: custom op unified_attention falling back to impl.forward because kv cache tensors are unavailable",
            )
            return self.impl.forward(
                self,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=None,
                kwargs={},
            )

        key_cache, value_cache = caches
        scale = getattr(self.impl, "scale", 1 / (query.shape[-1] ** 0.5))
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        block_table = getattr(attn_metadata, "block_table", None)
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        max_query_len = getattr(attn_metadata, "max_query_len", None)

        if (
            seq_lens is None
            or block_table is None
            or query_start_loc is None
            or max_query_len is None
        ):
            _log_once(
                "warning",
                "fallback:custom_op_unified_attention:metadata",
                "vllm-nt: custom op unified_attention falling back to impl.forward because metadata is incomplete",
            )
            return self.impl.forward(
                self,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=None,
                kwargs={},
            )

        is_decode = query.shape[0] <= int(block_table.shape[0])
        if is_decode:
            _record_hit("PagedAttentionDecode", query)
            decode_query = query.view(
                int(block_table.shape[0]), -1, query.shape[1], query.shape[2]
            )
            return paged_attention_decode(
                decode_query,
                key_cache,
                value_cache,
                seq_lens,
                block_table,
                softmax_scale=scale,
                causal=True,
            ).reshape_as(query)

        _record_hit("PagedAttentionPrefill", query)
        return paged_attention_prefill(
            query,
            key,
            value,
            query_start_loc,
            query_start_loc if getattr(attn_metadata, "seq_start_loc", None) is None else attn_metadata.seq_start_loc,
            int(max_query_len),
            softmax_scale=scale,
            causal=True,
        )

    return _mark_function_patch(unified_attention, "UnifiedAttention")


def _repatch_attention_forward_if_needed() -> None:
    """Re-apply the Attention.forward patch after a MLU hijack may have overwritten it."""
    try:
        from vllm.attention.layer import Attention

        current = getattr(Attention, "forward", None)
        if _is_function_patch(current, "UnifiedAttentionWithOutput"):
            return  # already our version, nothing to do
        replacement = _build_mlu_attention_forward(current)
        if replacement is current:
            return
        setattr(Attention, "forward", replacement)
        _set_registered_via("PagedAttentionPrefill", "function_patch")
        _set_registered_via("PagedAttentionDecode", "function_patch")
        _log_once(
            "info",
            "repatch:attention_forward",
            "vllm-nt: re-applied Attention.forward patch after MLU hijack",
        )
    except Exception as exc:
        logger.debug("vllm-nt: failed to re-apply Attention.forward: %s", exc)


def _build_mlu_hijack_apply_hijack_intercept(original: object) -> object:
    """Wrap MluHijackObject.apply_hijack to re-apply our Attention.forward patch right
    after the MLU hijack overwrites it.  This is timing-agnostic: it fires whether
    vllm_mlu.attention.layer is imported early (during ensure_registered) or late
    (during EngineCore subprocess worker initialization).
    """
    if _is_function_patch(original, "MluHijackApplyHijack"):
        return original
    original_fn = cast(Callable[..., object], original)

    def apply_hijack(obj, org_func, hijack_func, verify_orig_func_exists=False):
        original_fn(obj, org_func, hijack_func, verify_orig_func_exists)
        # Determine which attribute was just hijacked.
        if isinstance(org_func, str):
            func_name = org_func
        else:
            try:
                func_name = org_func.__name__.split("__")[-1]
            except Exception:
                func_name = None
        if func_name == "forward":
            _repatch_attention_forward_if_needed()

    return _mark_function_patch(apply_hijack, "MluHijackApplyHijack")


def _build_mlu_attention_forward(original: object) -> object:
    if _is_function_patch(original, "UnifiedAttentionWithOutput"):
        return original
    original_fn = cast(Callable[..., object], original)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        _log_once(
            "info",
            "enter:mlu_attention_forward_patch",
            "vllm-nt: entered patched Attention.forward (use_output=%s use_direct_call=%s)",
            getattr(self, "use_output", None),
            getattr(self, "use_direct_call", None),
        )
        try:
            if not getattr(self, "use_output", False):
                return _call_attention_forward_compat(
                    original_fn, self, query, key, value, output_shape, kwargs
                )
            # Proceed for both use_direct_call=True and use_direct_call=False.
            # For use_direct_call=True (MLU non-OOT + V1, or V0), we still call
            # unified_attention_with_output directly. Our patched version handles
            # the V1 dict-metadata path (NT kernels + _record_hit).  For V0
            # non-dict metadata it falls back to the original impl, which is
            # identical to what Attention_MluHjack.forward would have done.

            layer_mod = importlib.import_module("vllm_mlu.attention.layer")
            output_lse = None
            output_shape = output_shape if output_shape is not None else query.shape
            v_head_dim = getattr(self, "v_head_dim", self.head_size)
            if getattr(self, "use_mla", False):
                output_shape = [output_shape[0], self.num_heads * v_head_dim]
            output = torch.empty(
                output_shape,
                dtype=self.dtype if query.dtype == torch.int8 else query.dtype,
                device=query.device,
            )
            hidden_size = output_shape[-1]
            query_reshaped = query.view(-1, self.num_heads, self.head_size)
            output_reshaped = output.view(-1, self.num_heads, v_head_dim)
            key_reshaped = (
                None
                if key is None
                else key.view(-1, self.num_kv_heads, self.head_size)
            )
            value_reshaped = (
                None
                if value is None
                else value.view(-1, self.num_kv_heads, v_head_dim)
            )

            # Record NT hit directly from forward context so that we don't
            # depend on unified_attention_with_output being our patched version.
            # (The patch on that function may be silently skipped in the
            # EngineCore subprocess if vllm_mlu.attention.layer wasn't
            # importable yet at ensure_registered() time.)
            try:
                forward_context = layer_mod.get_forward_context()
                attn_metadata_raw = getattr(forward_context, "attn_metadata", None)
                if isinstance(attn_metadata_raw, dict):
                    common_meta = attn_metadata_raw.get("common_metadata")
                    if common_meta is not None:
                        num_actual = int(common_meta.num_actual_tokens)
                        if num_actual > 0:
                            is_pf = getattr(common_meta, "is_prefill_only", False) or getattr(
                                common_meta, "is_chunked", False
                            )
                            _record_hit(
                                "PagedAttentionPrefill" if is_pf else "PagedAttentionDecode",
                                query,
                            )
                elif attn_metadata_raw is not None:
                    # V0 metadata object
                    num_prefills = getattr(attn_metadata_raw, "num_prefills", None)
                    if num_prefills is not None and int(num_prefills) > 0:
                        _record_hit("PagedAttentionPrefill", query)
                    else:
                        _record_hit("PagedAttentionDecode", query)
            except Exception as _hit_exc:
                _log_once(
                    "warning",
                    "hit_record_failed:mlu_attention_forward",
                    "vllm-nt: hit recording in Attention.forward failed: %s",
                    _hit_exc,
                )

            attn_output_list = layer_mod.unified_attention_with_output(
                query_reshaped,
                key_reshaped,
                value_reshaped,
                output_reshaped,
                self.layer_name,
                kwargs=kwargs or {},
            )
            if (
                isinstance(attn_output_list, (list, tuple))
                and len(attn_output_list) > 1
            ):
                output_lse = attn_output_list[1]
            if output_lse is not None:
                return output.view(-1, hidden_size), output_lse
            return output.view(-1, hidden_size)
        except Exception as _exc:
            _log_once(
                "warning",
                "fallback:mlu_attention_forward_patch",
                "vllm-nt: patched Attention.forward falling back to original (%s)",
                _exc,
            )
            return _call_attention_forward_compat(
                original_fn, self, query, key, value, output_shape, kwargs
            )

    return _mark_function_patch(forward, "UnifiedAttentionWithOutput")


def _build_flash_attn_varlen_patch(original: object) -> object:
    if _is_function_patch(original, "PagedAttentionPrefill"):
        return original
    original_fn = cast(Callable[..., object], original)

    def wrapped(*args, **kwargs):
        query = args[0] if args else kwargs.get("q")
        if isinstance(query, torch.Tensor):
            _record_hit("PagedAttentionPrefill", query)
        try:
            return paged_attention_prefill(*args, **kwargs)
        except Exception:
            return original_fn(*args, **kwargs)

    return _mark_function_patch(wrapped, "PagedAttentionPrefill")


def _build_vit_flash_attn_backend(original: object) -> object:
    original_fn = cast(Callable[..., object], original)

    def wrapped(*args, **kwargs):
        backend, flash_attn = original_fn(*args, **kwargs)
        if flash_attn is None:
            return backend, flash_attn
        return backend, _build_flash_attn_varlen_patch(flash_attn)

    return wrapped


def _build_rotary_forward_oot(original: object) -> object:
    if _is_function_patch(original, "RoPE"):
        return original
    original_fn = cast(Callable[..., object], original)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ):
        if not self.is_neox_style or self.rotary_dim % 2:
            return original_fn(self, positions, query, key)

        _record_hit("RoPE", query)
        try:
            cos_sin_cache = self._match_cos_sin_cache_dtype(query)
            return rope(
                positions,
                query,
                key,
                cos_sin_cache=cos_sin_cache,
                head_size=self.head_size,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
            )
        except Exception:
            return original_fn(self, positions, query, key)

    return _mark_function_patch(forward_oot, "RoPE")


def _build_mlu_rotary_forward_oot(original: object) -> object:
    if _is_function_patch(original, "RoPE"):
        return original
    original_fn = cast(Callable[..., object], original)

    def forward_oot(
        self,
        positions: torch.Tensor,
        x: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ):
        if x is None or positions.ndim != 1:
            return original_fn(self, positions, x, offsets)

        rotary_dim = min(int(getattr(self, "rotary_dim", x.shape[-1])), x.shape[-1])
        if not getattr(self, "is_neox_style", False) or rotary_dim % 2:
            return original_fn(self, positions, x, offsets)

        _record_hit("RoPE", x)
        try:
            rope_positions = positions if offsets is None else (positions + offsets)
            cos_sin_cache = self._match_cos_sin_cache_dtype(x)
            out, _ = rope(
                rope_positions,
                x,
                None,
                cos_sin_cache=cos_sin_cache,
                head_size=x.shape[-1],
                rotary_dim=rotary_dim,
                is_neox_style=self.is_neox_style,
            )
            return out
        except Exception:
            return original_fn(self, positions, x, offsets)

    return _mark_function_patch(forward_oot, "RoPE")


def _build_mlu_deepseek_rotary_forward_oot(original: object) -> object:
    if _is_function_patch(original, "RoPE"):
        return original
    original_fn = cast(Callable[..., object], original)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        only_prefill: bool = False,
        only_decode: bool = False,
    ):
        base = query if query is not None else key
        if base is None or positions.ndim != 1:
            return original_fn(self, positions, query, key, offsets, only_prefill, only_decode)

        rotary_dim = min(int(getattr(self, "rotary_dim", base.shape[-1])), base.shape[-1])
        if not getattr(self, "is_neox_style", False) or rotary_dim % 2:
            return original_fn(self, positions, query, key, offsets, only_prefill, only_decode)

        _record_hit("RoPE", base)
        try:
            rope_positions = positions if offsets is None else (positions + offsets)
            cos_sin_cache = self._match_cos_sin_cache_dtype(base)
            return rope(
                rope_positions,
                query,
                key,
                cos_sin_cache=cos_sin_cache,
                head_size=base.shape[-1],
                rotary_dim=rotary_dim,
                is_neox_style=self.is_neox_style,
            )
        except Exception:
            return original_fn(self, positions, query, key, offsets, only_prefill, only_decode)

    return _mark_function_patch(forward_oot, "RoPE")


def _build_apply_top_k_top_p_patch(original: object) -> object:
    if _is_function_patch(original, "TopKTopP"):
        return original

    def wrapped(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ):
        if k is not None or p is not None:
            _record_hit("TopKTopP", logits)
        return nt_apply_top_k_top_p(logits, k, p)

    return _mark_function_patch(wrapped, "TopKTopP")


def _build_random_sample_patch(original: object) -> object:
    if _is_function_patch(original, "RandomSample"):
        return original

    def wrapped(probs: torch.Tensor, generators: dict[int, torch.Generator]):
        _record_hit("RandomSample", probs)
        return nt_random_sample(probs, generators)

    return _mark_function_patch(wrapped, "RandomSample")


def _build_rejection_sample_patch(original: object) -> object:
    if _is_function_patch(original, "RejectionSample"):
        return original
    original_fn = cast(Callable[..., object], original)

    def wrapped(*args: object, **kwargs: object):
        target = kwargs.get("target_logits")
        if target is None:
            target = kwargs.get("target_probs")
        if target is None and len(args) >= 6 and isinstance(args[5], torch.Tensor):
            target = args[5]
        if isinstance(target, torch.Tensor):
            _record_hit("RejectionSample", target)
        return original_fn(*args, **kwargs)

    return _mark_function_patch(wrapped, "RejectionSample")


def _apply_min_p_mask(logits: torch.Tensor, min_p: torch.Tensor | None) -> torch.Tensor:
    if min_p is None:
        return logits
    min_p = min_p.to(device=logits.device, dtype=logits.dtype)
    if not torch.any(min_p):
        return logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_probs = torch.amax(probs, dim=-1, keepdim=True)
    adjusted_min_p = max_probs * min_p.unsqueeze(-1)
    logits.masked_fill_(probs < adjusted_min_p, -float("inf"))
    return logits


def _build_mlu_apply_topkp_v2_patch(original: object) -> object:
    if _is_function_patch(original, "TopKTopP"):
        return original

    def wrapped(
        logits: torch.Tensor,
        index_in: torch.Tensor,
        temperature_list: torch.Tensor,
        minp_list: torch.Tensor | None,
        topk_list: torch.Tensor | None,
        topp_list: torch.Tensor | None,
        logits_out: torch.Tensor | None = None,
        sorted_logits_out: torch.Tensor | None = None,
        index_out: torch.Tensor | None = None,
        true_select_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _record_hit("TopKTopP", logits)

        filtered = logits.to(torch.float32).clone()
        temperature = temperature_list.to(device=filtered.device, dtype=filtered.dtype)
        temperature = torch.where(
            temperature.abs() < 1e-5,
            torch.ones_like(temperature),
            temperature,
        )
        filtered.div_(temperature.unsqueeze(-1))
        filtered = _apply_min_p_mask(filtered, minp_list)
        topk = None if topk_list is None else topk_list.to(device=filtered.device)
        topp = None if topp_list is None else topp_list.to(device=filtered.device)
        filtered = nt_apply_top_k_top_p(filtered, topk, topp)

        sorted_logits, sorted_indices = filtered.sort(dim=-1, descending=True)
        expanded_index = index_in.to(device=filtered.device, dtype=torch.int64)
        if expanded_index.ndim == 1:
            expanded_index = expanded_index.unsqueeze(0).expand(filtered.shape[0], -1)
        sorted_index = expanded_index.gather(1, sorted_indices)
        selected = torch.isfinite(filtered).sum(dim=-1)

        logits_res = filtered if logits_out is None else logits_out.copy_(filtered)
        sorted_logits_res = (
            sorted_logits
            if sorted_logits_out is None
            else sorted_logits_out.copy_(sorted_logits)
        )
        index_res = (
            sorted_index
            if index_out is None
            else index_out.copy_(sorted_index.to(dtype=index_out.dtype))
        )
        true_select_len_res = (
            selected
            if true_select_len is None
            else true_select_len.copy_(selected.to(dtype=true_select_len.dtype))
        )
        return logits_res, sorted_logits_res, index_res, true_select_len_res

    return _mark_function_patch(wrapped, "TopKTopP")


def _build_mlu_random_sample_patch(original: object) -> object:
    if _is_function_patch(original, "RandomSample"):
        return original

    def wrapped(probs: torch.Tensor, generators: dict[int, torch.Generator]):
        _record_hit("RandomSample", probs)
        return nt_random_sample(probs, generators)

    return _mark_function_patch(wrapped, "RandomSample")


def _nt_layer_norm(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    _record_hit("LayerNorm", x)
    normalized_shape = getattr(layer, "normalized_shape", x.shape[-1:])
    weight = getattr(layer, "weight", None)
    bias = getattr(layer, "bias", None)
    eps = getattr(layer, "eps", 1e-5)
    return layer_norm(x, normalized_shape, weight, bias, eps)


def _nt_wpe(layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
    _record_hit("WPE", input_)
    _record_hit("NTWPEKernel", input_)
    return nt_wpe(input_, layer.weight)


def _nt_plain_gelu(act: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    _record_hit("GELU", x)
    act_name = type(act).__name__
    if act_name in {"NewGELU", "FastGELU"}:
        return nt_gelu(x)
    if act_name == "QuickGELU":
        return x * torch.sigmoid(1.702 * x)
    if isinstance(act, torch.nn.GELU):
        approximate = getattr(act, "approximate", "none")
        if approximate == "tanh":
            return nt_gelu(x)
        return F.gelu(x, approximate=approximate)
    return act(x)


def _build_gpt2_mlp_forward(original: object) -> object:
    if _is_function_patch(original, "GELU"):
        return original
    original_fn = cast(Callable[..., object], original)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        try:
            hidden_states, _ = self.c_fc(hidden_states)
            hidden_states = _nt_plain_gelu(self.act, hidden_states)
            hidden_states, _ = self.c_proj(hidden_states)
            return hidden_states
        except Exception:
            return original_fn(self, hidden_states)

    return _mark_function_patch(forward, "GELU")


def _build_gpt2_block_forward(original: object) -> object:
    if _is_function_patch(original, "LayerNorm"):
        return original
    original_fn = cast(Callable[..., object], original)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        try:
            residual = hidden_states
            hidden_states = _nt_layer_norm(self.ln_1, hidden_states)
            attn_output = self.attn(hidden_states=hidden_states)
            hidden_states = attn_output + residual

            residual = hidden_states
            hidden_states = _nt_layer_norm(self.ln_2, hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            hidden_states = residual + feed_forward_hidden_states
            return hidden_states
        except Exception:
            return original_fn(self, hidden_states)

    return _mark_function_patch(forward, "LayerNorm")


def _build_gpt2_model_forward(original: object) -> object:
    if _is_function_patch(original, "LayerNorm"):
        return original
    _set_registered_via("WPE", "function_patch")
    _set_registered_via("NTWPEKernel", "function_patch")
    original_fn = cast(Callable[..., object], original)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        intermediate_tensors,
        inputs_embeds: torch.Tensor | None,
    ):
        try:
            gpt2_mod = importlib.import_module("vllm.model_executor.models.gpt2")
            pp_group = gpt2_mod.get_pp_group()
            if pp_group.is_first_rank:
                if inputs_embeds is None:
                    inputs_embeds = self.embed_input_ids(input_ids)
                position_embeds = _nt_wpe(self.wpe, position_ids)
                hidden_states = inputs_embeds + position_embeds
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]

            for layer in gpt2_mod.islice(self.h, self.start_layer, self.end_layer):
                hidden_states = layer(hidden_states)

            if not pp_group.is_last_rank:
                return gpt2_mod.IntermediateTensors({"hidden_states": hidden_states})

            return _nt_layer_norm(self.ln_f, hidden_states)
        except Exception:
            return original_fn(
                self, input_ids, position_ids, intermediate_tensors, inputs_embeds
            )

    return _mark_function_patch(forward, "LayerNorm")


def _build_qwen2_mlp_forward(original: object) -> object:
    if _is_function_patch(original, "SiluAndMul"):
        return original
    original_fn = cast(Callable[..., object], original)

    def forward(self, x: torch.Tensor):
        try:
            gate_up = self.gate_up_proj(x)
            if isinstance(gate_up, tuple):
                gate_up = gate_up[0]
            x = _act("SiluAndMul", gate_up, nt_silu)
            down = self.down_proj(x)
            if isinstance(down, tuple):
                return down[0]
            return down
        except Exception:
            return original_fn(self, x)

    return _mark_function_patch(forward, "SiluAndMul")


def _build_mlu_active_patch(original: object) -> object:
    if _is_function_patch(original, "SiluAndMul"):
        return original
    original_fn = cast(Callable[..., object], original)

    def wrapped(input: torch.Tensor, act_mode: str, is_gated: bool):
        if is_gated:
            if act_mode == "silu":
                _record_hit("SiluAndMul", input)
            elif act_mode == "gelu":
                _record_hit("GeluAndMul", input)
        return original_fn(input, act_mode, is_gated)

    return _mark_function_patch(wrapped, "SiluAndMul")


def _build_sdpa_patch(original: object) -> object:
    if _is_function_patch(original, "SDPA"):
        return original
    original_fn = cast(Callable[..., object], original)

    def wrapped(*args, **kwargs):
        query = args[0] if args else kwargs.get("query")
        if isinstance(query, torch.Tensor):
            _record_hit("SDPA", query)
        try:
            return sdpa(*args, **kwargs)
        except Exception:
            return original_fn(*args, **kwargs)

    return _mark_function_patch(wrapped, "SDPA")


def _mlu_v1_store_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Store kv into MLU V1 paged cache layout: (num_blocks, num_kv_heads, block_size, head_size)."""
    valid = slot_mapping >= 0
    if not valid.any():
        return
    slots = slot_mapping[valid].long()
    block_size = key_cache.shape[2]
    bi = slots // block_size
    wi = slots % block_size
    # key[valid]: (N, num_kv_heads, head_size)
    # key_cache[bi, :, wi, :] assignment: for each i, sets key_cache[bi[i], :, wi[i], :]
    key_cache[bi, :, wi, :] = key[valid]
    value_cache[bi, :, wi, :] = value[valid]


def _mlu_v1_get_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple:
    """Gather kv from MLU V1 paged cache layout: (num_blocks, num_kv_heads, block_size, head_size)."""
    block_size = key_cache.shape[2]
    num_kv_heads = key_cache.shape[1]
    head_size = key_cache.shape[3]
    num_seqs = block_table.shape[0]
    cu_seqlens = F.pad(torch.cumsum(seq_lens.int(), dim=0), (1, 0))
    total = int(cu_seqlens[-1].item())
    if total == 0:
        empty = torch.empty((0, num_kv_heads, head_size), dtype=key_cache.dtype, device=key_cache.device)
        return empty, empty.clone(), cu_seqlens
    k = torch.empty((total, num_kv_heads, head_size), dtype=key_cache.dtype, device=key_cache.device)
    v = torch.empty_like(k)
    for si in range(num_seqs):
        slen = int(seq_lens[si].item())
        if slen <= 0:
            continue
        out_start = int(cu_seqlens[si].item())
        copied = 0
        for bi in range(block_table.shape[1]):
            if copied >= slen:
                break
            phys = int(block_table[si, bi].item())
            if phys < 0:
                break
            take = min(block_size, slen - copied)
            out_slice = slice(out_start + copied, out_start + copied + take)
            # key_cache[phys]: (num_kv_heads, block_size, head_size)
            # [:, :take, :] → (num_kv_heads, take, head_size) → permute → (take, num_kv_heads, head_size)
            k[out_slice] = key_cache[phys, :, :take, :].permute(1, 0, 2)
            v[out_slice] = value_cache[phys, :, :take, :].permute(1, 0, 2)
            copied += take
    return k, v, cu_seqlens


def _build_mlu_flash_attention_impl_forward(original: object) -> object:
    if _is_function_patch(original, "PagedAttentionDecode"):
        return original
    original_fn = cast(Callable[..., object], original)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        output: torch.Tensor | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        # Record hit at entry — proves the FlashAttentionImpl.forward patch is reached.
        _record_hit("PagedAttentionDecode", query)
        if output is None or attn_metadata is None:
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output, kwargs)

        if (
            getattr(attn_metadata, "use_cascade", False)
            or getattr(attn_metadata, "local_attn_metadata", None) is not None
            or getattr(self, "use_mla", False)
            or getattr(self, "kv_sharing_target_layer_name", None) is not None
        ):
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output, kwargs)

        if not isinstance(kv_cache, (tuple, list)) or len(kv_cache) != 2:
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output, kwargs)

        kv_cache_tensors, kv_cache_scale = kv_cache
        if not isinstance(kv_cache_tensors, torch.Tensor) or kv_cache_tensors.shape[0] < 2:
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output, kwargs)
        if isinstance(kv_cache_scale, torch.Tensor) and kv_cache_scale.numel() > 0:
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output, kwargs)

        key_cache = kv_cache_tensors[0]
        value_cache = kv_cache_tensors[1]

        try:
            backend_mod = importlib.import_module(
                "vllm_mlu.v1.attention.backends.flash_attn"
            )
            common_metadata = backend_mod.get_common_metadata()
            num_actual_tokens = int(common_metadata.num_actual_tokens)
            if num_actual_tokens <= 0:
                return output

            _log_once(
                "info",
                "detect:mlu_v1_path_in_flash_attn_impl",
                "vllm-nt: MLU V1 FlashAttentionImpl.forward NT path reached (mode=%s)",
                common_metadata.infer_mode,
            )

            # Record hits BEFORE any kernel calls that might fail on MLU.
            if common_metadata.is_prefill_only or common_metadata.is_chunked:
                _record_hit("PagedAttentionPrefill", query[:num_actual_tokens])
            else:
                _record_hit("PagedAttentionDecode", query[:num_actual_tokens])

            # Store kv cache using MLU V1 layout-aware scatter.
            # Wrapped in inner try/except so a store failure doesn't abort attention.
            try:
                slot_mapping = attn_metadata.slot_mapping.flatten()[:num_actual_tokens]
                _mlu_v1_store_kv(
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    slot_mapping,
                )
            except Exception as _store_exc:
                _log_once(
                    "warning",
                    "fallback:mlu_v1_store_kv",
                    "vllm-nt: MLU V1 kv store failed (%s); falling back to original for full forward",
                    _store_exc,
                )
                return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output, kwargs)

            if common_metadata.is_prefill_only:
                num_prefill_query_tokens = int(common_metadata.num_prefill_query_tokens)
                num_prefill_kv_tokens = int(common_metadata.num_prefill_kv_tokens)
                output_slice = paged_attention_prefill(
                    query[:num_prefill_query_tokens],
                    key[:num_prefill_kv_tokens],
                    value[:num_prefill_kv_tokens],
                    attn_metadata.query_start_loc,
                    attn_metadata.seq_start_loc,
                    attn_metadata.max_query_len,
                    softmax_scale=self.scale,
                    causal=True,
                )
                output[:num_prefill_query_tokens].copy_(output_slice)
                return output

            # Gather kv from paged cache using MLU V1 layout-aware function.
            k_tokens, v_tokens, cu_seqlens_k = _mlu_v1_get_kv(
                key_cache,
                value_cache,
                attn_metadata.seq_lens,
                attn_metadata.block_table,
            )

            if common_metadata.is_chunked:
                output_slice = paged_attention_prefill(
                    query[:num_actual_tokens],
                    k_tokens,
                    v_tokens,
                    attn_metadata.query_start_loc,
                    cu_seqlens_k,
                    attn_metadata.max_query_len,
                    softmax_scale=self.scale,
                    causal=True,
                )
                output[:num_actual_tokens].copy_(output_slice)
                return output

            # Decode-only: one query token per sequence, use varlen attention
            # over gathered kv (avoids the paged-decode layout mismatch).
            batch_size = int(attn_metadata.block_table.shape[0])
            cu_seqlens_q = torch.arange(
                batch_size + 1, device=query.device, dtype=torch.int32
            )
            decode_output = paged_attention_prefill(
                query[:num_actual_tokens],
                k_tokens,
                v_tokens,
                cu_seqlens_q,
                cu_seqlens_k,
                1,
                softmax_scale=self.scale,
                causal=True,
            )
            output[:num_actual_tokens].copy_(decode_output)
            return output
        except Exception as exc:
            _log_once(
                "warning",
                "fallback:mlu_flash_attn_impl_forward",
                "vllm-nt: MLU V1 FlashAttentionImpl.forward NT path failed (%s)",
                exc,
            )
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output, kwargs)

    return _mark_function_patch(forward, "PagedAttentionDecode")


def _build_musa_flash_attention_impl_forward(original: object) -> object:
    if _is_function_patch(original, "PagedAttentionDecode"):
        return original
    original_fn = cast(Callable[..., object], original)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            output is None
            or attn_metadata is None
            or output_scale is not None
            or output_block_scale is not None
        ):
            return original_fn(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if (
            getattr(attn_metadata, "use_cascade", False)
            or getattr(self, "dcp_world_size", 1) > 1
            or getattr(self, "sliding_window", None) is not None
            or getattr(self, "alibi_slopes", None) is not None
            or getattr(self, "sinks", None) is not None
            or getattr(self, "logits_soft_cap", 0) not in (0, None)
        ):
            return original_fn(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if not isinstance(kv_cache, torch.Tensor) or kv_cache.shape[0] < 2:
            return original_fn(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        key_cache, value_cache = kv_cache.unbind(0)
        num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", 0))
        if num_actual_tokens <= 0:
            return output

        try:
            num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0))
            num_prefill_tokens = int(getattr(attn_metadata, "num_prefill_tokens", 0))
            causal = bool(getattr(attn_metadata, "causal", True))

            if num_decode_tokens > 0:
                decode_seq_lens = getattr(attn_metadata, "decode_seq_lens", None)
                decode_block_table = getattr(attn_metadata, "decode_block_table", None)
                if decode_seq_lens is None or decode_block_table is None:
                    return original_fn(
                        self,
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        output_scale,
                        output_block_scale,
                    )

                decode_query = query[:num_decode_tokens].view(
                    -1, 1, self.num_heads, self.head_size
                )
                _record_hit("PagedAttentionDecode", query[:num_decode_tokens])
                decode_output = paged_attention_decode(
                    decode_query,
                    key_cache,
                    value_cache,
                    decode_seq_lens,
                    decode_block_table,
                    softmax_scale=self.scale,
                    causal=causal,
                )
                output[:num_decode_tokens].copy_(
                    decode_output.reshape_as(output[:num_decode_tokens])
                )

            if num_prefill_tokens > 0:
                prefill_query_start_loc = getattr(
                    attn_metadata, "prefill_query_start_loc", None
                )
                prefill_max_seq_len = getattr(
                    attn_metadata, "prefill_max_seq_len", None
                )
                if (
                    prefill_query_start_loc is None
                    or prefill_max_seq_len is None
                ):
                    return original_fn(
                        self,
                        layer,
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                        output,
                        output_scale,
                        output_block_scale,
                    )

                prefill_start = num_decode_tokens
                prefill_end = prefill_start + num_prefill_tokens
                prefill_query = query[prefill_start:prefill_end]
                _record_hit("PagedAttentionPrefill", prefill_query)
                prefill_output = paged_attention_prefill(
                    prefill_query,
                    key[prefill_start:prefill_end],
                    value[prefill_start:prefill_end],
                    prefill_query_start_loc,
                    prefill_query_start_loc,
                    int(prefill_max_seq_len),
                    softmax_scale=self.scale,
                    causal=causal,
                )
                output[prefill_start:prefill_end].copy_(
                    prefill_output.reshape_as(output[prefill_start:prefill_end])
                )

            return output
        except Exception as exc:
            _log_once(
                "warning",
                "fallback:musa_flash_attn_impl_forward",
                "vllm-nt: MUSA FlashAttentionImpl.forward NT path failed (%s)",
                exc,
            )
            return original_fn(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

    return _mark_function_patch(forward, "PagedAttentionDecode")


_OPERATOR_SPECS: dict[str, OperatorSpec] = {
    "RMSNorm": (layernorm.RMSNorm, _nt_rms_norm_forward),
    "SiluAndMul": (activation.SiluAndMul, _nt_silu_and_mul_forward),
}
for name, cls, forward in (
    ("MulAndSilu", getattr(activation, "MulAndSilu", None), _nt_mul_and_silu_forward),
    ("GeluAndMul", getattr(activation, "GeluAndMul", None), _nt_gelu_and_mul_forward),
    (
        "GemmaRMSNorm",
        getattr(layernorm, "GemmaRMSNorm", None),
        _nt_gemma_rms_norm_forward,
    ),
):
    if cls is not None:
        _OPERATOR_SPECS[name] = (cls, forward)
_OPERATOR_STATS = {name: OperatorStats() for name in _OPERATOR_SPECS} | {
    "GELU": OperatorStats(),
    "LayerNorm": OperatorStats(),
    "MatMul": OperatorStats(),
    "Embedding": OperatorStats(),
    "WPE": OperatorStats(),
    "NTWPEKernel": OperatorStats(),
    "LMHead": OperatorStats(),
    "PagedAttentionPrefill": OperatorStats(),
    "PagedAttentionDecode": OperatorStats(),
    "RoPE": OperatorStats(),
    "SDPA": OperatorStats(),
    "TopKTopP": OperatorStats(),
    "RandomSample": OperatorStats(),
    "RejectionSample": OperatorStats(),
}
_FUNCTION_PATCH_SPECS_BASE: tuple[FunctionPatchSpec, ...] = (
    # Must come FIRST: intercept MluHijackObject.apply_hijack before
    # vllm_mlu.attention.layer is imported, so that when the hijack overwrites
    # Attention.forward we immediately re-apply our NT patch on top.
    FunctionPatchSpec(
        patch_id="UnifiedAttentionWithOutput",
        module_path="vllm_mlu.mlu_hijack_utils",
        object_name="MluHijackObject",
        attr_name="apply_hijack",
        required=False,
        builder=_build_mlu_hijack_apply_hijack_intercept,
    ),
    FunctionPatchSpec(
        patch_id="StoreKVCache",
        module_path="vllm.model_executor.layers.attention.attention",
        attr_name="unified_kv_cache_update",
        required=False,
        builder=_build_unified_kv_cache_update,
    ),
    FunctionPatchSpec(
        patch_id="UnifiedAttentionWithOutput",
        module_path="vllm.model_executor.layers.attention.attention",
        attr_name="unified_attention_with_output",
        required=False,
        builder=_build_unified_attention_with_output,
    ),
    FunctionPatchSpec(
        patch_id="UnifiedAttentionWithOutput",
        module_path="vllm_mlu.attention.layer",
        attr_name="unified_attention_with_output",
        required=False,
        builder=_build_mlu_unified_attention_with_output,
    ),
    FunctionPatchSpec(
        patch_id="PagedAttentionPrefill",
        module_path="vllm.attention.layer",
        attr_name="maybe_get_vit_flash_attn_backend",
        required=False,
        builder=_build_vit_flash_attn_backend,
    ),
    FunctionPatchSpec(
        patch_id="PagedAttentionPrefill",
        module_path="vllm.attention.utils.fa_utils",
        attr_name="flash_attn_varlen_func",
        required=False,
        builder=_build_flash_attn_varlen_patch,
    ),
    FunctionPatchSpec(
        patch_id="PagedAttentionPrefill",
        module_path="vllm.v1.attention.backends.flash_attn",
        attr_name="flash_attn_varlen_func",
        required=False,
        builder=_build_flash_attn_varlen_patch,
    ),
    FunctionPatchSpec(
        patch_id="UnifiedAttention2D",
        module_path="vllm.attention.ops.triton_unified_attention",
        attr_name="unified_attention_2d",
        required=False,
        builder=_build_unified_attention_2d,
    ),
    FunctionPatchSpec(
        patch_id="UnifiedAttention2D",
        module_path="vllm.attention.ops.unified_attention",
        attr_name="unified_attention_2d",
        required=False,
        builder=_build_unified_attention_2d,
    ),
    FunctionPatchSpec(
        patch_id="RoPE",
        module_path="vllm.model_executor.layers.rotary_embedding.base",
        object_name="RotaryEmbedding",
        attr_name="forward_oot",
        required=False,
        builder=_build_rotary_forward_oot,
    ),
    FunctionPatchSpec(
        patch_id="RoPE",
        module_path="vllm.model_executor.layers.rotary_embedding",
        object_name="RotaryEmbedding",
        attr_name="forward_oot",
        required=False,
        builder=_build_rotary_forward_oot,
    ),
    FunctionPatchSpec(
        patch_id="RoPE",
        module_path="vllm_mlu.model_executor.layers.rotary_embedding",
        object_name="MLURotaryEmbedding",
        attr_name="forward_oot",
        required=False,
        builder=_build_mlu_rotary_forward_oot,
    ),
    FunctionPatchSpec(
        patch_id="RoPE",
        module_path="vllm_mlu.model_executor.layers.rotary_embedding",
        object_name="MLUDeepseekScalingRotaryEmbedding",
        attr_name="forward_oot",
        required=False,
        builder=_build_mlu_deepseek_rotary_forward_oot,
    ),
    FunctionPatchSpec(
        patch_id="LayerNorm",
        module_path="vllm.model_executor.models.gpt2",
        object_name="GPT2Block",
        attr_name="forward",
        required=False,
        builder=_build_gpt2_block_forward,
    ),
    FunctionPatchSpec(
        patch_id="LayerNorm",
        module_path="vllm.model_executor.models.gpt2",
        object_name="GPT2Model",
        attr_name="forward",
        required=False,
        builder=_build_gpt2_model_forward,
    ),
    FunctionPatchSpec(
        patch_id="GELU",
        module_path="vllm.model_executor.models.gpt2",
        object_name="GPT2MLP",
        attr_name="forward",
        required=False,
        builder=_build_gpt2_mlp_forward,
    ),
    FunctionPatchSpec(
        patch_id="SiluAndMul",
        module_path="vllm.model_executor.models.qwen2",
        object_name="Qwen2MLP",
        attr_name="forward",
        required=False,
        builder=_build_qwen2_mlp_forward,
    ),
    FunctionPatchSpec(
        patch_id="SiluAndMul",
        module_path="vllm.model_executor.models.qwen3",
        object_name="Qwen3MLP",
        attr_name="forward",
        required=False,
        builder=_build_qwen2_mlp_forward,
    ),
    FunctionPatchSpec(
        patch_id="SiluAndMul",
        module_path="vllm_mlu._mlu_ops",
        attr_name="active",
        required=False,
        builder=_build_mlu_active_patch,
    ),
    FunctionPatchSpec(
        patch_id="TopKTopP",
        module_path="vllm.v1.sample.ops.topk_topp_sampler",
        attr_name="apply_top_k_top_p",
        required=False,
        builder=_build_apply_top_k_top_p_patch,
    ),
    FunctionPatchSpec(
        patch_id="RandomSample",
        module_path="vllm.v1.sample.ops.topk_topp_sampler",
        attr_name="random_sample",
        required=False,
        builder=_build_random_sample_patch,
    ),
    FunctionPatchSpec(
        patch_id="RejectionSample",
        module_path="vllm.v1.sample.rejection_sampler",
        attr_name="rejection_sample",
        required=False,
        builder=_build_rejection_sample_patch,
    ),
    FunctionPatchSpec(
        patch_id="TopKTopP",
        module_path="vllm_mlu._mlu_ops",
        attr_name="apply_topkp_v2",
        required=False,
        builder=_build_mlu_apply_topkp_v2_patch,
    ),
    FunctionPatchSpec(
        patch_id="RandomSample",
        module_path="vllm_mlu.v1.sample.sampler",
        attr_name="mlu_random_sample",
        required=False,
        builder=_build_mlu_random_sample_patch,
    ),
    FunctionPatchSpec(
        patch_id="SDPA",
        module_path="vllm.attention.layer",
        object_name="F",
        attr_name="scaled_dot_product_attention",
        required=False,
        builder=_build_sdpa_patch,
    ),
    FunctionPatchSpec(
        patch_id="SDPA",
        module_path="vllm.attention.ops.sdpa",
        attr_name="scaled_dot_product_attention",
        required=False,
        builder=_build_sdpa_patch,
    ),
    FunctionPatchSpec(
        patch_id="PagedAttentionDecode",
        module_path="vllm_mlu.v1.attention.backends.flash_attn",
        object_name="FlashAttentionImpl",
        attr_name="forward",
        required=False,
        builder=_build_mlu_flash_attention_impl_forward,
    ),
    FunctionPatchSpec(
        patch_id="PagedAttentionDecode",
        module_path="vllm_mlu.attention.backends.flash_attn",
        object_name="FlashAttentionImpl",
        attr_name="forward",
        required=False,
        builder=_build_mlu_flash_attention_impl_forward,
    ),
    FunctionPatchSpec(
        patch_id="PagedAttentionDecode",
        module_path="vllm_musa.v1.attention.backends.flash_attn",
        object_name="FlashAttentionImpl",
        attr_name="forward",
        required=False,
        builder=_build_musa_flash_attention_impl_forward,
    ),
    # Patch Attention.forward AFTER vllm_mlu.attention.layer is imported above,
    # because importing that module runs MluHijackObject.apply_hijack which does
    # setattr(Attention, "forward", Attention_MluHjack.forward) — a direct copy.
    # Patching Attention_MluHjack.forward afterward has no effect on Attention.forward.
    # We must overwrite Attention.forward itself, after the hijack has already run.
    FunctionPatchSpec(
        patch_id="UnifiedAttentionWithOutput",
        module_path="vllm.attention.layer",
        object_name="Attention",
        attr_name="forward",
        required=False,
        builder=_build_mlu_attention_forward,
    ),
)

_EXPERIMENTAL_FORWARD_PATCH_SPECS: tuple[FunctionPatchSpec, ...] = (
    FunctionPatchSpec(
        patch_id="UnifiedAttentionWithOutput",
        module_path="vllm.attention.layer",
        object_name="Attention",
        attr_name="forward",
        required=False,
        builder=_build_mlu_attention_forward,
    ),
    FunctionPatchSpec(
        patch_id="UnifiedAttentionWithOutput",
        module_path="vllm_mlu.attention.layer",
        object_name="Attention_MluHjack",
        attr_name="forward",
        required=False,
        builder=_build_mlu_attention_forward,
    ),
)

if os.environ.get("VLLM_NT_ENABLE_EXPERIMENTAL_FORWARD_PATCH") == "1":
    _FUNCTION_PATCH_SPECS = (
        _FUNCTION_PATCH_SPECS_BASE + _EXPERIMENTAL_FORWARD_PATCH_SPECS
    )
else:
    _FUNCTION_PATCH_SPECS = _FUNCTION_PATCH_SPECS_BASE
_summary_printed = False
_registered = False
_APPLIED_FUNCTION_PATCHES: list[_AppliedFunctionPatch] = []
_APPLIED_CUSTOM_OP_REBINDS: list[_AppliedCustomOpRebind] = []
_INSTALLED_CUSTOM_OP_INTERCEPT: _InstalledCustomOpIntercept | None = None


def _try_register_oot() -> bool:
    try:
        for name, (cls, forward) in _OPERATOR_SPECS.items():
            if not _operator_enabled(name):
                continue
            cls.register_oot(name=name)(
                type(f"NT{name}", (cls,), {"forward_oot": forward})
            )
            _set_registered_via(name, "oot")
        logger.info(
            "vllm-nt: OOT registration succeeded for %s", ", ".join(_OPERATOR_SPECS)
        )
        return True
    except Exception as e:
        logger.warning("OOT registration failed (%s), will monkey-patch", e)
        return False


def _monkey_patch() -> None:
    for name, (cls, forward) in _OPERATOR_SPECS.items():
        if not _operator_enabled(name):
            continue
        cls.forward_oot = forward
        cls.forward_native = forward
        _set_registered_via(name, "monkey_patch")
    logger.info("vllm-nt: monkey-patched %s", ", ".join(_OPERATOR_SPECS))


def _nt_unquantized_linear_apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    _record_hit("MatMul", x)
    residual = kwargs.get("residual")
    nt_output = linear(x, layer.weight, bias)
    if residual is not None:
        nt_output = nt_output + residual
    return nt_output


def _nt_unquantized_embedding(
    self, layer: torch.nn.Module, input_: torch.Tensor
) -> torch.Tensor:
    _record_hit("Embedding", input_)
    return embedding(layer, input_)


def _nt_unquantized_embedding_apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    _record_hit("LMHead", x)
    residual = kwargs.get("residual")
    nt_output = linear(x, layer.weight, bias)
    if residual is not None:
        nt_output = nt_output + residual
    return nt_output


def _patch_leaf_methods() -> None:
    if _operator_enabled("MatMul"):
        UnquantizedLinearMethod.apply = _nt_unquantized_linear_apply
        _set_registered_via("MatMul", "monkey_patch")
    if _operator_enabled("Embedding"):
        UnquantizedEmbeddingMethod.embedding = _nt_unquantized_embedding
        _set_registered_via("Embedding", "monkey_patch")
    if _operator_enabled("LMHead"):
        UnquantizedEmbeddingMethod.apply = _nt_unquantized_embedding_apply
        _set_registered_via("LMHead", "monkey_patch")


def _resolve_function_patch_target(spec: FunctionPatchSpec) -> tuple[object, object]:
    module = importlib.import_module(spec.module_path)
    target_obj = getattr(module, spec.object_name) if spec.object_name else module
    original = getattr(target_obj, spec.attr_name)
    return target_obj, original


def _apply_function_patches() -> None:
    global _APPLIED_FUNCTION_PATCHES
    applied: list[_AppliedFunctionPatch] = []
    try:
        for spec in _FUNCTION_PATCH_SPECS:
            if not _function_patch_enabled(spec):
                continue
            try:
                target_obj, original = _resolve_function_patch_target(spec)
                replacement = original if spec.builder is None else spec.builder(original)
                setattr(target_obj, spec.attr_name, replacement)
                applied.append(
                    _AppliedFunctionPatch(
                        spec=spec, target_obj=target_obj, original=original
                    )
                )
                if spec.patch_id in {"UnifiedAttention2D", "UnifiedAttentionWithOutput"}:
                    _set_registered_via("PagedAttentionPrefill", "function_patch")
                    _set_registered_via("PagedAttentionDecode", "function_patch")
                elif spec.patch_id in _OPERATOR_STATS:
                    _set_registered_via(spec.patch_id, "function_patch")
            except Exception as exc:
                if spec.required:
                    raise
                logger.debug("Skipping optional function patch %s: %s", spec.patch_id, exc)
    except Exception:
        for patch in reversed(applied):
            setattr(patch.target_obj, patch.spec.attr_name, patch.original)
        raise
    _APPLIED_FUNCTION_PATCHES = applied


def _build_direct_register_custom_op_intercept(
    original: Callable[..., object],
) -> Callable[..., object]:
    if _is_function_patch(original, "CustomOpRegisterIntercept"):
        return cast(Callable[..., object], original)

    def wrapped(
        op_name: str,
        op_func: Callable[..., object],
        mutates_args: list[str] | None = None,
        fake_impl: Callable[..., object] | None = None,
        target_lib: Library | None = None,
        dispatch_key: str | None = None,
        tags: tuple[torch.Tag, ...] = (),
    ):
        replacement = op_func
        if op_name == "unified_kv_cache_update" and _nt_feature_enabled("FA"):
            replacement = _build_custom_op_unified_kv_cache_update()
        elif op_name == "unified_attention" and _nt_feature_enabled("FA"):
            replacement = _build_custom_op_unified_attention()
            _set_registered_via("PagedAttentionPrefill", "custom_op_intercept")
            _set_registered_via("PagedAttentionDecode", "custom_op_intercept")
        elif op_name == "unified_attention_with_output" and _nt_feature_enabled("FA"):
            replacement = _build_custom_op_unified_attention_with_output()
            _set_registered_via("PagedAttentionPrefill", "custom_op_intercept")
            _set_registered_via("PagedAttentionDecode", "custom_op_intercept")
        return original(
            op_name,
            replacement,
            mutates_args=mutates_args,
            fake_impl=fake_impl,
            target_lib=target_lib,
            dispatch_key=dispatch_key,
            tags=tags,
        )

    return cast(Callable[..., object], _mark_function_patch(wrapped, "CustomOpRegisterIntercept"))


def _install_custom_op_register_intercept() -> None:
    global _INSTALLED_CUSTOM_OP_INTERCEPT
    if os.environ.get("VLLM_NT_ENABLE_CUSTOM_OP_REGISTER_INTERCEPT") != "1":
        return
    if _INSTALLED_CUSTOM_OP_INTERCEPT is not None:
        return

    try:
        torch_utils = importlib.import_module("vllm.utils.torch_utils")
        original = cast(Callable[..., object], torch_utils.direct_register_custom_op)
        wrapped = _build_direct_register_custom_op_intercept(original)
        torch_utils.direct_register_custom_op = wrapped
        _INSTALLED_CUSTOM_OP_INTERCEPT = _InstalledCustomOpIntercept(
            original=original, wrapped=wrapped
        )
        if "vllm.model_executor.layers.attention.attention" in sys.modules:
            logger.warning(
                "vllm-nt: custom op register intercept installed after attention module import; it may be too late for unified_attention_with_output"
            )
    except Exception as exc:
        logger.warning("vllm-nt: custom op register intercept skipped (%s)", exc)


def _rebind_custom_op(
    op_name: str,
    op_func: Callable[..., object],
    dispatch_key: str,
) -> _AppliedCustomOpRebind:
    lib = Library("vllm", "IMPL")
    lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    return _AppliedCustomOpRebind(op_name=op_name, dispatch_key=dispatch_key, library=lib)


def _apply_custom_op_rebindings() -> None:
    global _APPLIED_CUSTOM_OP_REBINDS
    if os.environ.get("VLLM_NT_ENABLE_CUSTOM_OP_REBIND") != "1":
        return
    if not _nt_feature_enabled("FA"):
        return

    applied: list[_AppliedCustomOpRebind] = []
    try:
        from vllm.platforms import current_platform

        dispatch_key = current_platform.dispatch_key
        applied.append(
            _rebind_custom_op(
                "unified_kv_cache_update",
                _build_custom_op_unified_kv_cache_update(),
                dispatch_key,
            )
        )
        applied.append(
            _rebind_custom_op(
                "unified_attention_with_output",
                _build_custom_op_unified_attention_with_output(),
                dispatch_key,
            )
        )
        _set_registered_via("PagedAttentionPrefill", "custom_op_rebind")
        _set_registered_via("PagedAttentionDecode", "custom_op_rebind")
        logger.info(
            "vllm-nt: rebound custom ops for dispatch key %s: %s",
            dispatch_key,
            ", ".join(rebind.op_name for rebind in applied),
        )
    except Exception as exc:
        logger.warning("vllm-nt: custom op rebinding skipped (%s)", exc)
        return

    _APPLIED_CUSTOM_OP_REBINDS = applied


def get_usage_summary() -> dict[str, object]:
    operators = {
        name: {"hits": stats.hits, "registered_via": stats.registered_via}
        for name, stats in _OPERATOR_STATS.items()
    }
    hit_ops = [
        name
        for name, stats in operators.items()
        if cast(dict[str, Any], stats)["hits"] > 0
    ]
    return {
        "registered_ops": list(_OPERATOR_STATS),
        "hit_ops": hit_ops,
        "missed_ops": [name for name in _OPERATOR_STATS if name not in hit_ops],
        "operators": operators,
    }


def format_usage_summary(use_color: bool = True) -> str:
    summary = get_usage_summary()
    colors = {
        "blue": "\033[94m" if use_color else "",
        "reset": "\033[0m" if use_color else "",
    }
    lines = [f"{colors['blue']}Operator usage summary{colors['reset']}"]
    for name, stats in cast(dict[str, dict[str, Any]], summary["operators"]).items():
        lines.append(
            f"{colors['blue']}- {name}: hits={stats['hits']} ({stats['registered_via'] or 'unregistered'}){colors['reset']}"
        )
    lines.append(
        f"{colors['blue']}Missed operators: {', '.join(cast(list[str], summary['missed_ops'])) or 'None'}{colors['reset']}"
    )
    return "\n".join(lines)


def maybe_print_usage_summary(*, include_empty: bool = False) -> bool:
    global _summary_printed
    summary = get_usage_summary()
    if _summary_printed or (not include_empty and not summary["hit_ops"]):
        return False
    print(format_usage_summary(), file=sys.stderr)
    _summary_printed = True
    return True


def _reset_usage_state() -> None:
    global _summary_printed
    for stats in _OPERATOR_STATS.values():
        stats.hits = 0
        stats.logged = False
    _summary_printed = False


def _print_worker_summary_on_exit() -> None:
    if os.getpid() != int(os.environ.get(_PARENT_PID_ENV, os.getpid())):
        maybe_print_usage_summary(include_empty=True)


atexit.register(_print_worker_summary_on_exit)


def ensure_registered() -> None:
    global _registered
    if _registered:
        return
    if not _nt_feature_enabled():
        logger.info("vllm-nt: registration skipped because VLLM_NT_ENABLE_ALL is disabled")
        _registered = True
        return
    _registered = True
    _install_custom_op_register_intercept()
    if not _try_register_oot():
        _monkey_patch()
    _patch_leaf_methods()
    _apply_function_patches()
    _apply_custom_op_rebindings()
