import atexit
import importlib
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
from vllm.model_executor.layers import activation, layernorm
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_nt._ntops.oot_support import (
    OperatorStats,
    act_and_mul,
    embedding,
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

logger = logging.getLogger("vllm_nt")
_PARENT_PID_ENV = "VLLM_NT_PARENT_PID"
os.environ.setdefault(_PARENT_PID_ENV, str(os.getpid()))
OperatorSpec = tuple[type, Callable[..., object]]


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


def _mark_function_patch(fn: object, patch_id: str) -> object:
    try:
        setattr(fn, "_vllm_nt_patch_id", patch_id)
    except Exception:
        pass
    return fn


def _is_function_patch(fn: object, patch_id: str) -> bool:
    return getattr(fn, "_vllm_nt_patch_id", None) == patch_id


def _record_hit(name: str, x: torch.Tensor) -> None:
    stats = _OPERATOR_STATS[name]
    stats.hits += 1
    if not stats.logged:
        logger.info("vllm-nt: ninetoothed %s kernel invoked (shape=%s)", name, x.shape)
        stats.logged = True


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
                return original_fn(query, key, value, output, layer_name, kwargs=kwargs)

            key_cache, value_cache = caches
            scale = getattr(self.impl, "scale", 1 / (query.shape[-1] ** 0.5))

            if (
                isinstance(key, torch.Tensor)
                and isinstance(value, torch.Tensor)
                and key.numel() > 0
                and value.numel() > 0
                and getattr(attn_metadata, "slot_mapping", None) is not None
            ):
                store_kv_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping.flatten(),
                )

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
                    k_tokens, v_tokens, cu_seqlens_k = get_kv_from_cache(
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
                    if int(decode_meta.max_decode_query_len) > 1:
                        decode_seq_lens = torch.as_tensor(
                            decode_meta.seq_lens_tensor,
                            device=query.device,
                            dtype=torch.int32,
                        )
                        k_tokens, v_tokens, cu_seqlens_k = get_kv_from_cache(
                            key_cache,
                            value_cache,
                            decode_seq_lens,
                            decode_meta.block_tables,
                        )
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
                    else:
                        batch_size = int(decode_meta.block_tables.shape[0])
                        decode_query_view = decode_query.view(
                            batch_size, -1, query.shape[1], query.shape[2]
                        )
                        (
                            seq_lens_arg,
                            _max_context_len,
                            block_tables_arg,
                        ) = importlib.import_module(
                            "vllm_mlu.attention.backends.flash_attn"
                        ).get_seq_len_block_table_args(
                            decode_meta, False, self.impl.attn_type
                        )
                        decode_seq_lens = torch.as_tensor(
                            seq_lens_arg, device=query.device, dtype=torch.int32
                        )
                        decode_out = paged_attention_decode(
                            decode_query_view,
                            key_cache,
                            value_cache,
                            decode_seq_lens,
                            block_tables_arg,
                            softmax_scale=scale,
                            causal=True,
                        )
                        output[num_prefill_query_tokens:].copy_(
                            decode_out.reshape_as(output[num_prefill_query_tokens:])
                        )

            layer_mod.maybe_save_kv_layer_to_connector(layer_name, kv_cache)
            return output
        except Exception:
            return original_fn(query, key, value, output, layer_name, kwargs=kwargs)

    return _mark_function_patch(unified_attention_with_output, "UnifiedAttentionWithOutput")


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
        del layer
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

            slot_mapping = attn_metadata.slot_mapping.flatten()[:num_actual_tokens]
            store_kv_cache(
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                key_cache,
                value_cache,
                slot_mapping,
            )

            if common_metadata.is_prefill_only:
                _record_hit("PagedAttentionPrefill", query[:num_actual_tokens])
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

            if common_metadata.is_chunked:
                _record_hit("PagedAttentionPrefill", query[:num_actual_tokens])
                k_tokens, v_tokens, cu_seqlens_k = get_kv_from_cache(
                    key_cache,
                    value_cache,
                    attn_metadata.seq_lens,
                    attn_metadata.block_table,
                )
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

            _record_hit("PagedAttentionDecode", query[:num_actual_tokens])
            batch_size = int(attn_metadata.block_table.shape[0])
            decode_query = query[:num_actual_tokens].view(
                batch_size, -1, self.num_heads, self.head_size
            )
            decode_output = paged_attention_decode(
                decode_query,
                key_cache,
                value_cache,
                attn_metadata.seq_lens,
                attn_metadata.block_table,
                softmax_scale=self.scale,
                causal=True,
            )
            output[:num_actual_tokens].copy_(
                decode_output.reshape_as(output[:num_actual_tokens])
            )
            return output
        except Exception:
            return original_fn(self, layer, query, key, value, kv_cache, attn_metadata, output, kwargs)

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
    "MatMul": OperatorStats(),
    "Embedding": OperatorStats(),
    "PagedAttentionPrefill": OperatorStats(),
    "PagedAttentionDecode": OperatorStats(),
    "RoPE": OperatorStats(),
    "SDPA": OperatorStats(),
}
_FUNCTION_PATCH_SPECS: tuple[FunctionPatchSpec, ...] = (
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
)
_summary_printed = False
_registered = False
_APPLIED_FUNCTION_PATCHES: list[_AppliedFunctionPatch] = []


def _try_register_oot() -> bool:
    try:
        for name, (cls, forward) in _OPERATOR_SPECS.items():
            cls.register_oot(name=name)(
                type(f"NT{name}", (cls,), {"forward_oot": forward})
            )
            _OPERATOR_STATS[name].registered_via = "oot"
        logger.info(
            "vllm-nt: OOT registration succeeded for %s", ", ".join(_OPERATOR_SPECS)
        )
        return True
    except Exception as e:
        logger.warning("OOT registration failed (%s), will monkey-patch", e)
        return False


def _monkey_patch() -> None:
    for name, (cls, forward) in _OPERATOR_SPECS.items():
        cls.forward_oot = forward
        cls.forward_native = forward
        _OPERATOR_STATS[name].registered_via = "monkey_patch"
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


def _patch_leaf_methods() -> None:
    UnquantizedLinearMethod.apply = _nt_unquantized_linear_apply
    _OPERATOR_STATS["MatMul"].registered_via = "monkey_patch"
    UnquantizedEmbeddingMethod.embedding = _nt_unquantized_embedding
    _OPERATOR_STATS["Embedding"].registered_via = "monkey_patch"


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
                    _OPERATOR_STATS["PagedAttentionPrefill"].registered_via = (
                        "function_patch"
                    )
                    _OPERATOR_STATS["PagedAttentionDecode"].registered_via = (
                        "function_patch"
                    )
                elif spec.patch_id in _OPERATOR_STATS:
                    _OPERATOR_STATS[spec.patch_id].registered_via = "function_patch"
            except Exception as exc:
                if spec.required:
                    raise
                logger.debug("Skipping optional function patch %s: %s", spec.patch_id, exc)
    except Exception:
        for patch in reversed(applied):
            setattr(patch.target_obj, patch.spec.attr_name, patch.original)
        raise
    _APPLIED_FUNCTION_PATCHES = applied


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
    _registered = True
    if not _try_register_oot():
        _monkey_patch()
    _patch_leaf_methods()
    _apply_function_patches()
