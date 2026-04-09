from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from vllm_nt._ntops.torch.attention import (
    flash_attn_varlen_func as nt_flash_attn_varlen_func,
)
from vllm_nt._ntops.torch.attention import (
    flash_attn_with_kvcache as nt_flash_attn_with_kvcache,
)
from vllm_nt._ntops.torch.kv_cache import store_kvcache as nt_store_kvcache
from vllm_nt._ntops.torch.kv_cache import get_kv_from_cache as nt_get_kv_from_cache
from vllm_nt._ntops.torch import linear as nt_linear
from vllm_nt._ntops.torch import layer_norm as nt_layer_norm
from vllm_nt._ntops.torch.rotary_emb import apply_rotary_emb as nt_apply_rotary_emb
from vllm_nt._ntops.torch import rms_norm as nt_rms_norm
from vllm_nt._ntops.torch.sdpa import (
    scaled_dot_product_attention as nt_scaled_dot_product_attention,
)


@dataclass
class OperatorStats:
    hits: int = 0
    logged: bool = False
    registered_via: str | None = None


def norm(
    self,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    gemma: bool = False,
):
    if residual is not None:
        x = x + residual
        residual = x if gemma else x.to(x.dtype)
    out = nt_rms_norm(
        x,
        normalized_shape=weight.shape[0] if weight is not None else self.hidden_size,
        weight=weight,
        eps=self.variance_epsilon,
    )
    return (out, residual) if residual is not None else out


def act_and_mul(
    x: torch.Tensor, act: Callable[[torch.Tensor], torch.Tensor], reverse: bool = False
) -> torch.Tensor:
    d = x.shape[-1] // 2
    left, right = (x[..., d:], x[..., :d]) if reverse else (x[..., :d], x[..., d:])
    return act(left) * right


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    output = nt_linear(x.reshape(-1, x.shape[-1]), weight, bias).reshape(
        *x.shape[:-1], weight.shape[0]
    )
    return output


def embedding(layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
    return F.embedding(input_, layer.weight)


def layer_norm(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return nt_layer_norm(
        input,
        normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )


def paged_attention_prefill(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    softmax_scale=None,
    causal=True,
    **kwargs,
):
    return nt_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        **kwargs,
    )


def paged_attention_decode(
    q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale=None, causal=True
):
    return nt_flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        block_table,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def store_kv_cache(key, value, k_cache, v_cache, slot_mapping):
    return nt_store_kvcache(key, value, k_cache, v_cache, slot_mapping)


def get_kv_from_cache(k_cache, v_cache, seq_lens, block_table):
    return nt_get_kv_from_cache(k_cache, v_cache, seq_lens, block_table)


def _apply_rotary(
    x: torch.Tensor,
    *,
    num_tokens: int,
    head_size: int,
    rotary_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    if not is_neox_style:
        raise NotImplementedError("Only NeoX-style RoPE is supported.")
    if rotary_dim % 2:
        raise ValueError("rotary_dim must be even.")

    x_view = x.view(num_tokens, -1, head_size)
    x_rot = x_view[..., :rotary_dim]
    rotated = nt_apply_rotary_emb(x_rot, cos, sin)

    if rotary_dim == head_size:
        return rotated.reshape_as(x)

    x_pass = x_view[..., rotary_dim:]
    return torch.cat((rotated, x_pass), dim=-1).reshape_as(x)


def rope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    *,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_out = _apply_rotary(
        query,
        num_tokens=num_tokens,
        head_size=head_size,
        rotary_dim=rotary_dim,
        cos=cos,
        sin=sin,
        is_neox_style=is_neox_style,
    )
    query.copy_(query_out)

    if key is not None:
        key_out = _apply_rotary(
            key,
            num_tokens=num_tokens,
            head_size=head_size,
            rotary_dim=rotary_dim,
            cos=cos,
            sin=sin,
            is_neox_style=is_neox_style,
        )
        key.copy_(key_out)

    return query, key


def sdpa(*args, **kwargs):
    return nt_scaled_dot_product_attention(*args, **kwargs)
