import math

import torch

from vllm_nt._ntops.kernels import sdpa as sdpa_kernel
from vllm_nt._ntops.torch.utils import _cached_make

CausalVariant = sdpa_kernel.CausalVariant


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    causal_variant=None,
    present_key=None,
    present_value=None,
    present_key_slot=None,
    present_value_slot=None,
):
    assert dropout_p == 0, "`dropout_p` is not supported yet."
    assert attn_mask is None or not is_causal, (
        "Cannot use `attn_mask` and `is_causal` together."
    )

    num_heads_q = query.shape[-3]
    num_heads_kv = key.shape[-3]
    assert num_heads_kv == value.shape[-3], (
        "Number of heads in key and value must be the same."
    )
    if not enable_gqa:
        assert num_heads_q == num_heads_kv, (
            "Number of heads in query, key, and value must match when GQA is disabled."
        )
    else:
        assert num_heads_q % num_heads_kv == 0, (
            "Number of query heads must be divisible by key/value heads when GQA is enabled."
        )

    mask_shape = query.shape[:-1] + (key.shape[-2],)
    if attn_mask is not None:
        with_attn_mask = True
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.where(attn_mask, 0, float("-inf"))
        attn_mask = attn_mask.expand(mask_shape)
    else:
        with_attn_mask = False
        attn_mask = torch.empty(mask_shape, device="meta")

    if scale is None:
        scale = 1 / math.sqrt(query.shape[-1])
    if causal_variant is None:
        causal_variant = CausalVariant.UPPER_LEFT

    with_kv_cache = present_key is not None
    output = torch.empty_like(query, dtype=value.dtype)
    kernel = _cached_make(sdpa_kernel.premake, with_kv_cache)

    if with_kv_cache:
        kernel(
            query,
            key,
            value,
            present_key,
            present_value,
            present_key_slot,
            present_value_slot,
            attn_mask,
            is_causal,
            scale,
            output,
            with_attn_mask,
            int(causal_variant),
        )
    else:
        kernel(
            query,
            key,
            value,
            attn_mask,
            is_causal,
            scale,
            output,
            with_attn_mask,
            int(causal_variant),
        )

    return output
