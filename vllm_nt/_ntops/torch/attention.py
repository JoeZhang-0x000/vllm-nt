import math

import torch

from vllm_nt._ntops.kernels import attention as attention_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def flash_attn_varlen_func(
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
    del max_seqlen_q, kwargs
    q_jag = torch.nested.nested_tensor_from_jagged(q, cu_seqlens_q, jagged_dim=1)
    k_jag = torch.nested.nested_tensor_from_jagged(k, cu_seqlens_k, jagged_dim=1)
    v_jag = torch.nested.nested_tensor_from_jagged(v, cu_seqlens_k, jagged_dim=1)

    output = torch.empty_like(q_jag)
    num_heads_q = q_jag.shape[2]
    num_heads_kv = k_jag.shape[2]
    assert num_heads_q % num_heads_kv == 0, (
        "Number of heads in query must be divisible by key/value heads when GQA is enabled."
    )
    queries_per_group = num_heads_q // num_heads_kv

    q_jag = q_jag.view(
        q_jag.shape[0],
        q_jag.shape[1],
        q_jag.shape[2] // queries_per_group,
        queries_per_group,
        q_jag.shape[3],
    )
    output = output.view(q_jag.shape)
    k_jag = k_jag.view(k_jag.shape[0], k_jag.shape[1], k_jag.shape[2], 1, k_jag.shape[3])
    v_jag = v_jag.view(k_jag.shape)
    softmax_scale = attention_kernel.reference_scale(q_jag, softmax_scale)
    _cached_make(attention_kernel.varlen_premake, max_num_configs=2)(
        q_jag, k_jag, v_jag, output, softmax_scale, causal
    )
    out_shape = output._values.shape
    return output._values.view(out_shape[0], -1, out_shape[-1])


def flash_attn_with_kvcache(
    q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale=None, causal=True
):
    softmax_scale = 1 / math.sqrt(q.shape[-1]) if softmax_scale is None else softmax_scale
    output = torch.empty_like(q)
    replication = q.shape[2] // k_cache.shape[2]
    k_cache = (
        k_cache.unsqueeze(3)
        .expand((-1, -1, -1, replication, -1))
        .reshape(k_cache.shape[0], k_cache.shape[1], -1, k_cache.shape[-1])
    )
    v_cache = (
        v_cache.unsqueeze(3)
        .expand((-1, -1, -1, replication, -1))
        .reshape(k_cache.shape[0], k_cache.shape[1], -1, k_cache.shape[-1])
    )
    _cached_make(attention_kernel.cache_premake)(
        q,
        k_cache,
        v_cache,
        output,
        cache_seqlens.unsqueeze(1).unsqueeze(2).unsqueeze(3),
        block_table.unsqueeze(2).unsqueeze(3),
        softmax_scale,
        causal,
    )
    return output
