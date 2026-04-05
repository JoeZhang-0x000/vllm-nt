import torch

from vllm_nt._ntops.kernels import kv_cache as kv_cache_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    num_tokens, num_heads, head_dim = key.shape
    hidden_size = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == hidden_size and v_cache.stride(1) == hidden_size
    assert slot_mapping.numel() == num_tokens

    _cached_make(kv_cache_kernel.premake)(
        key.view(-1, hidden_size),
        value.view(-1, hidden_size),
        k_cache.view(-1, hidden_size),
        v_cache.view(-1, hidden_size),
        slot_mapping,
    )


def get_kv_from_cache(k_cache, v_cache, seq_lens, block_table):
    num_seqs = int(block_table.shape[0])
    block_size = int(k_cache.shape[1])
    num_heads = int(k_cache.shape[2])
    head_dim = int(k_cache.shape[3])

    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
    )
    total_tokens = int(cu_seqlens[-1].item())
    if total_tokens == 0:
        empty = torch.empty(
            (0, num_heads, head_dim), dtype=k_cache.dtype, device=k_cache.device
        )
        return empty, empty.clone(), cu_seqlens

    k = torch.empty(
        (total_tokens, num_heads, head_dim), dtype=k_cache.dtype, device=k_cache.device
    )
    v = torch.empty_like(k)

    for seq_idx in range(num_seqs):
        seq_len = int(seq_lens[seq_idx].item())
        if seq_len <= 0:
            continue
        out_start = int(cu_seqlens[seq_idx].item())
        copied = 0
        for logical_block_idx in range(int(block_table.shape[1])):
            if copied >= seq_len:
                break
            physical_block_id = int(block_table[seq_idx, logical_block_idx].item())
            if physical_block_id < 0:
                break
            take = min(block_size, seq_len - copied)
            out_slice = slice(out_start + copied, out_start + copied + take)
            k[out_slice] = k_cache[physical_block_id, :take]
            v[out_slice] = v_cache[physical_block_id, :take]
            copied += take

    return k, v, cu_seqlens
