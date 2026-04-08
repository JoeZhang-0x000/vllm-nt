import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_mlu  # noqa: F401

        if torch.mlu.is_available():
            return "mlu"
    except ImportError:
        pass
    pytest.skip("No accelerator device available (need CUDA or MLU)")


def _reference_varlen(q, k, v, cu_q, cu_k):
    outputs = []
    batch = cu_q.numel() - 1
    for i in range(batch):
        q_i = q[cu_q[i] : cu_q[i + 1]].transpose(0, 1).unsqueeze(0)
        k_i = k[cu_k[i] : cu_k[i + 1]].transpose(0, 1).unsqueeze(0)
        v_i = v[cu_k[i] : cu_k[i + 1]].transpose(0, 1).unsqueeze(0)
        out_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i, is_causal=True, dropout_p=0
        )
        outputs.append(out_i.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _gather_cache_tokens(k_cache, v_cache, block_table, seqlen):
    block_size = k_cache.shape[1]
    k_tokens = []
    v_tokens = []
    remaining = int(seqlen)
    for block in block_table.tolist():
        if remaining <= 0:
            break
        take = min(block_size, remaining)
        k_tokens.append(k_cache[block, :take])
        v_tokens.append(v_cache[block, :take])
        remaining -= take
    return torch.cat(k_tokens, dim=0), torch.cat(v_tokens, dim=0)


class TestNTAttention:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_varlen_matches_reference(self, dtype):
        device = _get_device()
        from vllm_nt._ntops.torch.attention import flash_attn_varlen_func

        q = torch.randn((5, 2, 8), dtype=dtype, device=device)
        k = torch.randn((5, 2, 8), dtype=dtype, device=device)
        v = torch.randn((5, 2, 8), dtype=dtype, device=device)
        cu_q = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
        cu_k = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)

        output = flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, max_seqlen_q=3, softmax_scale=None, causal=True
        )
        reference = _reference_varlen(q, k, v, cu_q, cu_k)

        torch.testing.assert_close(output, reference, atol=0.05, rtol=0.05)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_decode_with_kvcache_matches_reference(self, dtype):
        device = _get_device()
        from vllm_nt._ntops.torch.attention import flash_attn_with_kvcache

        q = torch.randn((2, 1, 2, 8), dtype=dtype, device=device)
        k_cache = torch.randn((4, 2, 2, 8), dtype=dtype, device=device)
        v_cache = torch.randn((4, 2, 2, 8), dtype=dtype, device=device)
        cache_seqlens = torch.tensor([3, 4], dtype=torch.int32, device=device)
        block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)

        output = flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens, block_table, causal=True
        )

        references = []
        for batch_idx in range(q.shape[0]):
            k_tokens, v_tokens = _gather_cache_tokens(
                k_cache, v_cache, block_table[batch_idx], cache_seqlens[batch_idx]
            )
            q_i = q[batch_idx].transpose(0, 1).unsqueeze(0)
            k_i = k_tokens.transpose(0, 1).unsqueeze(0)
            v_i = v_tokens.transpose(0, 1).unsqueeze(0)
            ref = F.scaled_dot_product_attention(
                q_i, k_i, v_i, is_causal=False, dropout_p=0
            )
            references.append(ref.squeeze(0).transpose(0, 1))
        reference = torch.stack(references, dim=0)

        torch.testing.assert_close(output, reference, atol=0.05, rtol=0.05)
