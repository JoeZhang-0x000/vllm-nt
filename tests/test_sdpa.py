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
    try:
        import torch_musa  # noqa: F401

        if torch.musa.is_available():
            return "musa"
    except ImportError:
        pass
    pytest.skip("No accelerator device available (need CUDA, MLU, or MUSA)")


class TestNTSDPA:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdpa_without_kv_cache_matches_reference(self, dtype):
        device = _get_device()
        from vllm_nt._ntops.torch.sdpa import scaled_dot_product_attention

        query = torch.randn((1, 2, 4, 8), dtype=dtype, device=device)
        key = torch.randn((1, 2, 4, 8), dtype=dtype, device=device)
        value = torch.randn((1, 2, 4, 8), dtype=dtype, device=device)

        output = scaled_dot_product_attention(query, key, value, is_causal=True)
        reference = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, dropout_p=0
        )

        torch.testing.assert_close(output, reference, atol=0.05, rtol=0.05)
