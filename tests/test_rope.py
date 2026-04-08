import pytest

torch = pytest.importorskip("torch")


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


def _reference_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class TestNTRoPE:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_apply_rotary_emb_matches_reference(self, dtype):
        device = _get_device()
        from vllm_nt._ntops.torch.rotary_emb import apply_rotary_emb

        x = torch.randn((5, 4, 8), dtype=dtype, device=device)
        cos = torch.randn((5, 1, 4), dtype=dtype, device=device)
        sin = torch.randn((5, 1, 4), dtype=dtype, device=device)

        output = apply_rotary_emb(x.clone(), cos, sin)
        reference = _reference_rope(x, cos.expand(-1, x.shape[1], -1), sin.expand(-1, x.shape[1], -1))

        torch.testing.assert_close(output, reference, atol=0.05, rtol=0.05)
