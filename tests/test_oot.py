"""Tests for NineToothed OOT layer overrides."""

import pytest
import torch
import torch.nn.functional as F


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class TestNTRMSNorm:
    """Test NTRMSNorm forward_oot matches reference implementation."""

    def _reference_rms_norm(self, x, weight, eps):
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + eps)
        return (x_normed * weight.float()).to(x.dtype)

    @pytest.mark.parametrize("shape", [(1, 4096), (32, 4096), (1, 32, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_no_residual(self, shape, dtype):
        _skip_if_no_cuda()
        from vllm_nt._ntops.torch import rms_norm

        hidden_size = shape[-1]
        x = torch.randn(shape, dtype=dtype, device="cuda")
        weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
        eps = 1e-6

        output = rms_norm(x, normalized_shape=hidden_size, weight=weight, eps=eps)
        reference = self._reference_rms_norm(x, weight, eps)

        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)

    @pytest.mark.parametrize("shape", [(1, 4096), (32, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_with_residual(self, shape, dtype):
        _skip_if_no_cuda()
        from vllm_nt._ntops.torch import rms_norm

        hidden_size = shape[-1]
        x = torch.randn(shape, dtype=dtype, device="cuda")
        residual = torch.randn(shape, dtype=dtype, device="cuda")
        weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
        eps = 1e-6

        # Simulate the fused add + rms_norm
        x_combined = x + residual
        expected_residual = x_combined.to(dtype)
        expected_output = self._reference_rms_norm(x_combined, weight, eps)

        output = rms_norm(
            x_combined, normalized_shape=hidden_size, weight=weight, eps=eps
        )

        torch.testing.assert_close(output, expected_output, atol=0.01, rtol=0.01)


class TestNTSiluAndMul:
    """Test NTSiluAndMul forward_oot matches reference implementation."""

    def _reference_silu_and_mul(self, x):
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    @pytest.mark.parametrize("shape", [(1, 8192), (32, 8192), (1, 32, 8192)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, shape, dtype):
        _skip_if_no_cuda()
        from vllm_nt._ntops.torch import silu

        x = torch.randn(shape, dtype=dtype, device="cuda")
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]

        output = silu(gate) * up
        reference = self._reference_silu_and_mul(x)

        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
