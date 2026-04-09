"""Tests for NineToothed OOT layer overrides."""

import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F


def _get_device():
    """Return the first available accelerator device, or skip."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_mlu  # noqa: F401

        if torch.mlu.is_available():
            return "mlu"
    except ImportError:
        pass
    pytest.skip("No accelerator device available (need CUDA or MLU)")


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
        device = _get_device()
        from vllm_nt._ntops.torch import rms_norm

        hidden_size = shape[-1]
        x = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(hidden_size, dtype=dtype, device=device)
        eps = 1e-6

        output = rms_norm(x, normalized_shape=hidden_size, weight=weight, eps=eps)
        reference = self._reference_rms_norm(x, weight, eps)

        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)

    @pytest.mark.parametrize("shape", [(1, 4096), (32, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_with_residual(self, shape, dtype):
        device = _get_device()
        from vllm_nt._ntops.torch import rms_norm

        hidden_size = shape[-1]
        x = torch.randn(shape, dtype=dtype, device=device)
        residual = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(hidden_size, dtype=dtype, device=device)
        eps = 1e-6

        x_combined = x + residual
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
        device = _get_device()
        from vllm_nt._ntops.torch import silu

        x = torch.randn(shape, dtype=dtype, device=device)
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]

        output = silu(gate) * up
        reference = self._reference_silu_and_mul(x)

        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


class TestNTGelu:
    def _reference_gelu(self, x):
        return F.gelu(x, approximate="tanh")

    @pytest.mark.parametrize("shape", [(1, 4096), (32, 4096), (1, 32, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward(self, shape, dtype):
        device = _get_device()
        from vllm_nt._ntops.torch import gelu

        x = torch.randn(shape, dtype=dtype, device=device)
        output = gelu(x)
        reference = self._reference_gelu(x)

        torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)


class TestNTMatMul:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, dtype):
        device = _get_device()
        from vllm_nt._ntops.torch import matmul

        lhs = torch.randn((64, 128), dtype=dtype, device=device)
        rhs = torch.randn((128, 32), dtype=dtype, device=device)
        output = matmul(lhs, rhs)
        reference = torch.matmul(lhs, rhs)

        torch.testing.assert_close(output, reference, atol=0.05, rtol=0.05)


class TestNTEmbedding:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, dtype):
        device = _get_device()
        from vllm_nt._ntops.torch import embedding

        weight = torch.randn((32, 16), dtype=dtype, device=device)
        input_ids = torch.tensor([[0, 3, 7], [4, 1, 2]], device=device)
        output = embedding(input_ids, weight)
        reference = F.embedding(input_ids, weight)

        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)


class TestNTLinear:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_matches_f_linear(self, dtype):
        device = _get_device()
        from vllm_nt._ntops.oot_support import linear

        x = torch.randn((2, 3, 16), dtype=dtype, device=device)
        weight = torch.randn((8, 16), dtype=dtype, device=device)
        bias = torch.randn((8,), dtype=dtype, device=device)

        output = linear(x, weight, bias)
        reference = F.linear(x, weight, bias)

        torch.testing.assert_close(output, reference, atol=0.05, rtol=0.05)


class TestNTWPE:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_matches_embedding(self, dtype):
        device = _get_device()
        from vllm_nt._ntops.torch.wpe import wpe

        weight = torch.randn((1024, 768), dtype=dtype, device=device)
        position_ids = torch.tensor([[0, 1, 2], [511, 512, 513]], device=device)

        output = wpe(position_ids, weight)
        reference = F.embedding(position_ids, weight)

        torch.testing.assert_close(output, reference, atol=0.05, rtol=0.05)


class TestNTLayerNormDelegation:
    def test_layer_norm_prefers_ntops_torch(self, monkeypatch):
        from vllm_nt._ntops.torch.layer_norm import layer_norm

        fake_ntops = ModuleType("ntops")
        fake_torch = ModuleType("ntops.torch")
        sentinel = torch.randn((2, 4), dtype=torch.float32)

        def fake_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
            return sentinel

        fake_torch.layer_norm = fake_layer_norm
        fake_ntops.torch = fake_torch
        monkeypatch.setitem(sys.modules, "ntops", fake_ntops)
        monkeypatch.setitem(sys.modules, "ntops.torch", fake_torch)

        out = layer_norm(torch.randn((2, 4)), (4,))

        assert out is sentinel

    def test_torch_namespace_delegates_missing_ops(self, monkeypatch):
        import vllm_nt._ntops.torch as nt_torch

        fake_ntops = ModuleType("ntops")
        fake_torch = ModuleType("ntops.torch")

        def fake_relu(x):
            return ("relu", x)

        def fake_softmax(x, dim=-1):
            return ("softmax", x, dim)

        fake_torch.relu = fake_relu
        fake_torch.softmax = fake_softmax
        fake_ntops.torch = fake_torch
        monkeypatch.setitem(sys.modules, "ntops", fake_ntops)
        monkeypatch.setitem(sys.modules, "ntops.torch", fake_torch)

        assert nt_torch.relu("x") == ("relu", "x")
        assert nt_torch.softmax("x", dim=1) == ("softmax", "x", 1)


class TestNTGeluDelegation:
    def test_gelu_prefers_ntops_torch(self, monkeypatch):
        from vllm_nt._ntops.torch.gelu import gelu

        fake_ntops = ModuleType("ntops")
        fake_torch = ModuleType("ntops.torch")
        sentinel = torch.randn((2, 4), dtype=torch.float32)

        def fake_gelu(input, approximate="tanh"):
            return sentinel

        fake_torch.gelu = fake_gelu
        fake_ntops.torch = fake_torch
        monkeypatch.setitem(sys.modules, "ntops", fake_ntops)
        monkeypatch.setitem(sys.modules, "ntops.torch", fake_torch)

        out = gelu(torch.randn((2, 4)), approximate="tanh")

        assert out is sentinel
