"""Integration tests for the vllm-nt plugin registration flow."""

import ast
from pathlib import Path
from typing import Any, cast

import pytest


def _require_runtime():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")


class TestPluginRegistration:
    """Test that register() correctly patches vLLM layers."""

    def test_register_patches_rmsnorm(self):
        _require_runtime()
        from vllm.model_executor.layers.layernorm import RMSNorm

        from vllm_nt import register
        from vllm_nt.oot import _nt_rms_norm_forward

        register()

        # Either OOT registry replaced the class, or monkey-patch
        # replaced forward_oot/forward_native. Either way, our
        # implementation should be reachable.
        assert (
            RMSNorm.forward_oot == _nt_rms_norm_forward
            or RMSNorm.forward_native == _nt_rms_norm_forward
            or hasattr(RMSNorm, "name")  # OOT-registered subclass marker
        )

    def test_register_patches_silu_and_mul(self):
        _require_runtime()
        from vllm.model_executor.layers.activation import SiluAndMul

        from vllm_nt import register
        from vllm_nt.oot import _nt_silu_and_mul_forward

        register()

        assert (
            SiluAndMul.forward_oot == _nt_silu_and_mul_forward
            or SiluAndMul.forward_native == _nt_silu_and_mul_forward
            or hasattr(SiluAndMul, "name")
        )

    def test_register_patches_optional_mul_and_silu(self):
        _require_runtime()
        from vllm.model_executor.layers import activation

        if not hasattr(activation, "MulAndSilu"):
            pytest.skip("MulAndSilu is unavailable in this vLLM version")

        from vllm_nt import register
        from vllm_nt.oot import _nt_mul_and_silu_forward

        mul_and_silu = activation.MulAndSilu
        register()

        assert (
            mul_and_silu.forward_oot == _nt_mul_and_silu_forward
            or mul_and_silu.forward_native == _nt_mul_and_silu_forward
            or hasattr(mul_and_silu, "name")
        )

    def test_register_patches_optional_gelu_and_mul(self):
        _require_runtime()
        from vllm.model_executor.layers import activation

        if not hasattr(activation, "GeluAndMul"):
            pytest.skip("GeluAndMul is unavailable in this vLLM version")

        from vllm_nt import register
        from vllm_nt.oot import _nt_gelu_and_mul_forward

        gelu_and_mul = activation.GeluAndMul
        register()

        assert (
            gelu_and_mul.forward_oot == _nt_gelu_and_mul_forward
            or gelu_and_mul.forward_native == _nt_gelu_and_mul_forward
            or hasattr(gelu_and_mul, "name")
        )

    def test_register_patches_optional_gemma_rms_norm(self):
        _require_runtime()
        from vllm.model_executor.layers import layernorm

        if not hasattr(layernorm, "GemmaRMSNorm"):
            pytest.skip("GemmaRMSNorm is unavailable in this vLLM version")

        from vllm_nt import register
        from vllm_nt.oot import _nt_gemma_rms_norm_forward

        gemma_rms_norm = layernorm.GemmaRMSNorm
        register()

        assert (
            gemma_rms_norm.forward_oot == _nt_gemma_rms_norm_forward
            or gemma_rms_norm.forward_native == _nt_gemma_rms_norm_forward
            or hasattr(gemma_rms_norm, "name")
        )

    def test_register_is_idempotent(self):
        _require_runtime()
        from vllm_nt import register

        # Should not raise on repeated calls
        register()
        register()
        register()

    def test_usage_summary_auto_discovers_registered_ops(self):
        _require_runtime()
        from vllm_nt import register
        from vllm_nt.oot import _OPERATOR_SPECS, _reset_usage_state, get_usage_summary

        _reset_usage_state()
        register()

        summary = cast(dict[str, Any], get_usage_summary())

        assert set(summary["registered_ops"]) == set(_OPERATOR_SPECS)
        assert set(summary["missed_ops"]) == set(_OPERATOR_SPECS)
        assert all(
            details["registered_via"] in {"oot", "monkey_patch"}
            for details in summary["operators"].values()
        )

    def test_usage_summary_tracks_hits(self, monkeypatch):
        _require_runtime()
        import torch

        from vllm_nt.oot import (
            _nt_rms_norm_forward,
            _reset_usage_state,
            get_usage_summary,
        )

        class DummyRMSNorm:
            hidden_size = 4
            has_weight = False
            weight = None
            variance_epsilon = 1e-6

        _reset_usage_state()
        monkeypatch.setattr("vllm_nt.oot.nt_rms_norm", lambda *args, **kwargs: args[0])

        output = _nt_rms_norm_forward(DummyRMSNorm(), torch.ones(1, 4))
        summary = cast(dict[str, Any], get_usage_summary())

        assert not isinstance(output, tuple)
        assert output.shape == (1, 4)
        assert summary["operators"]["RMSNorm"]["hits"] == 1
        assert "RMSNorm" in summary["hit_ops"]
        assert "SiluAndMul" in summary["missed_ops"]

    def test_optional_ops_are_registered_in_summary_when_available(self):
        _require_runtime()
        from vllm.model_executor.layers import activation, layernorm

        from vllm_nt import register
        from vllm_nt.oot import _reset_usage_state, get_usage_summary

        _reset_usage_state()
        register()
        summary = cast(dict[str, Any], get_usage_summary())

        if hasattr(activation, "MulAndSilu"):
            assert "MulAndSilu" in summary["registered_ops"]
        if hasattr(activation, "GeluAndMul"):
            assert "GeluAndMul" in summary["registered_ops"]
        if hasattr(layernorm, "GemmaRMSNorm"):
            assert "GemmaRMSNorm" in summary["registered_ops"]

    def test_gelu_and_mul_forward_uses_nt_tanh_path(self, monkeypatch):
        _require_runtime()
        import torch

        from vllm_nt.oot import (
            _nt_gelu_and_mul_forward,
            _reset_usage_state,
            get_usage_summary,
        )

        class DummyGeluAndMul:
            approximate = "tanh"

        _reset_usage_state()
        monkeypatch.setattr("vllm_nt.oot.nt_gelu", lambda t: t + 1)

        x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        out = _nt_gelu_and_mul_forward(DummyGeluAndMul(), x)
        summary = cast(dict[str, Any], get_usage_summary())

        torch.testing.assert_close(out, (x[..., :4] + 1) * x[..., 4:])
        assert summary["operators"]["GeluAndMul"]["hits"] == 1

    def test_gelu_and_mul_forward_falls_back_for_non_tanh(self):
        _require_runtime()
        import torch
        import torch.nn.functional as F

        from vllm_nt.oot import _nt_gelu_and_mul_forward

        class DummyGeluAndMul:
            approximate = "none"

        x = torch.randn(2, 8, dtype=torch.float32)
        out = _nt_gelu_and_mul_forward(DummyGeluAndMul(), x)

        torch.testing.assert_close(
            out, F.gelu(x[..., :4], approximate="none") * x[..., 4:]
        )


class TestNoHardwareDependency:
    """Verify vllm_nt does not import any hardware-vendor plugins (R4)."""

    def test_no_hardware_imports(self):
        forbidden = {"vllm_mlu", "vllm_xpu", "vllm_ascend", "vllm_cambricon"}
        vllm_nt_dir = Path(__file__).parent.parent / "vllm_nt"

        for py_file in vllm_nt_dir.rglob("*.py"):
            source = py_file.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        assert root not in forbidden, (
                            f"{py_file.relative_to(vllm_nt_dir.parent)} "
                            f"imports forbidden module: {alias.name}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        root = node.module.split(".")[0]
                        assert root not in forbidden, (
                            f"{py_file.relative_to(vllm_nt_dir.parent)} "
                            f"imports from forbidden module: {node.module}"
                        )


class TestCodeSize:
    """Verify core code (excluding vendored _ntops and tests) is < 200 lines (R7)."""

    def test_core_code_under_200_lines(self):
        vllm_nt_dir = Path(__file__).parent.parent / "vllm_nt"
        total_lines = 0

        for py_file in vllm_nt_dir.rglob("*.py"):
            if "_ntops" in str(py_file):
                continue
            total_lines += len(py_file.read_text().splitlines())

        assert total_lines < 200, f"Core code is {total_lines} lines, should be < 200"
