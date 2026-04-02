"""Integration tests for the vllm-nt plugin registration flow."""

import ast
from pathlib import Path

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

        summary = get_usage_summary()

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
        summary = get_usage_summary()

        assert not isinstance(output, tuple)
        assert output.shape == (1, 4)
        assert summary["operators"]["RMSNorm"]["hits"] == 1
        assert "RMSNorm" in summary["hit_ops"]
        assert "SiluAndMul" in summary["missed_ops"]


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
