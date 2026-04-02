"""Integration tests for the vllm-nt plugin registration flow."""

import ast
import os
from pathlib import Path


class TestPluginRegistration:
    """Test that register() correctly populates vLLM's OOT registry."""

    def test_register_populates_oot_registry(self):
        from vllm.model_executor.custom_op import op_registry_oot
        from vllm_nt import register
        from vllm_nt.oot import NTRMSNorm, NTSiluAndMul

        register()

        assert "RMSNorm" in op_registry_oot
        assert op_registry_oot["RMSNorm"] is NTRMSNorm

        assert "SiluAndMul" in op_registry_oot
        assert op_registry_oot["SiluAndMul"] is NTSiluAndMul

    def test_register_is_idempotent(self):
        from vllm.model_executor.custom_op import op_registry_oot
        from vllm_nt import register

        # register() imports oot.py; decorators only run once at import time
        # Calling register() again should not raise
        register()
        register()

        assert "RMSNorm" in op_registry_oot
        assert "SiluAndMul" in op_registry_oot


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
            # Skip vendored ntops code
            if "_ntops" in str(py_file):
                continue
            total_lines += len(py_file.read_text().splitlines())

        assert total_lines < 200, (
            f"Core code is {total_lines} lines, should be < 200"
        )
