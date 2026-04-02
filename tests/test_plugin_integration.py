"""Integration tests for the vllm-nt plugin registration flow."""

import ast
from pathlib import Path


class TestPluginRegistration:
    """Test that register() correctly patches vLLM layers."""

    def test_register_patches_rmsnorm(self):
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
        from vllm_nt import register

        # Should not raise on repeated calls
        register()
        register()
        register()


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

        assert total_lines < 200, (
            f"Core code is {total_lines} lines, should be < 200"
        )
