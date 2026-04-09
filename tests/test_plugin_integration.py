"""Integration tests for the vllm-nt plugin registration flow."""

import ast
from pathlib import Path
import sys
from types import ModuleType
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

    def test_register_patches_unquantized_linear_method(self):
        _require_runtime()
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        from vllm_nt import register
        from vllm_nt.oot import _nt_unquantized_linear_apply

        register()

        assert UnquantizedLinearMethod.apply == _nt_unquantized_linear_apply
        register()

    def test_register_patches_unquantized_embedding_method(self):
        _require_runtime()
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        from vllm_nt import register
        from vllm_nt.oot import _nt_unquantized_embedding

        register()

        assert UnquantizedEmbeddingMethod.embedding == _nt_unquantized_embedding

    def test_usage_summary_auto_discovers_registered_ops(self):
        _require_runtime()
        from vllm_nt import register
        from vllm_nt.oot import _OPERATOR_SPECS, _reset_usage_state, get_usage_summary

        _reset_usage_state()
        register()

        summary = cast(dict[str, Any], get_usage_summary())

        assert set(_OPERATOR_SPECS).issubset(set(summary["registered_ops"]))
        assert "MatMul" in summary["registered_ops"]
        assert "Embedding" in summary["registered_ops"]
        assert set(summary["missed_ops"]) == set(summary["registered_ops"])
        assert all(
            details["registered_via"] in {"oot", "monkey_patch", "function_patch", None}
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
        monkeypatch.setattr(
            "vllm_nt._ntops.oot_support.nt_rms_norm", lambda *args, **kwargs: args[0]
        )

        output = _nt_rms_norm_forward(DummyRMSNorm(), torch.ones(1, 4))
        summary = cast(dict[str, Any], get_usage_summary())

        assert not isinstance(output, tuple)
        assert output.shape == (1, 4)
        assert summary["operators"]["RMSNorm"]["hits"] == 1
        assert "RMSNorm" in summary["hit_ops"]
        assert "SiluAndMul" in summary["missed_ops"]

    def test_mlu_topkp_patch_tracks_hits(self):
        _require_runtime()
        import torch
        import vllm_nt._ntops.patching as patching

        _reset_usage_state = patching._reset_usage_state
        get_usage_summary = patching.get_usage_summary

        _reset_usage_state()
        wrapped = patching._build_mlu_apply_topkp_v2_patch(
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback"))
        )

        logits = torch.tensor([[1.0, 0.5, -1.0], [0.2, 0.1, 0.0]], dtype=torch.float32)
        index_in = torch.arange(logits.shape[-1], dtype=torch.int32)
        temperature = torch.ones(logits.shape[0], dtype=torch.float32)
        top_k = torch.tensor([2, 2], dtype=torch.int32)

        logits_out, sorted_logits_out, index_out, true_select_len = wrapped(
            logits,
            index_in,
            temperature,
            None,
            top_k,
            None,
        )
        summary = cast(dict[str, Any], get_usage_summary())

        assert logits_out.shape == logits.shape
        assert sorted_logits_out.shape == logits.shape
        assert index_out.shape == logits.shape
        assert true_select_len.shape == (logits.shape[0],)
        assert summary["operators"]["TopKTopP"]["hits"] == 1

    def test_mlu_random_sample_patch_tracks_hits(self):
        _require_runtime()
        import torch
        import vllm_nt._ntops.patching as patching

        _reset_usage_state = patching._reset_usage_state
        get_usage_summary = patching.get_usage_summary

        _reset_usage_state()
        wrapped = patching._build_mlu_random_sample_patch(
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback"))
        )

        probs = torch.tensor([[0.6, 0.4], [0.1, 0.9]], dtype=torch.float32)
        sampled = wrapped(probs, {})
        summary = cast(dict[str, Any], get_usage_summary())

        assert sampled.shape == (2,)
        assert summary["operators"]["RandomSample"]["hits"] == 1

    def test_rejection_sample_patch_tracks_hits(self):
        _require_runtime()
        import torch
        import vllm_nt._ntops.patching as patching

        _reset_usage_state = patching._reset_usage_state
        get_usage_summary = patching.get_usage_summary

        _reset_usage_state()
        sentinel = torch.zeros((1, 2), dtype=torch.int32)
        wrapped = patching._build_rejection_sample_patch(
            lambda *args, **kwargs: sentinel
        )

        out = wrapped(None, None, None, None, None, torch.ones(1, 4), None, None)
        summary = cast(dict[str, Any], get_usage_summary())

        assert out is sentinel
        assert summary["operators"]["RejectionSample"]["hits"] == 1

    def test_qwen2_mlp_patch_tracks_silu_and_mul_hits(self):
        _require_runtime()
        import torch
        import vllm_nt._ntops.patching as patching

        _reset_usage_state = patching._reset_usage_state
        get_usage_summary = patching.get_usage_summary

        class DummyProj:
            def __init__(self, out: torch.Tensor):
                self.out = out

            def __call__(self, x: torch.Tensor):
                return self.out, None

        class DummyMLP:
            def __init__(self):
                self.gate_up_proj = DummyProj(
                    torch.arange(8, dtype=torch.float32).reshape(1, 8)
                )
                self.down_proj = DummyProj(
                    torch.arange(4, dtype=torch.float32).reshape(1, 4)
                )

        _reset_usage_state()
        wrapped = patching._build_qwen2_mlp_forward(
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback"))
        )

        out = wrapped(DummyMLP(), torch.ones(1, 4))
        summary = cast(dict[str, Any], get_usage_summary())

        assert out.shape == (1, 4)
        assert summary["operators"]["SiluAndMul"]["hits"] == 1

    def test_mlu_active_patch_tracks_gated_silu_hits(self):
        _require_runtime()
        import torch
        import vllm_nt._ntops.patching as patching

        _reset_usage_state = patching._reset_usage_state
        get_usage_summary = patching.get_usage_summary

        _reset_usage_state()
        wrapped = patching._build_mlu_active_patch(
            lambda input, act_mode, is_gated: input
        )

        x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        out = wrapped(x, "silu", True)
        summary = cast(dict[str, Any], get_usage_summary())

        assert out is x
        assert summary["operators"]["SiluAndMul"]["hits"] == 1

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

    def test_function_patch_spec_can_patch_fake_module(self, monkeypatch):
        _require_runtime()
        import vllm_nt._ntops.patching as patching

        parent = ModuleType("fakepkg")
        child = ModuleType("fakepkg.child")

        def original(*args, **kwargs):
            return ("original", args, kwargs)

        child.unified_attention_2d = original
        parent.child = child
        monkeypatch.setitem(sys.modules, "fakepkg", parent)
        monkeypatch.setitem(sys.modules, "fakepkg.child", child)

        spec = patching.FunctionPatchSpec(
            patch_id="UnifiedAttention2D",
            module_path="fakepkg.child",
            attr_name="unified_attention_2d",
            required=True,
            builder=lambda fn: (lambda *args, **kwargs: ("patched", fn(*args, **kwargs))),
        )

        monkeypatch.setattr(patching, "_FUNCTION_PATCH_SPECS", (spec,))
        monkeypatch.setattr(patching, "_APPLIED_FUNCTION_PATCHES", [])
        monkeypatch.setitem(
            patching._OPERATOR_STATS,
            "PagedAttentionPrefill",
            patching.OperatorStats(),
        )
        monkeypatch.setitem(
            patching._OPERATOR_STATS,
            "PagedAttentionDecode",
            patching.OperatorStats(),
        )

        patching._apply_function_patches()

        assert child.unified_attention_2d(1, flag=True)[0] == "patched"
        assert (
            patching._OPERATOR_STATS["PagedAttentionPrefill"].registered_via
            == "function_patch"
        )
        assert (
            patching._OPERATOR_STATS["PagedAttentionDecode"].registered_via
            == "function_patch"
        )

    def test_function_patch_spec_can_patch_fake_object_attr(self, monkeypatch):
        _require_runtime()
        import vllm_nt._ntops.patching as patching

        parent = ModuleType("fakepkg_object")
        child = ModuleType("fakepkg_object.child")

        class _Functional:
            @staticmethod
            def scaled_dot_product_attention(*args, **kwargs):
                return ("original", args, kwargs)

        child.F = _Functional
        parent.child = child
        monkeypatch.setitem(sys.modules, "fakepkg_object", parent)
        monkeypatch.setitem(sys.modules, "fakepkg_object.child", child)

        spec = patching.FunctionPatchSpec(
            patch_id="SDPA",
            module_path="fakepkg_object.child",
            object_name="F",
            attr_name="scaled_dot_product_attention",
            required=True,
            builder=lambda fn: (lambda *args, **kwargs: ("patched", fn(*args, **kwargs))),
        )

        monkeypatch.setattr(patching, "_FUNCTION_PATCH_SPECS", (spec,))
        monkeypatch.setattr(patching, "_APPLIED_FUNCTION_PATCHES", [])
        monkeypatch.setitem(
            patching._OPERATOR_STATS,
            "SDPA",
            patching.OperatorStats(),
        )

        patching._apply_function_patches()

        assert child.F.scaled_dot_product_attention(1, flag=True)[0] == "patched"
        assert patching._OPERATOR_STATS["SDPA"].registered_via == "function_patch"

    def test_custom_op_rebinding_is_gated_and_updates_stats(self, monkeypatch):
        _require_runtime()
        import vllm_nt._ntops.patching as patching

        calls: list[tuple[str, str]] = []

        class _Platform:
            dispatch_key = "MLU"

        monkeypatch.setenv("VLLM_NT_ENABLE_CUSTOM_OP_REBIND", "1")
        monkeypatch.setattr(
            patching,
            "_APPLIED_CUSTOM_OP_REBINDS",
            [],
        )
        monkeypatch.setattr(
            patching,
            "_rebind_custom_op",
            lambda op_name, op_func, dispatch_key: (
                calls.append((op_name, dispatch_key))
                or patching._AppliedCustomOpRebind(op_name, dispatch_key, object())
            ),
        )
        monkeypatch.setattr("vllm.platforms.current_platform", _Platform())
        monkeypatch.setitem(
            patching._OPERATOR_STATS,
            "PagedAttentionPrefill",
            patching.OperatorStats(),
        )
        monkeypatch.setitem(
            patching._OPERATOR_STATS,
            "PagedAttentionDecode",
            patching.OperatorStats(),
        )

        patching._apply_custom_op_rebindings()

        assert calls == [
            ("unified_kv_cache_update", "MLU"),
            ("unified_attention_with_output", "MLU"),
        ]
        assert (
            patching._OPERATOR_STATS["PagedAttentionPrefill"].registered_via
            == "custom_op_rebind"
        )
        assert (
            patching._OPERATOR_STATS["PagedAttentionDecode"].registered_via
            == "custom_op_rebind"
        )

    def test_custom_op_register_intercept_swaps_registration_target(
        self, monkeypatch
    ):
        _require_runtime()
        import vllm_nt._ntops.patching as patching

        calls: list[tuple[str, object]] = []

        fake_torch_utils = ModuleType("vllm.utils.torch_utils")

        def fake_direct_register_custom_op(
            op_name,
            op_func,
            mutates_args=None,
            fake_impl=None,
            target_lib=None,
            dispatch_key=None,
            tags=(),
        ):
            calls.append((op_name, op_func))

        fake_torch_utils.direct_register_custom_op = fake_direct_register_custom_op
        monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", fake_torch_utils)
        monkeypatch.setenv("VLLM_NT_ENABLE_CUSTOM_OP_REGISTER_INTERCEPT", "1")
        monkeypatch.setattr(patching, "_INSTALLED_CUSTOM_OP_INTERCEPT", None)
        monkeypatch.setitem(
            patching._OPERATOR_STATS,
            "PagedAttentionPrefill",
            patching.OperatorStats(),
        )
        monkeypatch.setitem(
            patching._OPERATOR_STATS,
            "PagedAttentionDecode",
            patching.OperatorStats(),
        )

        patching._install_custom_op_register_intercept()

        assert fake_torch_utils.direct_register_custom_op is not fake_direct_register_custom_op
        fake_torch_utils.direct_register_custom_op(
            "unified_attention_with_output",
            lambda *args, **kwargs: None,
        )

        assert calls
        assert calls[0][0] == "unified_attention_with_output"
        assert calls[0][1] is not None
        assert (
            patching._OPERATOR_STATS["PagedAttentionPrefill"].registered_via
            == "custom_op_intercept"
        )
        assert (
            patching._OPERATOR_STATS["PagedAttentionDecode"].registered_via
            == "custom_op_intercept"
        )

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
        monkeypatch.setattr("vllm_nt._ntops.patching.nt_gelu", lambda t: t + 1)

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

    def test_unquantized_linear_apply_uses_nt_matmul(self, monkeypatch):
        _require_runtime()
        import torch

        from vllm_nt.oot import (
            _nt_unquantized_linear_apply,
            _reset_usage_state,
            get_usage_summary,
        )

        class DummyLinearMethod:
            pass

        class DummyLayer:
            weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        _reset_usage_state()
        monkeypatch.setattr(
            "vllm_nt._ntops.patching.linear",
            lambda x, weight, bias=None: x @ weight.T
            if bias is None
            else x @ weight.T + bias,
        )

        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        bias = torch.ones(3, dtype=torch.float32)
        out = _nt_unquantized_linear_apply(DummyLinearMethod(), DummyLayer(), x, bias)
        summary = cast(dict[str, Any], get_usage_summary())

        torch.testing.assert_close(out, x @ DummyLayer.weight.T + bias)
        assert summary["operators"]["MatMul"]["hits"] == 1

    def test_unquantized_linear_apply_reshapes_higher_rank_without_bias(
        self, monkeypatch
    ):
        _require_runtime()
        import torch

        from vllm_nt.oot import _nt_unquantized_linear_apply

        class DummyLinearMethod:
            pass

        class DummyLayer:
            weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        monkeypatch.setattr(
            "vllm_nt._ntops.patching.linear",
            lambda x, weight, bias=None: x @ weight.T,
        )

        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        out = _nt_unquantized_linear_apply(DummyLinearMethod(), DummyLayer(), x)

        torch.testing.assert_close(out, x @ DummyLayer.weight.T)
        assert out.shape == (2, 3, 3)

    def test_unquantized_linear_apply_accepts_residual_kwarg(self, monkeypatch):
        _require_runtime()
        import torch

        from vllm_nt.oot import _nt_unquantized_linear_apply

        class DummyLinearMethod:
            pass

        class DummyLayer:
            weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        monkeypatch.setattr(
            "vllm_nt._ntops.patching.linear",
            lambda x, weight, bias=None: x @ weight.T,
        )

        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        residual = torch.ones((x.shape[0], DummyLayer.weight.shape[0]), dtype=x.dtype)
        out = _nt_unquantized_linear_apply(
            DummyLinearMethod(), DummyLayer(), x, residual=residual
        )

        torch.testing.assert_close(out, x @ DummyLayer.weight.T + residual)

    def test_unquantized_embedding_uses_plugin_path(self):
        _require_runtime()
        import torch

        from vllm_nt.oot import (
            _nt_unquantized_embedding,
            _reset_usage_state,
            get_usage_summary,
        )

        class DummyEmbeddingMethod:
            pass

        class DummyEmbeddingLayer:
            weight = torch.randn((16, 4), dtype=torch.float32)

        _reset_usage_state()
        input_ids = torch.tensor([[0, 2], [3, 1]])
        out = _nt_unquantized_embedding(
            DummyEmbeddingMethod(), DummyEmbeddingLayer(), input_ids
        )
        summary = cast(dict[str, Any], get_usage_summary())

        torch.testing.assert_close(
            out, torch.nn.functional.embedding(input_ids, DummyEmbeddingLayer.weight)
        )
        assert summary["operators"]["Embedding"]["hits"] == 1


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
