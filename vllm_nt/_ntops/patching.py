import atexit
import logging
import os
import sys
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
from vllm.model_executor.layers import activation, layernorm
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_nt._ntops.oot_support import (
    OperatorStats,
    act_and_mul,
    embedding,
    linear,
    norm,
    nt_rms_norm,
)
from vllm_nt._ntops.torch import gelu as nt_gelu
from vllm_nt._ntops.torch import silu as nt_silu

logger = logging.getLogger("vllm_nt")
_PARENT_PID_ENV = "VLLM_NT_PARENT_PID"
os.environ.setdefault(_PARENT_PID_ENV, str(os.getpid()))
OperatorSpec = tuple[type, Callable[..., object]]


def _record_hit(name: str, x: torch.Tensor) -> None:
    stats = _OPERATOR_STATS[name]
    stats.hits += 1
    if not stats.logged:
        logger.info("vllm-nt: ninetoothed %s kernel invoked (shape=%s)", name, x.shape)
        stats.logged = True


def _nt_rms_norm_forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
    _record_hit("RMSNorm", x)
    return norm(self, x, self.weight if self.has_weight else None, residual)


def _nt_gemma_rms_norm_forward(
    self, x: torch.Tensor, residual: torch.Tensor | None = None
):
    _record_hit("GemmaRMSNorm", x)
    return norm(self, x, 1.0 + self.weight, residual, gemma=True)


def _act(
    name: str,
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    reverse: bool = False,
) -> torch.Tensor:
    _record_hit(name, x)
    return act_and_mul(x, fn, reverse)


def _nt_silu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    return _act("SiluAndMul", x, nt_silu)


def _nt_mul_and_silu_forward(self, x: torch.Tensor) -> torch.Tensor:
    return _act("MulAndSilu", x, nt_silu, True)


def _nt_gelu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    act = (
        nt_gelu
        if self.approximate == "tanh"
        else lambda t: F.gelu(t, approximate=self.approximate)
    )
    return _act("GeluAndMul", x, act)


_OPERATOR_SPECS: dict[str, OperatorSpec] = {
    "RMSNorm": (layernorm.RMSNorm, _nt_rms_norm_forward),
    "SiluAndMul": (activation.SiluAndMul, _nt_silu_and_mul_forward),
}
for name, cls, forward in (
    ("MulAndSilu", getattr(activation, "MulAndSilu", None), _nt_mul_and_silu_forward),
    ("GeluAndMul", getattr(activation, "GeluAndMul", None), _nt_gelu_and_mul_forward),
    (
        "GemmaRMSNorm",
        getattr(layernorm, "GemmaRMSNorm", None),
        _nt_gemma_rms_norm_forward,
    ),
):
    if cls is not None:
        _OPERATOR_SPECS[name] = (cls, forward)
_OPERATOR_STATS = {name: OperatorStats() for name in _OPERATOR_SPECS} | {
    "MatMul": OperatorStats(),
    "Embedding": OperatorStats(),
}
_summary_printed = False
_registered = False
_linear_debug_compare_calls = 0


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("vllm-nt: invalid integer for %s=%r; using %d", name, value, default)
        return default


def _read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("vllm-nt: invalid float for %s=%r; using %.6f", name, value, default)
        return default


def _maybe_compare_unquantized_linear_output(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    nt_output: torch.Tensor,
) -> torch.Tensor:
    global _linear_debug_compare_calls
    if not _read_bool_env("VLLM_NT_DEBUG_LINEAR_COMPARE"):
        return nt_output

    ref_output = F.linear(x, weight, bias)
    nt_output_fp32 = nt_output.to(torch.float32)
    ref_output_fp32 = ref_output.to(torch.float32)
    abs_diff = (nt_output_fp32 - ref_output_fp32).abs()
    max_abs_diff = abs_diff.max().item() if abs_diff.numel() else 0.0
    mean_abs_diff = abs_diff.mean().item() if abs_diff.numel() else 0.0
    ref_abs = ref_output_fp32.abs().clamp_min(1e-8)
    max_rel_diff = (abs_diff / ref_abs).max().item() if abs_diff.numel() else 0.0

    _linear_debug_compare_calls += 1
    compare_call = _linear_debug_compare_calls
    max_log_calls = _read_int_env("VLLM_NT_DEBUG_LINEAR_COMPARE_MAX_CALLS", 10)
    flattened_m = x.numel() // x.shape[-1]
    if compare_call <= max_log_calls:
        logger.warning(
            "vllm-nt: linear compare[%d] x=%s weight=%s bias=%s dtype=%s device=%s flattened_m=%d max_abs=%.6e mean_abs=%.6e max_rel=%.6e",
            compare_call,
            tuple(x.shape),
            tuple(weight.shape),
            None if bias is None else tuple(bias.shape),
            x.dtype,
            x.device,
            flattened_m,
            max_abs_diff,
            mean_abs_diff,
            max_rel_diff,
        )

    fail_open_atol = _read_float_env("VLLM_NT_DEBUG_LINEAR_FAIL_OPEN_ATOL", 0.0)
    if fail_open_atol > 0.0 and max_abs_diff > fail_open_atol:
        logger.warning(
            "vllm-nt: linear compare[%d] exceeded fail-open threshold %.6e; falling back to torch.nn.functional.linear",
            compare_call,
            fail_open_atol,
        )
        return ref_output

    return nt_output


def _try_register_oot() -> bool:
    try:
        for name, (cls, forward) in _OPERATOR_SPECS.items():
            cls.register_oot(name=name)(
                type(f"NT{name}", (cls,), {"forward_oot": forward})
            )
            _OPERATOR_STATS[name].registered_via = "oot"
        logger.info(
            "vllm-nt: OOT registration succeeded for %s", ", ".join(_OPERATOR_SPECS)
        )
        return True
    except Exception as e:
        logger.warning("OOT registration failed (%s), will monkey-patch", e)
        return False


def _monkey_patch() -> None:
    for name, (cls, forward) in _OPERATOR_SPECS.items():
        cls.forward_oot = forward
        cls.forward_native = forward
        _OPERATOR_STATS[name].registered_via = "monkey_patch"
    logger.info("vllm-nt: monkey-patched %s", ", ".join(_OPERATOR_SPECS))


def _nt_unquantized_linear_apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    _record_hit("MatMul", x)
    nt_output = linear(x, layer.weight, bias)
    return _maybe_compare_unquantized_linear_output(x, layer.weight, bias, nt_output)


def _nt_unquantized_embedding(
    self, layer: torch.nn.Module, input_: torch.Tensor
) -> torch.Tensor:
    _record_hit("Embedding", input_)
    return embedding(layer, input_)


def _patch_leaf_methods() -> None:
    if _read_bool_env("VLLM_NT_ENABLE_UNQUANTIZED_LINEAR_PATCH", True):
        UnquantizedLinearMethod.apply = _nt_unquantized_linear_apply
        _OPERATOR_STATS["MatMul"].registered_via = "monkey_patch"
    else:
        logger.info(
            "vllm-nt: skipping UnquantizedLinearMethod.apply patch because "
            "VLLM_NT_ENABLE_UNQUANTIZED_LINEAR_PATCH is disabled"
        )
    UnquantizedEmbeddingMethod.embedding = _nt_unquantized_embedding
    _OPERATOR_STATS["Embedding"].registered_via = "monkey_patch"


def get_usage_summary() -> dict[str, object]:
    operators = {
        name: {"hits": stats.hits, "registered_via": stats.registered_via}
        for name, stats in _OPERATOR_STATS.items()
    }
    hit_ops = [
        name
        for name, stats in operators.items()
        if cast(dict[str, Any], stats)["hits"] > 0
    ]
    return {
        "registered_ops": list(_OPERATOR_STATS),
        "hit_ops": hit_ops,
        "missed_ops": [name for name in _OPERATOR_STATS if name not in hit_ops],
        "operators": operators,
    }


def format_usage_summary(use_color: bool = True) -> str:
    summary = get_usage_summary()
    colors = {
        "blue": "\033[94m" if use_color else "",
        "reset": "\033[0m" if use_color else "",
    }
    lines = [f"{colors['blue']}Operator usage summary{colors['reset']}"]
    for name, stats in cast(dict[str, dict[str, Any]], summary["operators"]).items():
        lines.append(
            f"{colors['blue']}- {name}: hits={stats['hits']} ({stats['registered_via'] or 'unregistered'}){colors['reset']}"
        )
    lines.append(
        f"{colors['blue']}Missed operators: {', '.join(cast(list[str], summary['missed_ops'])) or 'None'}{colors['reset']}"
    )
    return "\n".join(lines)


def maybe_print_usage_summary(*, include_empty: bool = False) -> bool:
    global _summary_printed
    summary = get_usage_summary()
    if _summary_printed or (not include_empty and not summary["hit_ops"]):
        return False
    print(format_usage_summary(), file=sys.stderr)
    _summary_printed = True
    return True


def _reset_usage_state() -> None:
    global _summary_printed, _linear_debug_compare_calls
    for stats in _OPERATOR_STATS.values():
        stats.hits = 0
        stats.logged = False
    _summary_printed = False
    _linear_debug_compare_calls = 0


def _print_worker_summary_on_exit() -> None:
    if os.getpid() != int(os.environ.get(_PARENT_PID_ENV, os.getpid())):
        maybe_print_usage_summary(include_empty=True)


atexit.register(_print_worker_summary_on_exit)


def ensure_registered() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    if not _try_register_oot():
        _monkey_patch()
    _patch_leaf_methods()
