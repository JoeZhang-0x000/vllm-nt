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
    return linear(x, layer.weight, bias)


def _nt_unquantized_embedding(
    self, layer: torch.nn.Module, input_: torch.Tensor
) -> torch.Tensor:
    _record_hit("Embedding", input_)
    return embedding(layer, input_)


def _patch_leaf_methods() -> None:
    UnquantizedLinearMethod.apply = _nt_unquantized_linear_apply
    UnquantizedEmbeddingMethod.embedding = _nt_unquantized_embedding
    _OPERATOR_STATS["MatMul"].registered_via = _OPERATOR_STATS[
        "Embedding"
    ].registered_via = "monkey_patch"


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
    global _summary_printed
    for stats in _OPERATOR_STATS.values():
        stats.hits = 0
        stats.logged = False
    _summary_printed = False


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
