import atexit
from dataclasses import dataclass
import logging
import os
import sys
from typing import Any, Callable, cast

import torch
from vllm.model_executor.layers import activation, layernorm

from vllm_nt._ntops.torch import rms_norm as nt_rms_norm
from vllm_nt._ntops.torch import silu as nt_silu

logger = logging.getLogger("vllm_nt")
_PARENT_PID_ENV = "VLLM_NT_PARENT_PID"
os.environ.setdefault(_PARENT_PID_ENV, str(os.getpid()))
RMSNorm = layernorm.RMSNorm
SiluAndMul = activation.SiluAndMul
OperatorSpec = tuple[type, Callable[..., object]]


@dataclass
class OperatorStats:
    hits: int = 0
    logged: bool = False
    registered_via: str | None = None


def _record_hit(name: str, x: torch.Tensor) -> None:
    stats = _OPERATOR_STATS[name]
    stats.hits += 1
    if not stats.logged:
        logger.info("vllm-nt: ninetoothed %s kernel invoked (shape=%s)", name, x.shape)
        stats.logged = True


def _nt_rms_norm_forward(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    _record_hit("RMSNorm", x)
    if residual is not None:
        x = x + residual
        residual = x.to(x.dtype)

    out = nt_rms_norm(
        x,
        normalized_shape=self.hidden_size,
        weight=self.weight if self.has_weight else None,
        eps=self.variance_epsilon,
    )
    return (out, residual) if residual is not None else out


def _nt_silu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    _record_hit("SiluAndMul", x)
    d = x.shape[-1] // 2
    return nt_silu(x[..., :d]) * x[..., d:]


def _nt_mul_and_silu_forward(self, x: torch.Tensor) -> torch.Tensor:
    _record_hit("MulAndSilu", x)
    d = x.shape[-1] // 2
    return x[..., :d] * nt_silu(x[..., d:])


def _nt_gemma_rms_norm_forward(
    self, x: torch.Tensor, residual: torch.Tensor | None = None
):
    _record_hit("GemmaRMSNorm", x)
    if residual is not None:
        x = x + residual
        residual = x
    out = nt_rms_norm(
        x,
        normalized_shape=self.weight.shape[0],
        weight=1.0 + self.weight,
        eps=self.variance_epsilon,
    )
    return (out, residual) if residual is not None else out


_OPERATOR_SPECS: dict[str, OperatorSpec] = {
    "RMSNorm": (RMSNorm, _nt_rms_norm_forward),
    "SiluAndMul": (SiluAndMul, _nt_silu_and_mul_forward),
}
for name, cls, forward in (
    ("MulAndSilu", getattr(activation, "MulAndSilu", None), _nt_mul_and_silu_forward),
    (
        "GemmaRMSNorm",
        getattr(layernorm, "GemmaRMSNorm", None),
        _nt_gemma_rms_norm_forward,
    ),
):
    if cls is not None:
        _OPERATOR_SPECS[name] = (cls, forward)
_OPERATOR_STATS = {name: OperatorStats() for name in _OPERATOR_SPECS}
_summary_printed = False
_registered = False


def _try_register_oot() -> bool:
    try:
        for name, spec in _OPERATOR_SPECS.items():
            cls, forward = spec
            decorator = cls.register_oot(name=name)
            decorator(type(f"NT{name}", (cls,), {"forward_oot": forward}))
            _OPERATOR_STATS[name].registered_via = "oot"
        logger.info(
            "vllm-nt: OOT registration succeeded for %s", ", ".join(_OPERATOR_SPECS)
        )
        return True
    except Exception as e:
        logger.warning("OOT registration failed (%s), will monkey-patch", e)
        return False


def _monkey_patch() -> None:
    for name, spec in _OPERATOR_SPECS.items():
        cls, forward = spec
        cls.forward_oot = forward
        cls.forward_native = forward
        _OPERATOR_STATS[name].registered_via = "monkey_patch"
    logger.info("vllm-nt: monkey-patched %s", ", ".join(_OPERATOR_SPECS))


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
        "registered_ops": list(_OPERATOR_SPECS),
        "hit_ops": hit_ops,
        "missed_ops": [name for name in _OPERATOR_SPECS if name not in hit_ops],
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
        mode = stats["registered_via"] or "unregistered"
        lines.append(
            f"{colors['blue']}- {name}: hits={stats['hits']} ({mode}){colors['reset']}"
        )
    missed = ", ".join(cast(list[str], summary["missed_ops"])) or "None"
    lines.append(f"{colors['blue']}Missed operators: {missed}{colors['reset']}")
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


ensure_registered()
