import atexit
import logging
import os
import sys
from typing import Callable, Protocol, TypedDict

import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_nt._ntops.torch import rms_norm as nt_rms_norm
from vllm_nt._ntops.torch import silu as nt_silu

logger = logging.getLogger("vllm_nt")
_PARENT_PID_ENV = "VLLM_NT_PARENT_PID"
os.environ.setdefault(_PARENT_PID_ENV, str(os.getpid()))


class OOTOperator(Protocol):
    forward_oot: Callable[..., object]
    forward_native: Callable[..., object]

    @classmethod
    def register_oot(cls, *, name: str) -> Callable[[type[object]], object]: ...


class OperatorSpec(TypedDict):
    cls: type[OOTOperator]
    forward: Callable[..., object]


class OperatorStats(TypedDict):
    hits: int
    logged: bool
    registered_via: str | None


class UsageOperator(TypedDict):
    hits: int
    registered_via: str | None


class UsageSummary(TypedDict):
    registered_ops: list[str]
    hit_ops: list[str]
    missed_ops: list[str]
    operators: dict[str, UsageOperator]


def _record_hit(name: str, x: torch.Tensor) -> None:
    stats = _OPERATOR_STATS[name]
    stats["hits"] += 1
    if not stats["logged"]:
        logger.info("vllm-nt: ninetoothed %s kernel invoked (shape=%s)", name, x.shape)
        stats["logged"] = True


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

    if residual is not None:
        return out, residual
    return out


def _nt_silu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    _record_hit("SiluAndMul", x)
    d = x.shape[-1] // 2
    return nt_silu(x[..., :d]) * x[..., d:]


_OPERATOR_SPECS: dict[str, OperatorSpec] = {
    "RMSNorm": {"cls": RMSNorm, "forward": _nt_rms_norm_forward},
    "SiluAndMul": {"cls": SiluAndMul, "forward": _nt_silu_and_mul_forward},
}
_OPERATOR_STATS: dict[str, OperatorStats] = {
    name: {"hits": 0, "logged": False, "registered_via": None}
    for name in _OPERATOR_SPECS
}
_summary_printed = False
_registered = False


def _try_register_oot() -> bool:
    try:
        for name, spec in _OPERATOR_SPECS.items():
            decorator = spec["cls"].register_oot(name=name)
            decorator(
                type(f"NT{name}", (spec["cls"],), {"forward_oot": spec["forward"]})
            )
            _OPERATOR_STATS[name]["registered_via"] = "oot"
        logger.info(
            "vllm-nt: OOT registration succeeded for %s",
            ", ".join(_OPERATOR_SPECS),
        )
        return True
    except Exception as e:
        logger.warning("OOT registration failed (%s), will monkey-patch", e)
        return False


def _monkey_patch() -> None:
    for name, spec in _OPERATOR_SPECS.items():
        spec["cls"].forward_oot = spec["forward"]
        spec["cls"].forward_native = spec["forward"]
        _OPERATOR_STATS[name]["registered_via"] = "monkey_patch"
    logger.info("vllm-nt: monkey-patched %s", ", ".join(_OPERATOR_SPECS))


def get_usage_summary() -> UsageSummary:
    operators: dict[str, UsageOperator] = {
        name: {
            "hits": stats["hits"],
            "registered_via": stats["registered_via"],
        }
        for name, stats in _OPERATOR_STATS.items()
    }
    hit_ops = [name for name, stats in operators.items() if stats["hits"] > 0]
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
    for name, stats in summary["operators"].items():
        mode = stats["registered_via"] or "unregistered"
        lines.append(
            f"{colors['blue']}- {name}: hits={stats['hits']} ({mode}){colors['reset']}"
        )
    missed = ", ".join(summary["missed_ops"]) or "None"
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
        stats.update(hits=0, logged=False)
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
