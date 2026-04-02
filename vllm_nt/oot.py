"""OOT (out-of-tree) layer overrides for vLLM.

Strategy:
  1. Try @register_oot() — the clean, official vLLM extension path.
  2. If that fails (API missing or version mismatch), fall back to
     monkey-patching forward_oot / forward_native directly.
"""

import logging

import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_nt._ntops.torch import rms_norm as nt_rms_norm
from vllm_nt._ntops.torch import silu as nt_silu

logger = logging.getLogger("vllm_nt")


# ── NineToothed forward implementations ──────────────────────────


_nt_rms_norm_called = False


def _nt_rms_norm_forward(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    global _nt_rms_norm_called
    if not _nt_rms_norm_called:
        logger.info("vllm-nt: ninetoothed RMSNorm kernel invoked (shape=%s)", x.shape)
        _nt_rms_norm_called = True
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


_nt_silu_called = False


def _nt_silu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    global _nt_silu_called
    if not _nt_silu_called:
        logger.info("vllm-nt: ninetoothed SiluAndMul kernel invoked (shape=%s)", x.shape)
        _nt_silu_called = True
    d = x.shape[-1] // 2
    return nt_silu(x[..., :d]) * x[..., d:]


# ── Registration: OOT first, monkey-patch fallback ───────────────

_registered = False


def _try_register_oot() -> bool:
    """Attempt clean OOT registration. Returns True on success."""
    try:

        @RMSNorm.register_oot(name="RMSNorm")
        class NTRMSNorm(RMSNorm):
            forward_oot = _nt_rms_norm_forward

        @SiluAndMul.register_oot(name="SiluAndMul")
        class NTSiluAndMul(SiluAndMul):
            forward_oot = _nt_silu_and_mul_forward

        logger.info("vllm-nt: OOT registration succeeded for RMSNorm and SiluAndMul")
        return True
    except Exception as e:
        logger.warning("OOT registration failed (%s), will monkey-patch", e)
        return False


def _monkey_patch() -> None:
    """Fallback: directly replace forward methods on vLLM layer classes."""
    RMSNorm.forward_oot = _nt_rms_norm_forward
    RMSNorm.forward_native = _nt_rms_norm_forward

    SiluAndMul.forward_oot = _nt_silu_and_mul_forward
    SiluAndMul.forward_native = _nt_silu_and_mul_forward

    logger.info("vllm-nt: monkey-patched RMSNorm and SiluAndMul")


def ensure_registered() -> None:
    """Register NineToothed operators with vLLM (idempotent)."""
    global _registered
    if _registered:
        return
    _registered = True

    if not _try_register_oot():
        _monkey_patch()


# Auto-register on import
ensure_registered()
