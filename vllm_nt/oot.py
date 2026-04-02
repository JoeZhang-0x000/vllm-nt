"""OOT (out-of-tree) layer overrides for vLLM.

Strategy:
  1. Try @register_oot() — the clean, official vLLM extension path.
  2. If that fails (API missing or version mismatch), fall back to
     monkey-patching forward_oot / forward_native directly.

Grid chunking:
  Some hardware (e.g. MLU) has a grid size limit of 65535.
  When the input row count exceeds this, we split into chunks
  and process each chunk separately.
"""

import logging

import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_nt._ntops.torch import rms_norm as nt_rms_norm
from vllm_nt._ntops.torch import silu as nt_silu

logger = logging.getLogger("vllm_nt")

# Hardware grid limit — conservative default for MLU / other domestic cards.
_MAX_GRID_SIZE = 65535


# ── Chunked kernel calls ─────────────────────────────────────────


def _chunked_rms_norm(x, normalized_shape, weight, eps):
    """Call nt_rms_norm in chunks if row count exceeds grid limit."""
    # Flatten leading dims to (N, hidden_size) for chunking
    orig_shape = x.shape
    flat = x.reshape(-1, orig_shape[-1])
    n_rows = flat.shape[0]

    if n_rows <= _MAX_GRID_SIZE:
        return nt_rms_norm(x, normalized_shape=normalized_shape,
                           weight=weight, eps=eps)

    chunks = []
    for start in range(0, n_rows, _MAX_GRID_SIZE):
        end = min(start + _MAX_GRID_SIZE, n_rows)
        chunk = flat[start:end]
        chunks.append(
            nt_rms_norm(chunk, normalized_shape=normalized_shape,
                        weight=weight, eps=eps)
        )
    return torch.cat(chunks, dim=0).reshape(orig_shape)


def _chunked_silu(x):
    """Call nt_silu in chunks if element count exceeds grid limit."""
    orig_shape = x.shape
    flat = x.reshape(-1)
    n = flat.shape[0]

    if n <= _MAX_GRID_SIZE:
        return nt_silu(x)

    # For element-wise ops, chunk on flattened elements
    flat_out = torch.empty_like(flat)
    for start in range(0, n, _MAX_GRID_SIZE):
        end = min(start + _MAX_GRID_SIZE, n)
        flat_out[start:end] = nt_silu(flat[start:end])
    return flat_out.reshape(orig_shape)


# ── NineToothed forward implementations ──────────────────────────


def _nt_rms_norm_forward(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if residual is not None:
        x = x + residual
        residual = x.to(x.dtype)

    out = _chunked_rms_norm(
        x,
        normalized_shape=self.hidden_size,
        weight=self.weight if self.has_weight else None,
        eps=self.variance_epsilon,
    )

    if residual is not None:
        return out, residual
    return out


def _nt_silu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return _chunked_silu(x[..., :d]) * x[..., d:]


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
