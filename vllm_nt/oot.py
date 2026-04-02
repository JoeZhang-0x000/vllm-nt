"""OOT (out-of-tree) layer overrides for vLLM.

Importing this module registers NineToothed implementations as
replacements for vLLM's built-in RMSNorm and SiluAndMul layers
via the @CustomOp.register_oot() mechanism.
"""

import torch

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_nt._ntops.torch import rms_norm as nt_rms_norm
from vllm_nt._ntops.torch import silu as nt_silu


@RMSNorm.register_oot(name="RMSNorm")
class NTRMSNorm(RMSNorm):
    """RMSNorm using NineToothed kernel."""

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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


@SiluAndMul.register_oot(name="SiluAndMul")
class NTSiluAndMul(SiluAndMul):
    """SiluAndMul using NineToothed kernel."""

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        return nt_silu(gate) * up
