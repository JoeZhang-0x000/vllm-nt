import torch
import torch.nn.functional as F

from vllm_nt._ntops.kernels import wpe as wpe_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def wpe(
    positions: torch.Tensor,
    weight: torch.Tensor,
    block_size_t: int = 64,
    block_size_h: int = 64,
    return_status: bool = False,
    fallback: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, str]:
    flat_positions = positions.reshape(-1, 1)
    output = torch.empty(
        (flat_positions.shape[0], weight.shape[-1]),
        dtype=weight.dtype,
        device=weight.device,
    )
    try:
        _cached_make(
            wpe_kernel.premake,
            positions_dtype=flat_positions.dtype,
            weight_dtype=weight.dtype,
            output_dtype=output.dtype,
            block_size_t=block_size_t,
            block_size_h=block_size_h,
        )(flat_positions, weight, output)
    except Exception:
        if not fallback:
            raise
        fallback = F.embedding(positions, weight)
        return (fallback, "fallback") if return_status else fallback
    result = output.reshape(*positions.shape, weight.shape[-1])
    return (result, "kernel") if return_status else result
