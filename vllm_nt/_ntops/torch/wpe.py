import torch

from vllm_nt._ntops.kernels import wpe as wpe_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def wpe(
    positions: torch.Tensor,
    weight: torch.Tensor,
    block_size_t: int = 64,
    block_size_h: int = 64,
) -> torch.Tensor:
    flat_positions = positions.reshape(-1, 1)
    output = torch.empty(
        (flat_positions.shape[0], weight.shape[-1]),
        dtype=weight.dtype,
        device=weight.device,
    )
    _cached_make(
        wpe_kernel.premake,
        positions_dtype=flat_positions.dtype,
        weight_dtype=weight.dtype,
        output_dtype=output.dtype,
        num_tokens=flat_positions.shape[0],
        num_positions=weight.shape[0],
        hidden_size=weight.shape[-1],
        block_size_t=block_size_t,
        block_size_h=block_size_h,
    )(flat_positions, weight, output)
    return output.reshape(*positions.shape, weight.shape[-1])
