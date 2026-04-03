import torch

from vllm_nt._ntops.kernels import linear as linear_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def linear(
    input: torch.Tensor, other: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    output = torch.empty(
        (input.shape[0], other.shape[0]), dtype=input.dtype, device=input.device
    )
    if bias is not None:
        _cached_make(
            linear_kernel.premake, input.dtype, other.dtype, output.dtype, True
        )(input, other, output, bias.view(1, -1))
    else:
        _cached_make(
            linear_kernel.premake, input.dtype, other.dtype, output.dtype, False
        )(input, other, output)
    return output
