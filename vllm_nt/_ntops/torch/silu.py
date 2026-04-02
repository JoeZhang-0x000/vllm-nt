import torch

from vllm_nt._ntops.kernels import silu as silu_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def silu(input, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)

    kernel = _cached_make(silu_kernel.premake, input.ndim)

    kernel(input, output)

    return output
