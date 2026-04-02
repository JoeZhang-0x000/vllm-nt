import torch

from vllm_nt._ntops.kernels import gelu as gelu_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def gelu(input, approximate="tanh"):
    if approximate != "tanh":
        raise NotImplementedError("nt gelu currently supports only approximate='tanh'")
    output = torch.empty_like(input)
    _cached_make(gelu_kernel.premake, input.ndim)(input, output)
    return output
