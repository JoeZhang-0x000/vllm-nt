import math

import torch

from vllm_nt._ntops.kernels import rms_norm as rms_norm_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def rms_norm(input, normalized_shape, weight=None, eps=None):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    normalized_shape = tuple(normalized_shape)

    if weight is None:
        weight = torch.ones_like(input)
    else:
        weight = weight.expand_as(input)

    if eps is None:
        eps = torch.finfo(input.dtype).eps

    output = torch.empty_like(input)

    kernel = _cached_make(
        rms_norm_kernel.premake, input.ndim, len(normalized_shape)
    )

    kernel(input, weight, eps, output, math.prod(normalized_shape))

    return output
