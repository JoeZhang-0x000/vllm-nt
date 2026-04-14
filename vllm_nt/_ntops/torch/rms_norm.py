import math

import torch

from vllm_nt._ntops.kernels import fused_rms_norm as fused_rms_norm_kernel
from vllm_nt._ntops.kernels import rms_norm as rms_norm_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def _fused_rms_norm_last_dim(input, normalized_shape, weight, eps):
    hidden_size = normalized_shape[0]
    input_2d = input.reshape(-1, hidden_size)

    if weight is None:
        weight_1d = torch.ones((hidden_size,), dtype=input.dtype, device=input.device)
    else:
        if weight.numel() == hidden_size:
            weight_1d = weight.reshape(hidden_size)
        else:
            weight_1d = weight.reshape(-1, hidden_size)[0]

    weight_2d = weight_1d.reshape(1, hidden_size).expand_as(input_2d)
    output_2d = torch.empty_like(input_2d)

    fused_rms_norm_kernel.kernel(
        input_2d,
        weight_2d,
        eps,
        output_2d,
        BLOCK_SIZE=hidden_size,
    )

    return output_2d.reshape_as(input)


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

    if len(normalized_shape) == 1 and normalized_shape[0] == input.shape[-1]:
        return _fused_rms_norm_last_dim(input, normalized_shape, weight, eps)

    output = torch.empty_like(input)

    kernel = _cached_make(
        rms_norm_kernel.premake, input.ndim, len(normalized_shape)
    )

    kernel(input, weight, eps, output, math.prod(normalized_shape))

    return output
