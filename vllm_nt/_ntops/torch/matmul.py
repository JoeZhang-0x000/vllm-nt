import torch

from vllm_nt._ntops.kernels import matmul as matmul_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def matmul(lhs, rhs, *, out_dtype=None):
    output = torch.empty(
        (lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=out_dtype or lhs.dtype
    )
    _cached_make(matmul_kernel.premake, lhs.dtype, rhs.dtype, output.dtype)(
        lhs, rhs, output
    )
    return output
