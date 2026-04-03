import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from vllm_nt._ntops.kernels.element_wise import arrangement


def application(input, output):
    x = ntl.cast(input, ntl.float32)
    inner = x * (0.7978845608028654 * (1 + 0.044715 * x * x))
    exp2 = ntl.exp(2 * inner)
    output = 0.5 * x * (1 + (exp2 - 1) / (exp2 + 1))  # noqa: F841


def premake(ndim, dtype=None, block_size=None):
    return (
        functools.partial(arrangement, block_size=block_size),
        application,
        (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype)),
    )
