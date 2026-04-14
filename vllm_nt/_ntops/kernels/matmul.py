import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()
BLOCK_SIZE_K = ninetoothed.block_size()


def arrangement(
    lhs,
    rhs,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
    lhs_tiled = (
        lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    rhs_tiled = (
        rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)
    return lhs_tiled, rhs_tiled, output_tiled


def application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])
    output = accumulator.to(output.dtype)  # noqa: F841


def premake(
    lhs_dtype=None,
    rhs_dtype=None,
    output_dtype=None,
    block_size_m=BLOCK_SIZE_M,
    block_size_n=BLOCK_SIZE_N,
    block_size_k=BLOCK_SIZE_K,
):
    arrangement_ = functools.partial(
        arrangement,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
    )
    return (
        arrangement_,
        application,
        (
            Tensor(2, dtype=lhs_dtype),
            Tensor(2, dtype=rhs_dtype),
            Tensor(2, dtype=output_dtype),
        ),
    )
