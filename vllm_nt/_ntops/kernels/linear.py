import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor

from vllm_nt._ntops.kernels.matmul import BLOCK_SIZE_K, BLOCK_SIZE_M, BLOCK_SIZE_N


def arrangement(
    input,
    other,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
    input_arranged = (
        input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_arranged.shape[1]))
    )
    input_arranged.dtype = input_arranged.dtype.squeeze(0)
    other_arranged = (
        other.permute((1, 0))
        .tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_arranged.shape[0], -1))
    )
    other_arranged.dtype = other_arranged.dtype.squeeze(1)
    return input_arranged, other_arranged, output_arranged


def arrangement_with_bias(
    input,
    other,
    output,
    bias,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    input_arranged, other_arranged, output_arranged = arrangement(
        input, other, output, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    bias_arranged = bias.tile((1, BLOCK_SIZE_N)).expand((output_arranged.shape[0], -1))
    return input_arranged, other_arranged, output_arranged, bias_arranged


def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])
    output = accumulator  # noqa: F841


def application_with_bias(input, other, output, bias):
    application(input, other, output)
    output = output + bias  # noqa: F841


def premake(
    input_dtype=None,
    other_dtype=None,
    output_dtype=None,
    bias=False,
    block_size_m=BLOCK_SIZE_M,
    block_size_n=BLOCK_SIZE_N,
    block_size_k=BLOCK_SIZE_K,
):
    arrangement_fn = functools.partial(
        arrangement_with_bias if bias else arrangement,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
    )
    application_fn = application_with_bias if bias else application
    tensors = [
        Tensor(2, dtype=input_dtype),
        Tensor(2, dtype=other_dtype),
        Tensor(2, dtype=output_dtype),
    ]
    if bias:
        tensors.append(Tensor(2, dtype=output_dtype))
    return arrangement_fn, application_fn, tuple(tensors)
