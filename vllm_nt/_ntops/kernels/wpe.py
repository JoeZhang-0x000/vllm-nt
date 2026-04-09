import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE_T = Symbol("BLOCK_SIZE_T", meta=True)
BLOCK_SIZE_H = Symbol("BLOCK_SIZE_H", meta=True)


def arrangement(
    positions,
    weight,
    output,
    BLOCK_SIZE_T=BLOCK_SIZE_T,
    BLOCK_SIZE_H=BLOCK_SIZE_H,
):
    output_arranged = output.tile((BLOCK_SIZE_T, BLOCK_SIZE_H))
    positions_arranged = positions.tile((-1, -1)).expand(output_arranged.shape)
    weight_arranged = weight.tile((-1, -1)).expand(output_arranged.shape)
    return positions_arranged, weight_arranged, output_arranged


def application(positions, weight, output):
    token_offsets = output.offsets(0)[:, None]
    hidden_offsets = output.offsets(1)[None, :]
    valid_token = token_offsets < positions.source.shape[0]
    valid_hidden = hidden_offsets < weight.source.shape[1]
    position = positions.source[token_offsets, 0]
    output = ntl.where(  # noqa: F841
        valid_token,
        ntl.where(valid_hidden, weight.source[position, hidden_offsets], 0),
        0,
    )


def premake(
    positions_dtype=ninetoothed.int64,
    weight_dtype=None,
    output_dtype=None,
    num_tokens=None,
    num_positions=None,
    hidden_size=None,
    block_size_t=64,
    block_size_h=64,
):
    arrangement_ = functools.partial(
        arrangement,
        BLOCK_SIZE_T=block_size_t,
        BLOCK_SIZE_H=block_size_h,
    )
    positions = Tensor(2, dtype=positions_dtype)
    weight = Tensor(2, dtype=weight_dtype)
    output = Tensor(2, dtype=output_dtype)

    if num_tokens is not None:
        positions.shape = (num_tokens, 1)
        output.shape = (num_tokens, hidden_size)

    if num_positions is not None:
        weight.shape = (num_positions, hidden_size)

    return (
        arrangement_,
        application,
        (positions, weight, output),
    )
