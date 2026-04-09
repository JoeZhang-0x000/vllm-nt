import torch

from vllm_nt._ntops.kernels import wpe as embedding_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def embedding(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    block_size_t: int = 64,
    block_size_h: int = 64,
) -> torch.Tensor:
    flat_input_ids = input_ids.reshape(-1, 1)
    output = torch.empty(
        (flat_input_ids.shape[0], weight.shape[-1]),
        dtype=weight.dtype,
        device=weight.device,
    )
    _cached_make(
        embedding_kernel.premake,
        positions_dtype=flat_input_ids.dtype,
        weight_dtype=weight.dtype,
        output_dtype=output.dtype,
        num_tokens=flat_input_ids.shape[0],
        num_positions=weight.shape[0],
        hidden_size=weight.shape[-1],
        block_size_t=block_size_t,
        block_size_h=block_size_h,
    )(flat_input_ids, weight, output)
    return output.reshape(*input_ids.shape, weight.shape[-1])
