import torch

from vllm_nt._ntops.torch.embedding import embedding


def wpe(
    positions: torch.Tensor,
    weight: torch.Tensor,
    block_size_t: int = 64,
    block_size_h: int = 64,
) -> torch.Tensor:
    return embedding(
        positions,
        weight,
        block_size_t=block_size_t,
        block_size_h=block_size_h,
    )
