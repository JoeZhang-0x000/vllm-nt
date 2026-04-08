import torch

from vllm_nt._ntops.kernels import rotary_emb as rotary_emb_kernel
from vllm_nt._ntops.torch.utils import _cached_make


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    _cached_make(rotary_emb_kernel.premake, cos.shape[-1])(x, cos, sin)
    return x
