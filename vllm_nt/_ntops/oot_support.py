from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from vllm_nt._ntops.torch import matmul as nt_matmul
from vllm_nt._ntops.torch import rms_norm as nt_rms_norm


@dataclass
class OperatorStats:
    hits: int = 0
    logged: bool = False
    registered_via: str | None = None


def norm(
    self,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    residual: torch.Tensor | None = None,
    gemma: bool = False,
):
    if residual is not None:
        x = x + residual
        residual = x if gemma else x.to(x.dtype)
    out = nt_rms_norm(
        x,
        normalized_shape=weight.shape[0] if weight is not None else self.hidden_size,
        weight=weight,
        eps=self.variance_epsilon,
    )
    return (out, residual) if residual is not None else out


def act_and_mul(
    x: torch.Tensor, act: Callable[[torch.Tensor], torch.Tensor], reverse: bool = False
) -> torch.Tensor:
    d = x.shape[-1] // 2
    left, right = (x[..., d:], x[..., :d]) if reverse else (x[..., :d], x[..., d:])
    return act(left) * right


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    output = nt_matmul(x.reshape(-1, x.shape[-1]), weight.T, out_dtype=x.dtype).reshape(
        *x.shape[:-1], weight.shape[0]
    )
    return output if bias is None else output + bias


def embedding(layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
    return F.embedding(input_, layer.weight)
