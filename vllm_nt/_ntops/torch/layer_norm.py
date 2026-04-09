import importlib

import torch
import torch.nn.functional as F


def layer_norm(
    input: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(normalized_shape)

    try:
        return importlib.import_module("ntops.torch").layer_norm(
            input,
            normalized_shape,
            weight=weight,
            bias=bias,
            eps=eps,
        )
    except Exception:
        return F.layer_norm(
            input,
            normalized_shape,
            weight,
            bias,
            eps,
        )
