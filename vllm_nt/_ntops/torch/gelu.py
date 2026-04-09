import importlib

import torch
import torch.nn.functional as F


def gelu(input, approximate="tanh"):
    try:
        return importlib.import_module("ntops.torch").gelu(
            input, approximate=approximate
        )
    except Exception:
        return F.gelu(input, approximate=approximate)
