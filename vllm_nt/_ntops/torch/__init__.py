import importlib

from vllm_nt._ntops.torch.attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from vllm_nt._ntops.torch.embedding import embedding
from vllm_nt._ntops.torch.gelu import gelu
from vllm_nt._ntops.torch.kv_cache import store_kvcache
from vllm_nt._ntops.torch.layer_norm import layer_norm
from vllm_nt._ntops.torch.linear import linear
from vllm_nt._ntops.torch.matmul import matmul
from vllm_nt._ntops.torch.rotary_emb import apply_rotary_emb
from vllm_nt._ntops.torch.rms_norm import rms_norm
from vllm_nt._ntops.torch.sdpa import CausalVariant, scaled_dot_product_attention
from vllm_nt._ntops.torch.silu import silu
from vllm_nt._ntops.torch.wpe import wpe

_DELEGATED_NTOPS_EXPORTS = {
    "abs",
    "add",
    "addmm",
    "avg_pool2d",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bmm",
    "clamp",
    "conv2d",
    "cos",
    "div",
    "dropout",
    "embedding",
    "eq",
    "exp",
    "ge",
    "gt",
    "isinf",
    "isnan",
    "le",
    "lt",
    "max_pool2d",
    "mm",
    "mul",
    "ne",
    "neg",
    "pow",
    "relu",
    "rotary_position_embedding",
    "rsqrt",
    "sigmoid",
    "sin",
    "softmax",
    "sub",
    "tanh",
}


def __getattr__(name: str):
    if name not in _DELEGATED_NTOPS_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(importlib.import_module("ntops.torch"), name)
    except Exception as exc:
        raise AttributeError(
            f"module {__name__!r} could not delegate {name!r} to ntops.torch"
        ) from exc
    globals()[name] = value
    return value


__all__ = [
    "CausalVariant",
    "abs",
    "add",
    "addmm",
    "apply_rotary_emb",
    "avg_pool2d",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bmm",
    "clamp",
    "conv2d",
    "cos",
    "div",
    "dropout",
    "embedding",
    "eq",
    "exp",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "ge",
    "gelu",
    "gt",
    "isinf",
    "isnan",
    "layer_norm",
    "le",
    "lt",
    "linear",
    "matmul",
    "max_pool2d",
    "mm",
    "mul",
    "ne",
    "neg",
    "pow",
    "relu",
    "rms_norm",
    "rotary_position_embedding",
    "rsqrt",
    "scaled_dot_product_attention",
    "sigmoid",
    "silu",
    "sin",
    "softmax",
    "store_kvcache",
    "sub",
    "tanh",
    "wpe",
]
