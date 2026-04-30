"""Vendored subset of ntops — NineToothed operators for vLLM."""

#
# Import torch-backed wrappers before raw kernel modules.
# On MUSA, ninetoothed imports Triton, and Triton's MTGPU backend imports torch.
# If kernels are imported first, this can trigger a re-entrant torch import
# during package initialization and fail with duplicate TORCH_LIBRARY
# registration for the "triton" namespace.
try:
    from vllm_nt._ntops import torch, kernels
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    torch = None
    kernels = None

__all__ = ["kernels", "torch"]
