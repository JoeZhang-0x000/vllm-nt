# Official ninetoothed-examples MM Benchmark on MLU

## Configuration
- examples repo: `/tmp/ninetoothed-examples`
- examples commit: `e873474d4b4de8e4fa427bf245da4a02512a68b1`
- benchmark source: `tests/test_benchmarks.py::TestMMBenchmark`
- device: `mlu`
- dtype: `bfloat16`
- shape sweep: `m=n=k=2^i`, `i in [3, 12]`

## Summary
- geometric mean slowdown vs `torch.mm`: `1.06x`
- geometric mean slowdown vs official Triton MM: `0.24x`

## Results
| size | NineToothed ms | torch.mm ms | official Triton ms | NT vs torch | NT vs Triton |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0028 | 0.0023 | 0.0042 | 1.25x | 0.67x |
| 16 | 0.0023 | 0.0023 | 0.0040 | 1.03x | 0.58x |
| 32 | 0.0022 | 0.0025 | 0.0046 | 0.87x | 0.48x |
| 64 | 0.0024 | 0.0030 | 0.0058 | 0.82x | 0.42x |
| 128 | 0.0028 | 0.0031 | 0.0081 | 0.92x | 0.35x |
| 256 | 0.0037 | 0.0041 | 0.0138 | 0.89x | 0.27x |
| 512 | 0.0083 | 0.0102 | 0.0462 | 0.82x | 0.18x |
| 1024 | 0.0312 | 0.0206 | 0.3080 | 1.51x | 0.10x |
| 2048 | 0.1795 | 0.1219 | 2.3850 | 1.47x | 0.08x |
| 4096 | 1.1749 | 0.9264 | 18.7600 | 1.27x | 0.06x |

## Interpretation
- This reuses the official `ninetoothed-examples` MM benchmark shape sweep, but runs it on local `MLU` instead of the upstream hard-coded CUDA path.
- On square MM shapes, NineToothed is close to `torch.mm` on MLU and substantially faster than the official Triton MM implementation from the examples repo.
- This does not match the severe slowdown seen in the `vllm-nt` model-shape MatMul microbenchmark, which suggests the main problem is shape-specific or integration-specific rather than a generic square-GEMM regression.
