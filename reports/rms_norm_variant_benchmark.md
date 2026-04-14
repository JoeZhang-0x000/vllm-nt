# RMSNorm Variant Benchmark on MLU

## Configuration
- examples repo: `/tmp/ninetoothed-examples`
- dtype: `bfloat16`
- device: `MLU`

## Results
| case | M | N | torch ms | current vllm-nt ms | official fused NT ms | official plain NT ms | current vs torch | current vs official fused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen_decode_like | 512 | 1024 | 0.0393 | 0.4505 | 0.4449 | 0.3641 | 11.46x | 1.01x |
| qwen_large_m | 8192 | 1024 | 0.3072 | 7.2112 | 7.1871 | 5.7343 | 23.48x | 1.00x |
| gpt2_like | 512 | 768 | 0.0348 | 0.4511 | 0.4519 | 0.3661 | 12.96x | 1.00x |

## Reading
- The current `vllm-nt` RMSNorm fast path now matches the official `fused_rms_norm` implementation closely.
- The official fused implementation is still far slower than `torch.nn.functional.rms_norm` on these MLU shapes, so RMSNorm remains a backend/kernel-level bottleneck rather than only a `vllm-nt` wrapper issue.
