# MM Hypothesis Benchmark (Current vllm-nt)

## Configuration
- examples repo: `/tmp/ninetoothed-examples`
- examples commit: `e873474d4b4de8e4fa427bf245da4a02512a68b1`
- device: `mlu`
- dtype: `bfloat16`

## Square Cases
| case | M | K | N | official NT ms | current matmul ms | current linear ms | current linear+bias ms | torch.mm ms | current matmul vs official | current matmul vs torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| square_1024 | 1024 | 1024 | 1024 | 0.0312 | 0.0433 | 0.0424 | 0.0495 | 0.0209 | 1.39x | 2.07x |
| square_2048 | 2048 | 2048 | 2048 | 0.1799 | 0.2762 | 0.2688 | 0.3050 | 0.1221 | 1.54x | 2.26x |
| square_4096 | 4096 | 4096 | 4096 | 1.1712 | 1.9486 | 1.9063 | 2.0630 | 0.9330 | 1.66x | 2.09x |

## Model Cases
| case | M | K | N | official NT ms | current matmul ms | current linear ms | current linear+bias ms | torch.mm ms | current matmul vs official | current matmul vs torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2_fc | 8192 | 768 | 768 | 0.1488 | 0.2208 | 0.2153 | 0.2673 | 0.0720 | 1.48x | 3.07x |
| gpt2_mlp_up | 8192 | 768 | 3072 | 0.6041 | 0.8187 | 0.8005 | 0.9646 | 0.2668 | 1.36x | 3.07x |
| gpt2_mlp_down | 8192 | 3072 | 768 | 0.3609 | 0.5834 | 0.5890 | 0.6424 | 0.2640 | 1.62x | 2.21x |
| gpt2_lm_head | 256 | 768 | 50257 | 0.4034 | 0.4562 | 0.4438 | 0.5636 | 0.1777 | 1.13x | 2.57x |
| qwen3_attn | 8192 | 1024 | 1024 | 0.2332 | 0.3394 | 0.3307 | 0.3863 | 0.1239 | 1.45x | 2.74x |
| qwen3_mlp_up | 8192 | 1024 | 3072 | 0.7114 | 1.0195 | 0.9920 | 1.1633 | 0.3561 | 1.43x | 2.86x |
| qwen3_mlp_down | 8192 | 1536 | 1024 | 0.2992 | 0.4490 | 0.4434 | 0.5144 | 0.1833 | 1.50x | 2.45x |
| qwen3_lm_head | 256 | 1024 | 151936 | 1.3513 | 1.5950 | 1.5555 | 1.8194 | 0.5375 | 1.18x | 2.97x |

## Summary
- geometric mean current matmul vs official NT (square): `1.52x`
- geometric mean current matmul vs official NT (model): `1.39x`
- geometric mean current linear+bias vs current linear (model): `1.18x`
