# MM Hypothesis Benchmark (Current vllm-nt)

## Configuration
- examples repo: `/tmp/ninetoothed-examples`
- examples commit: `e873474d4b4de8e4fa427bf245da4a02512a68b1`
- device: `mlu`
- dtype: `bfloat16`

## Square Cases
| case | M | K | N | official NT ms | current matmul ms | current linear ms | current linear+bias ms | torch.mm ms | current matmul vs official | current matmul vs torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| square_1024 | 1024 | 1024 | 1024 | 0.0312 | 0.8653 | 0.8767 | 0.8922 | 0.0209 | 27.76x | 41.45x |
| square_2048 | 2048 | 2048 | 2048 | 0.1797 | 6.3957 | 6.4350 | 6.5048 | 0.1222 | 35.60x | 52.36x |
| square_4096 | 4096 | 4096 | 4096 | 1.1755 | 48.7850 | 48.3355 | 48.5665 | 0.9299 | 41.50x | 52.47x |

## Model Cases
| case | M | K | N | official NT ms | current matmul ms | current linear ms | current linear+bias ms | torch.mm ms | current matmul vs official | current matmul vs torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2_fc | 8192 | 768 | 768 | 0.1489 | 4.0919 | 4.0862 | 4.1659 | 0.0720 | 27.48x | 56.85x |
| gpt2_mlp_up | 8192 | 768 | 3072 | 0.6042 | 16.5582 | 16.3053 | 16.6764 | 0.2660 | 27.41x | 62.24x |
| gpt2_mlp_down | 8192 | 3072 | 768 | 0.3604 | 13.6754 | 13.8137 | 13.9006 | 0.2640 | 37.94x | 51.80x |
| gpt2_lm_head | 256 | 768 | 50257 | 0.4024 | 8.7367 | 8.3056 | 9.2072 | 0.1779 | 21.71x | 49.10x |
| qwen3_attn | 8192 | 1024 | 1024 | 0.2337 | 6.9498 | 6.9944 | 7.1061 | 0.1239 | 29.73x | 56.10x |
| qwen3_mlp_up | 8192 | 1024 | 3072 | 0.7126 | 20.9212 | 20.8980 | 21.2253 | 0.3557 | 29.36x | 58.81x |
| qwen3_mlp_down | 8192 | 1536 | 1024 | 0.2993 | 9.6701 | 9.7119 | 9.8299 | 0.1831 | 32.31x | 52.83x |
| qwen3_lm_head | 256 | 1024 | 151936 | 1.3539 | 32.6157 | 32.3513 | 32.8487 | 0.5374 | 24.09x | 60.69x |

## Summary
- geometric mean current matmul vs official NT (square): `34.48x`
- geometric mean current matmul vs official NT (model): `28.39x`
- geometric mean current linear+bias vs current linear (model): `1.03x`
