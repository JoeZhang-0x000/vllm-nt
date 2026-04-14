# MM Hypothesis Benchmark (Current vllm-nt)

## Configuration
- examples repo: `/tmp/ninetoothed-examples`
- examples commit: `e873474d4b4de8e4fa427bf245da4a02512a68b1`
- device: `mlu`
- dtype: `bfloat16`

## Square Cases
| case | M | K | N | official NT ms | current matmul ms | current linear ms | current linear+bias ms | torch.mm ms | current matmul vs official | current matmul vs torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| square_1024 | 1024 | 1024 | 1024 | 0.0312 | 0.2088 | 0.2037 | 0.2177 | 0.0206 | 6.69x | 10.13x |
| square_2048 | 2048 | 2048 | 2048 | 0.1799 | 1.3125 | 1.2888 | 1.3406 | 0.1221 | 7.29x | 10.75x |
| square_4096 | 4096 | 4096 | 4096 | 1.1734 | 9.0408 | 8.9036 | 9.1619 | 0.9330 | 7.70x | 9.69x |

## Model Cases
| case | M | K | N | official NT ms | current matmul ms | current linear ms | current linear+bias ms | torch.mm ms | current matmul vs official | current matmul vs torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2_fc | 8192 | 768 | 768 | 0.1493 | 1.0632 | 1.0358 | 1.1083 | 0.0720 | 7.12x | 14.77x |
| gpt2_mlp_up | 8192 | 768 | 3072 | 0.6051 | 4.3116 | 4.1589 | 4.4854 | 0.2668 | 7.13x | 16.16x |
| gpt2_mlp_down | 8192 | 3072 | 768 | 0.3601 | 2.6739 | 2.6547 | 2.7401 | 0.2639 | 7.43x | 10.13x |
| gpt2_lm_head | 256 | 768 | 50257 | 0.4023 | 2.3332 | 2.1349 | 2.7811 | 0.1778 | 5.80x | 13.12x |
| qwen3_attn | 8192 | 1024 | 1024 | 0.2336 | 1.6568 | 1.6146 | 1.7207 | 0.1238 | 7.09x | 13.38x |
| qwen3_mlp_up | 8192 | 1024 | 3072 | 0.7127 | 5.0034 | 4.8539 | 5.1704 | 0.3563 | 7.02x | 14.04x |
| qwen3_mlp_down | 8192 | 1536 | 1024 | 0.2995 | 2.1392 | 2.0982 | 2.2054 | 0.1831 | 7.14x | 11.69x |
| qwen3_lm_head | 256 | 1024 | 151936 | 1.3528 | 7.8277 | 7.5308 | 8.0433 | 0.5376 | 5.79x | 14.56x |

## Summary
- geometric mean current matmul vs official NT (square): `7.22x`
- geometric mean current matmul vs official NT (model): `6.79x`
- geometric mean current linear+bias vs current linear (model): `1.09x`
