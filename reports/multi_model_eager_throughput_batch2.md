# Multi-Model Eager Throughput Summary

## Configuration
- models root: `/data02/jiangqiu/models`
- model source list: `/data02/jiangqiu/models/down_models.sh`
- dtype: `bfloat16`
- enforce_eager: `1`
- max_num_configs: `10`
- max_model_len/max_num_batched_tokens: `512` / `512`
- input_len/output_len: `64` / `64`
- warmup/measure iters: `1` / `3`
- mlu_visible_devices: `1`

## Summary Table
| model | batch | native tok/s | NT_All tok/s | NO_RMS tok/s | NO_MM tok/s | NO_FA tok/s | NO_RMS_MM_FA tok/s | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Llama-2-7b-chat-hf | 128 | 5236.80 | 3085.83 | 3780.82 | 3088.50 | 3086.37 | 3778.86 | best NO_RMS (72.2% native) |
| MiniCPM4.1-8B | 64 | 4179.73 | 739.00 | 791.21 | 737.82 | 739.95 | 795.62 | best NO_RMS_MM_FA (19.0% native) |
| glm-4-9b | 64 | 4312.44 | 2709.42 | 3381.08 | 2715.41 | 2712.16 | 3380.44 | best NO_RMS (78.4% native) |
| Mistral-7B-Instruct-v0.3 | 128 | 6779.80 | 3970.97 | 5196.05 | 3973.14 | 3969.46 | 5198.06 | best NO_RMS_MM_FA (76.7% native) |

## Failures
- None
