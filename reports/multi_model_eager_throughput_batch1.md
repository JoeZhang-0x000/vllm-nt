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
| Qwen2.5-7B-Instruct | 128 | 7538.25 | 4309.50 | 5559.26 | 4313.66 | 4322.39 | 5563.97 | best NO_RMS_MM_FA (73.8% native) |
| gpt2 | 256 | 45935.69 | 35004.33 | 35287.70 | 35250.21 | 35443.79 | 35623.78 | best NO_RMS_MM_FA (77.6% native) |
| DeepSeek-R1-Distill-Qwen-7B | 96 | 6552.77 | 3767.45 | 4689.14 | 3768.33 | 3766.97 | 4686.02 | best NO_RMS (71.6% native) |

## Failures
- None
