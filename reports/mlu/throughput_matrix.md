# Offline Throughput Matrix on MLU

## Load Configuration
- dtype: `bfloat16`
- enforce_eager: `1`
- input_len/output_len: `64` / `64`
- warmup/measure iters: `1` / `3`
- max_model_len/max_num_batched_tokens: `512` / `512`
- max_num_configs: `10`
- mlu_visible_devices: `1`

## Summary Table
| model | batch | native tok/s | NT_All tok/s | NO_RMS tok/s | NO_MM tok/s | NO_FA tok/s | NO_RMS_MM_FA tok/s | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Llama-2-7b-chat-hf` | `128` | 5251.13 | 3108.28 | 3785.72 | 4028.77 | 3099.92 | 5215.48 | best NO_RMS_MM_FA (99.3% native) |
| `Qwen2.5-7B-Instruct` | `128` | 7519.98 | 4009.07 | 5569.03 | 5486.63 | 4314.40 | 7504.95 | best NO_RMS_MM_FA (99.8% native) |
| `MiniCPM4.1-8B` | `64` | 4166.13 | 749.86 | 801.60 | 896.30 | 747.34 | 972.83 | best NO_RMS_MM_FA (23.4% native) |
| `glm-4-9b` | `64` | 4317.21 | 2732.99 | 3387.56 | 3329.52 | 2734.75 | 4300.52 | best NO_RMS_MM_FA (99.6% native) |
| `Mistral-7B-Instruct-v0.3` | `128` | 6800.23 | 4020.62 | 5212.75 | 4928.45 | 4013.94 | 6756.30 | best NO_RMS_MM_FA (99.4% native) |
| `gpt2` | `256` | 45921.82 | 35584.29 | 35239.49 | 44570.90 | 35671.32 | 44631.57 | best NO_RMS_MM_FA (97.2% native) |
| `DeepSeek-R1-Distill-Qwen-7B` | `96` | 6587.16 | 3785.96 | 4699.16 | 4929.12 | 3794.93 | 6553.42 | best NO_RMS_MM_FA (99.5% native) |

## Failures
- None
