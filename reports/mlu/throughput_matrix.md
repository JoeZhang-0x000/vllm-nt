# Offline Throughput Matrix on MLU

## Load Configuration
- dtype: `bfloat16`
- enforce_eager: `1`
- prompt/output lengths: `64` / `64` tokens
- warmup/measure iters: `2` / `5`
- primary metric: `output tok/s`
- secondary metrics: `req/s`, `total tok/s`, `mean iter sec`, `std iter sec`
- prompts are passed as `prompt_token_ids`; detokenization is disabled during measurement
- `max_model_len` defaults to `input_len + output_len` unless overridden
- `max_num_batched_tokens` defaults to `batch_size * input_len` unless overridden
- max_num_configs: `10`
- mlu_visible_devices: `1`

## Summary Table
| model | batch | max_model_len | max_num_batched_tokens | native output tok/s | NT_All output tok/s | NO_RMS output tok/s | NO_MM output tok/s | NO_FA output tok/s | NO_RMS_MM_FA output tok/s | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Llama-2-7b-chat-hf` | `128` | `128` | `8192` | 3240.76 | 2005.74 | 2335.92 | 2625.66 | 2004.68 | 3217.75 | best NO_RMS_MM_FA (99.3% native) |
| `Qwen2.5-7B-Instruct` | `128` | `128` | `8192` | 5239.90 | 3002.63 | 3300.91 | 3935.14 | 2994.41 | 5227.65 | best NO_RMS_MM_FA (99.8% native) |
| `MiniCPM4.1-8B` | `64` | `128` | `4096` | 2661.34 | FAIL | FAIL | FAIL | FAIL | FAIL | no NT result |
| `glm-4-9b` | `64` | `128` | `4096` | 2852.52 | 1838.88 | 2181.39 | 2288.74 | 1841.09 | 2841.69 | best NO_RMS_MM_FA (99.6% native) |
| `Mistral-7B-Instruct-v0.3` | `128` | `128` | `8192` | 4981.21 | 2893.60 | 3627.99 | 3686.11 | 2892.24 | 4966.05 | best NO_RMS_MM_FA (99.7% native) |
| `gpt2` | `256` | `128` | `16384` | 27336.08 | 24239.08 | 24353.31 | 28307.10 | 24274.56 | 28246.27 | best NO_MM (103.6% native) |
| `DeepSeek-R1-Distill-Qwen-7B` | `96` | `128` | `6144` | 4366.88 | 2516.54 | 2980.93 | 3425.24 | 2509.24 | 4360.65 | best NO_RMS_MM_FA (99.9% native) |

## Detailed Metrics
### `Llama-2-7b-chat-hf`
| mode | output tok/s | req/s | total tok/s | mean iter sec | std iter sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 3240.76 | 50.64 | 6481.52 | 2.5278 | 0.0019 |
| NT_All | 2005.74 | 31.34 | 4011.48 | 4.0843 | 0.0012 |
| NO_RMS | 2335.92 | 36.50 | 4671.83 | 3.5070 | 0.0011 |
| NO_MM | 2625.66 | 41.03 | 5251.32 | 3.1200 | 0.0016 |
| NO_FA | 2004.68 | 31.32 | 4009.36 | 4.0864 | 0.0039 |
| NO_RMS_MM_FA | 3217.75 | 50.28 | 6435.49 | 2.5459 | 0.0071 |

### `Qwen2.5-7B-Instruct`
| mode | output tok/s | req/s | total tok/s | mean iter sec | std iter sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 5239.90 | 81.87 | 10479.80 | 1.5634 | 0.0013 |
| NT_All | 3002.63 | 46.92 | 6005.26 | 2.7283 | 0.0004 |
| NO_RMS | 3300.91 | 51.58 | 6601.82 | 2.4817 | 0.0704 |
| NO_MM | 3935.14 | 61.49 | 7870.28 | 2.0818 | 0.0015 |
| NO_FA | 2994.41 | 46.79 | 5988.81 | 2.7358 | 0.0078 |
| NO_RMS_MM_FA | 5227.65 | 81.68 | 10455.31 | 1.5671 | 0.0009 |

### `MiniCPM4.1-8B`
| mode | output tok/s | req/s | total tok/s | mean iter sec | std iter sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 2661.34 | 41.58 | 5322.69 | 1.5391 | 0.0117 |
| NT_All | FAIL | FAIL | FAIL | FAIL | FAIL |
| NO_RMS | FAIL | FAIL | FAIL | FAIL | FAIL |
| NO_MM | FAIL | FAIL | FAIL | FAIL | FAIL |
| NO_FA | FAIL | FAIL | FAIL | FAIL | FAIL |
| NO_RMS_MM_FA | FAIL | FAIL | FAIL | FAIL | FAIL |

### `glm-4-9b`
| mode | output tok/s | req/s | total tok/s | mean iter sec | std iter sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 2852.52 | 44.57 | 5705.04 | 1.4359 | 0.0012 |
| NT_All | 1838.88 | 28.73 | 3677.75 | 2.2274 | 0.0012 |
| NO_RMS | 2181.39 | 34.08 | 4362.78 | 1.8777 | 0.0007 |
| NO_MM | 2288.74 | 35.76 | 4577.48 | 1.7896 | 0.0044 |
| NO_FA | 1841.09 | 28.77 | 3682.18 | 2.2248 | 0.0027 |
| NO_RMS_MM_FA | 2841.69 | 44.40 | 5683.37 | 1.4414 | 0.0013 |

### `Mistral-7B-Instruct-v0.3`
| mode | output tok/s | req/s | total tok/s | mean iter sec | std iter sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 4981.21 | 77.83 | 9962.43 | 1.6446 | 0.0020 |
| NT_All | 2893.60 | 45.21 | 5787.20 | 2.8311 | 0.0008 |
| NO_RMS | 3627.99 | 56.69 | 7255.98 | 2.2580 | 0.0005 |
| NO_MM | 3686.11 | 57.60 | 7372.21 | 2.2224 | 0.0023 |
| NO_FA | 2892.24 | 45.19 | 5784.47 | 2.8324 | 0.0013 |
| NO_RMS_MM_FA | 4966.05 | 77.59 | 9932.10 | 1.6496 | 0.0006 |

### `gpt2`
| mode | output tok/s | req/s | total tok/s | mean iter sec | std iter sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 27336.08 | 427.13 | 54672.16 | 0.5994 | 0.0013 |
| NT_All | 24239.08 | 378.74 | 48478.16 | 0.6759 | 0.0014 |
| NO_RMS | 24353.31 | 380.52 | 48706.62 | 0.6728 | 0.0002 |
| NO_MM | 28307.10 | 442.30 | 56614.19 | 0.5788 | 0.0006 |
| NO_FA | 24274.56 | 379.29 | 48549.13 | 0.6749 | 0.0007 |
| NO_RMS_MM_FA | 28246.27 | 441.35 | 56492.53 | 0.5800 | 0.0019 |

### `DeepSeek-R1-Distill-Qwen-7B`
| mode | output tok/s | req/s | total tok/s | mean iter sec | std iter sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 4366.88 | 68.23 | 8733.75 | 1.4070 | 0.0051 |
| NT_All | 2516.54 | 39.32 | 5033.09 | 2.4414 | 0.0016 |
| NO_RMS | 2980.93 | 46.58 | 5961.86 | 2.0611 | 0.0009 |
| NO_MM | 3425.24 | 53.52 | 6850.48 | 1.7937 | 0.0017 |
| NO_FA | 2509.24 | 39.21 | 5018.49 | 2.4485 | 0.0073 |
| NO_RMS_MM_FA | 4360.65 | 68.14 | 8721.29 | 1.4090 | 0.0014 |


## Failures
- `MiniCPM4.1-8B` / `NT_All`: rgs, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/site-packages/vllm/model_executor/models/minicpm.py", line 472, in forward
[rank0]:     hidden_states, residual = layer(
[rank0]:                               ^^^^^^
[rank0]:   File "/usr/loc
- `MiniCPM4.1-8B` / `NO_RMS`: rgs, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/site-packages/vllm/model_executor/models/minicpm.py", line 472, in forward
[rank0]:     hidden_states, residual = layer(
[rank0]:                               ^^^^^^
[rank0]:   File "/usr/loc
- `MiniCPM4.1-8B` / `NO_MM`: rgs, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/site-packages/vllm/model_executor/models/minicpm.py", line 472, in forward
[rank0]:     hidden_states, residual = layer(
[rank0]:                               ^^^^^^
[rank0]:   File "/usr/loc
- `MiniCPM4.1-8B` / `NO_FA`: rgs, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/site-packages/vllm/model_executor/models/minicpm.py", line 472, in forward
[rank0]:     hidden_states, residual = layer(
[rank0]:                               ^^^^^^
[rank0]:   File "/usr/loc
- `MiniCPM4.1-8B` / `NO_RMS_MM_FA`: rgs, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/site-packages/vllm/model_executor/models/minicpm.py", line 472, in forward
[rank0]:     hidden_states, residual = layer(
[rank0]:                               ^^^^^^
[rank0]:   File "/usr/loc
