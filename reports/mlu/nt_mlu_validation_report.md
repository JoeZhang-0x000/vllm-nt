# NT MLU Validation Report

## Configuration
- dtype: `bfloat16`
- tensor_parallel_size: `1`
- gpu_memory_utilization: `0.7`
- max_num_configs: `10`
- max_num_configs_mode: `(explicit)`
- max_model_len: `512`
- max_num_batched_tokens: `512`
- mlu_visible_devices: `1`
- v1_multiprocessing: `0`
- accuracy max tokens: `32`
- throughput input/output lengths: `64` / `64`
- throughput warmup/measure iterations: `1` / `3`

## Accuracy Prompts
1. Answer in one short sentence: what is the moon?
2. Answer with one word only: the capital of France is
3. Translate to English: 今天天气很好，我们一起去散步。

## Missing Models
- `llama2_7b`: No cached snapshot found under /home/tiger/.cache/huggingface/hub

## Model Results

### `qwen3_0.6b`
- model id: `Qwen/Qwen3-0.6B`
- cache path: `/home/tiger/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`

#### Accuracy
- mode: `nt_all_off`
- hit_ops: `None`
- prompt: Answer in one short sentence: what is the moon?
- output: ' \n\nThe moon is a natural satellite of Earth, orbiting around Earth, and it is the only celestial body that has been observed to be in orbit around Earth'
- prompt: Answer with one word only: the capital of France is
- output: '...? \n\nThe answer is "Paris". \n\nBut why is that the case? What is the reasoning behind this answer? \n\nThe answer is "Paris".'
- prompt: Translate to English: 今天天气很好，我们一起去散步。
- output: ' 今天天气很好，我们一起去散步。 今天天气很好，我们一起去散步。 今天天气很好，我们一起去散步。 今天天气很好，'

- mode: `nt_all_on`
- hit_ops: `RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, PagedAttentionPrefill, PagedAttentionDecode, RoPE, TopKTopP, RandomSample`
- prompt: Answer in one short sentence: what is the moon?
- output: ' \n\nThe moon is a natural satellite of Earth, orbiting around Earth, and it is the only celestial body that has been observed to be in orbit around Earth'
- prompt: Answer with one word only: the capital of France is
- output: '...? \n\nThe answer is "Paris". \n\nBut why is that the case? What is the reasoning behind this answer? \n\nThe answer is "Paris".'
- prompt: Translate to English: 今天天气很好，我们一起去散步。
- output: ' 今天天气很好，我们一起去散步。 今天天气很好，我们一起去散步。 今天天气很好，我们一起去散步。 今天天气很好，'

#### Throughput
| mode | batch | total tok/s | output tok/s | mean sec/iter | hit_ops |
| --- | --- | --- | --- | --- | --- |
| vllm_native | 128 | 22298.55 | 11508.93 | 0.7118 | native |
| nt_all_on | 128 | 1113.04 | 574.47 | 14.2600 | RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, PagedAttentionPrefill, PagedAttentionDecode, RoPE, TopKTopP, RandomSample |
| nt_disable_fa | 128 | 1115.82 | 575.91 | 14.2245 | RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, RoPE, TopKTopP, RandomSample |
| nt_disable_fa_mm | 128 | 1132.78 | 584.66 | 14.0115 | RMSNorm, SiluAndMul, Embedding, RoPE, TopKTopP, RandomSample |

### `gpt2`
- model id: `openai-community/gpt2`
- cache path: `/home/tiger/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e`

#### Accuracy
- mode: `nt_all_off`
- hit_ops: `None`
- prompt: Answer in one short sentence: what is the moon?
- output: '\n\nThe moon is a celestial object that orbits the sun. The moon is a celestial object that orbits the sun.\n\nThe moon is a celestial object'
- prompt: Answer with one word only: the capital of France is
- output: ' not the capital of France.\n\nThe capital of France is not the capital of France.\n\nThe capital of France is not the capital of France.'
- prompt: Translate to English: 今天天气很好，我们一起去散步。
- output: '\n\nTranslation: 今天天气很好，我们一起去散步'

- mode: `nt_all_on`
- hit_ops: `GELU, LayerNorm, MatMul, Embedding, WPE, NTWPEKernel, LMHead, PagedAttentionPrefill, PagedAttentionDecode, TopKTopP, RandomSample`
- prompt: Answer in one short sentence: what is the moon?
- output: '\n\nThe moon is a celestial object that orbits the sun. The moon is a celestial object that orbits the sun.\n\nThe moon is a celestial object'
- prompt: Answer with one word only: the capital of France is
- output: ' not the capital of France, but the capital of France.\n\nThe capital of France is not the capital of France, but the capital of France.\n'
- prompt: Translate to English: 今天天气很好，我们一起去散步。
- output: '\n\nTranslation: 今天天气很好，我们一起去散步'

#### Throughput
| mode | batch | total tok/s | output tok/s | mean sec/iter | hit_ops |
| --- | --- | --- | --- | --- | --- |
| vllm_native | 256 | 46731.59 | 24119.53 | 0.6793 | native |
| nt_all_on | 256 | 36441.27 | 18808.40 | 0.8711 | GELU, LayerNorm, MatMul, Embedding, WPE, NTWPEKernel, LMHead, PagedAttentionPrefill, PagedAttentionDecode, TopKTopP, RandomSample |
| nt_disable_fa | 256 | 36180.41 | 18673.76 | 0.8774 | GELU, LayerNorm, MatMul, Embedding, WPE, NTWPEKernel, LMHead, TopKTopP, RandomSample |
| nt_disable_fa_mm | 256 | 46129.27 | 23808.66 | 0.6882 | GELU, LayerNorm, Embedding, WPE, NTWPEKernel, TopKTopP, RandomSample |
