# Model Correctness and Hit Tracking on MLU

## Configuration
- prompt: `In one short sentence, explain what the moon is.`
- max_tokens: `32`
- dtype: `bfloat16`
- enforce_eager: `1`
- max_num_configs: `10`
- max_model_len/max_num_batched_tokens: `512` / `512`
- mlu_visible_devices: `1`

## `Llama-2-7b-chat-hf`
- model path: `/data02/jiangqiu/models/Llama-2-7b-chat-hf`
- prompt: `In one short sentence, explain what the moon is.`
- native output: `
The moon is the Earth's only natural satellite, orbiting our planet at an average distance of about 239,000 miles (`
- NT_ALL_ON output: `
The moon is the Earth's only natural satellite, orbiting our planet at an average distance of about 239,000 miles (`
- NT hit_ops: `RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, PagedAttentionPrefill, PagedAttentionDecode, RoPE, TopKTopP, RandomSample`

| operator | hits |
| --- | ---: |
| RMSNorm | 2275 |
| SiluAndMul | 1120 |
| MatMul | 2240 |
| Embedding | 35 |
| LMHead | 35 |
| PagedAttentionPrefill | 32 |
| PagedAttentionDecode | 992 |
| RoPE | 1120 |
| TopKTopP | 3 |
| RandomSample | 3 |

## `Qwen2.5-7B-Instruct`
- model path: `/data02/jiangqiu/models/Qwen2.5-7B-Instruct`
- prompt: `In one short sentence, explain what the moon is.`
- native output: ` The moon is a natural satellite that orbits the Earth and is composed of rock and dust.`
- NT_ALL_ON output: ` The moon is a natural satellite that orbits the Earth and is composed of rock and dust.`
- NT hit_ops: `RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, PagedAttentionPrefill, PagedAttentionDecode, RoPE, TopKTopP, RandomSample`

| operator | hits |
| --- | ---: |
| RMSNorm | 1254 |
| SiluAndMul | 616 |
| MatMul | 1232 |
| Embedding | 22 |
| LMHead | 22 |
| PagedAttentionPrefill | 28 |
| PagedAttentionDecode | 504 |
| RoPE | 616 |
| TopKTopP | 3 |
| RandomSample | 3 |

## `MiniCPM4.1-8B`
- model path: `/data02/jiangqiu/models/MiniCPM4.1-8B`
- prompt: `In one short sentence, explain what the moon is.`
- native output: ` The moon is Earth's natural satellite.
# In one short sentence, explain what the moon is. The moon is Earth's natural satellite.
`
- NT_ALL_ON output: ` The moon is Earth's natural satellite.
# In one short sentence, explain what the moon is. The moon is Earth's natural satellite.
`
- NT hit_ops: `RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, PagedAttentionPrefill, PagedAttentionDecode, TopKTopP, RandomSample`

| operator | hits |
| --- | ---: |
| RMSNorm | 2275 |
| SiluAndMul | 1120 |
| MatMul | 4480 |
| Embedding | 35 |
| LMHead | 35 |
| PagedAttentionPrefill | 32 |
| PagedAttentionDecode | 992 |
| TopKTopP | 3 |
| RandomSample | 3 |

## `glm-4-9b`
- model path: `/data02/jiangqiu/models/glm-4-9b`
- prompt: `In one short sentence, explain what the moon is.`
- native output: `Solution:

Step 1: The moon is a natural satellite that orbits around the Earth. It is the fifth largest moon in the solar system and is the only`
- NT_ALL_ON output: `Solution:

Step 1: The moon is a natural satellite that orbits around the Earth. It is the fifth largest moon in the solar system and is the only`
- NT hit_ops: `RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, PagedAttentionPrefill, PagedAttentionDecode, TopKTopP, RandomSample`

| operator | hits |
| --- | ---: |
| RMSNorm | 2835 |
| SiluAndMul | 1400 |
| MatMul | 2800 |
| Embedding | 35 |
| LMHead | 35 |
| PagedAttentionPrefill | 40 |
| PagedAttentionDecode | 1240 |
| TopKTopP | 3 |
| RandomSample | 3 |

## `Mistral-7B-Instruct-v0.3`
- model path: `/data02/jiangqiu/models/Mistral-7B-Instruct-v0.3`
- prompt: `In one short sentence, explain what the moon is.`
- native output: `

The moon is a natural satellite that orbits Earth.

What is the moon made of?

The moon is primarily made of silicate`
- NT_ALL_ON output: `

The moon is a natural satellite that orbits Earth.

What is the moon made of?

The moon is primarily made of silicate`
- NT hit_ops: `RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, PagedAttentionPrefill, PagedAttentionDecode, RoPE, TopKTopP, RandomSample`

| operator | hits |
| --- | ---: |
| RMSNorm | 2275 |
| SiluAndMul | 1120 |
| MatMul | 2240 |
| Embedding | 35 |
| LMHead | 35 |
| PagedAttentionPrefill | 32 |
| PagedAttentionDecode | 992 |
| RoPE | 1120 |
| TopKTopP | 3 |
| RandomSample | 3 |

## `gpt2`
- model path: `/data02/jiangqiu/models/gpt2`
- prompt: `In one short sentence, explain what the moon is.`
- native output: `

"The moon is a very small object, about the size of a small car," said Dr. John S. Siegel, a professor of astronomy`
- NT_ALL_ON output: `

"The moon is a very small object, about the size of a small car, and it is a very small object. It is a very small`
- NT hit_ops: `GELU, LayerNorm, MatMul, Embedding, WPE, NTWPEKernel, LMHead, PagedAttentionPrefill, PagedAttentionDecode, TopKTopP, RandomSample`

| operator | hits |
| --- | ---: |
| GELU | 420 |
| LayerNorm | 875 |
| MatMul | 1680 |
| Embedding | 35 |
| WPE | 35 |
| NTWPEKernel | 35 |
| LMHead | 35 |
| PagedAttentionPrefill | 12 |
| PagedAttentionDecode | 372 |
| TopKTopP | 3 |
| RandomSample | 3 |

## `DeepSeek-R1-Distill-Qwen-7B`
- model path: `/data02/jiangqiu/models/DeepSeek-R1-Distill-Qwen-7B`
- prompt: `In one short sentence, explain what the moon is.`
- native output: ` The moon is the natural satellite of Earth.

The moon is the only natural satellite of Earth.

The moon is the largest natural satellite of Earth.

The moon is`
- NT_ALL_ON output: ` The moon is the natural satellite of Earth.

The moon is the only natural satellite of Earth.

The moon is the largest natural satellite of Earth.

The moon is`
- NT hit_ops: `RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, PagedAttentionPrefill, PagedAttentionDecode, RoPE, TopKTopP, RandomSample`

| operator | hits |
| --- | ---: |
| RMSNorm | 1995 |
| SiluAndMul | 980 |
| MatMul | 1960 |
| Embedding | 35 |
| LMHead | 35 |
| PagedAttentionPrefill | 28 |
| PagedAttentionDecode | 868 |
| RoPE | 980 |
| TopKTopP | 3 |
| RandomSample | 3 |

