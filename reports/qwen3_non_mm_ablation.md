# Qwen3 Non-MM Ablation on MLU

## Configuration
- model: `Qwen/Qwen3-0.6B`
- dtype: `bfloat16`
- batch_size: `128`
- input_len/output_len: `64` / `64`
- max_num_configs: `10`
- max_model_len/max_num_batched_tokens: `512` / `512`
- mlu_visible_devices: `1`

## Throughput
| case | mode | disabled_ops | total tok/s | delta vs nt_disable_fa_mm | % of native | hit_ops |
| --- | --- | --- | ---: | ---: | ---: | --- |
| native | vllm_native | - | 22188.87 | +20516.86 | 100.00% | native |
| nt_all_on | nt_all_on | - | 1647.39 | -24.62 | 7.42% | RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, PagedAttentionPrefill, PagedAttentionDecode, RoPE, TopKTopP, RandomSample |
| nt_disable_fa | nt_disable_fa | - | 1648.48 | -23.53 | 7.43% | RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, RoPE, TopKTopP, RandomSample |
| nt_disable_fa_mm | nt_disable_fa_mm | - | 1672.01 | +0.00 | 7.54% | RMSNorm, SiluAndMul, Embedding, RoPE, TopKTopP, RandomSample |
| ablate_rmsnorm_keep_mm | nt_disable_fa | RMSNorm | 17406.20 | +15734.19 | 78.45% | SiluAndMul, MatMul, Embedding, LMHead, RoPE, TopKTopP, RandomSample |
| ablate_rmsnorm | nt_disable_fa_mm | RMSNorm | 22223.42 | +20551.41 | 100.16% | SiluAndMul, Embedding, RoPE, TopKTopP, RandomSample |
| ablate_silu_and_mul | nt_disable_fa_mm | SiluAndMul | 1672.03 | +0.02 | 7.54% | RMSNorm, Embedding, RoPE, TopKTopP, RandomSample |
| ablate_embedding | nt_disable_fa_mm | Embedding | 1691.92 | +19.91 | 7.63% | RMSNorm, SiluAndMul, RoPE, TopKTopP, RandomSample |
| ablate_rope | nt_disable_fa_mm | RoPE | 1635.94 | -36.07 | 7.37% | RMSNorm, SiluAndMul, Embedding, TopKTopP, RandomSample |
| ablate_topk_topp | nt_disable_fa_mm | TopKTopP | 1673.49 | +1.48 | 7.54% | RMSNorm, SiluAndMul, Embedding, RoPE, RandomSample |
| ablate_random_sample | nt_disable_fa_mm | RandomSample | 1672.64 | +0.63 | 7.54% | RMSNorm, SiluAndMul, Embedding, RoPE, TopKTopP |
| ablate_sampling | nt_disable_fa_mm | TopKTopP, RandomSample | 1673.94 | +1.93 | 7.54% | RMSNorm, SiluAndMul, Embedding, RoPE |

## Reading
- `native`: `22188.87 tok/s`
- `nt_disable_fa_mm`: `1672.01 tok/s`，这代表在 FA 和 MM 都关闭后，剩余 NT 非 MM 路径的基线吞吐。
- 最大单项收益是 `ablate_rmsnorm`，吞吐 `22223.42 tok/s`，相对 `nt_disable_fa_mm` 提升 `+20551.41 tok/s`。
