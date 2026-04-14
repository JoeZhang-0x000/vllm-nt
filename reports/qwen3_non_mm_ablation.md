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
| native | vllm_native | - | 22325.07 | +21193.44 | 100.00% | native |
| nt_all_on | nt_all_on | - | 1113.49 | -18.14 | 4.99% | RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, PagedAttentionPrefill, PagedAttentionDecode, RoPE, TopKTopP, RandomSample |
| nt_disable_fa | nt_disable_fa | - | 1113.27 | -18.36 | 4.99% | RMSNorm, SiluAndMul, MatMul, Embedding, LMHead, RoPE, TopKTopP, RandomSample |
| nt_disable_fa_mm | nt_disable_fa_mm | - | 1131.63 | +0.00 | 5.07% | RMSNorm, SiluAndMul, Embedding, RoPE, TopKTopP, RandomSample |
| ablate_rmsnorm_keep_mm | nt_disable_fa | RMSNorm | 17369.06 | +16237.43 | 77.80% | SiluAndMul, MatMul, Embedding, LMHead, RoPE, TopKTopP, RandomSample |
| ablate_rmsnorm | nt_disable_fa_mm | RMSNorm | 22290.37 | +21158.74 | 99.84% | SiluAndMul, Embedding, RoPE, TopKTopP, RandomSample |
| ablate_silu_and_mul | nt_disable_fa_mm | SiluAndMul | 1133.72 | +2.09 | 5.08% | RMSNorm, Embedding, RoPE, TopKTopP, RandomSample |
| ablate_embedding | nt_disable_fa_mm | Embedding | 1126.26 | -5.37 | 5.04% | RMSNorm, SiluAndMul, RoPE, TopKTopP, RandomSample |
| ablate_rope | nt_disable_fa_mm | RoPE | 1123.98 | -7.65 | 5.03% | RMSNorm, SiluAndMul, Embedding, TopKTopP, RandomSample |
| ablate_topk_topp | nt_disable_fa_mm | TopKTopP | 1130.93 | -0.70 | 5.07% | RMSNorm, SiluAndMul, Embedding, RoPE, RandomSample |
| ablate_random_sample | nt_disable_fa_mm | RandomSample | 1132.30 | +0.67 | 5.07% | RMSNorm, SiluAndMul, Embedding, RoPE, TopKTopP |
| ablate_sampling | nt_disable_fa_mm | TopKTopP, RandomSample | 1132.45 | +0.82 | 5.07% | RMSNorm, SiluAndMul, Embedding, RoPE |

## Reading
- `native`: `22325.07 tok/s`
- `nt_disable_fa_mm`: `1131.63 tok/s`，这代表在 FA 和 MM 都关闭后，剩余 NT 非 MM 路径的基线吞吐。
- `ablate_rmsnorm_keep_mm`: `17369.06 tok/s`，说明在去掉 RMSNorm 后，MM 仍然会带来约 `4921 tok/s` 的损失，但已经不是主导瓶颈。
- 最大单项收益是 `ablate_rmsnorm`，吞吐 `22290.37 tok/s`，相对 `nt_disable_fa_mm` 提升 `+21158.74 tok/s`。
