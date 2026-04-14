# MLU Benchmark Summary

本文收束当前在 MLU 上完成的 `vllm-nt` 调优、验证和分析结论。详细原始数据仍保留在同目录下的专项报告中。

## 我们做了什么

1. 增加 NT 算子接管开关，支持按类别或按算子名禁用：
   - `VLLM_NT_ENABLE_ALL`
   - `VLLM_NT_ENABLE_FA`
   - `VLLM_NT_ENABLE_MM`
   - `VLLM_NT_DISABLE_OPS=RMSNorm,...`
2. 增加全局 `max_num_configs` 调优入口：
   - quick: `max_num_configs=2`
   - tuning: `max_num_configs=10`
   - 显式覆盖：`VLLM_NT_MAX_NUM_CONFIGS=<int>`
3. 修复 MatMul/Linear 在 `vllm-nt` 中默认固定 `BLOCK_SIZE_M/N/K=64` 的问题，使其走官方 NineToothed 风格的 `ninetoothed.block_size()` autotune 路径。
4. 为 RMSNorm 增加 last-dim fused fast path，使 `vllm-nt` 当前实现对齐官方 `fused_rms_norm` 性能。
5. 新增多组 MLU benchmark 脚本和报告：
   - 官方 NineToothed MM 对比
   - `vllm-nt` MM 修复前后对比
   - RMSNorm variant 对比
   - Qwen3 非 MM ablation
   - 7 模型 eager 端到端吞吐矩阵

## 测试环境

| 项目 | 配置 |
| --- | --- |
| device | MLU，主要使用 `MLU_VISIBLE_DEVICES=1` |
| dtype | `bfloat16` |
| vLLM 进程模式 | `VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| max configs | `VLLM_NT_MAX_NUM_CONFIGS=10` |
| 端到端输入输出 | `input_len=64`，`output_len=64` |
| profile 限制 | `max_model_len=512`，`max_num_batched_tokens=512` |
| 多模型矩阵 | `enforce_eager=1` |

## 分算子结果

### MatMul

官方 `ninetoothed-examples` 的 square MM 在 MLU 上并不慢：

| size | NineToothed ms | torch.mm ms | NT vs torch |
| ---: | ---: | ---: | ---: |
| 1024 | 0.0312 | 0.0206 | 1.51x |
| 2048 | 0.1795 | 0.1219 | 1.47x |
| 4096 | 1.1749 | 0.9264 | 1.27x |

`vllm-nt` 原来的 MM 慢几十倍，根因是默认把 block size 固定成 `64/64/64`，没有真正走官方 autotune 路径。修复后：

| 对比 | 修复前 | 修复后 |
| --- | ---: | ---: |
| current MatMul vs official NT, square | 34.48x | 1.52x |
| current MatMul vs official NT, model shapes | 28.39x | 1.39x |

结论：MM 的主要集成问题已经修复。修复后 MM 仍比 `torch.mm` 慢约 `2x-3x`，但已经不是几十倍级别，也不再是多数模型当前的最大瓶颈。

### RMSNorm

RMSNorm 是当前 MLU 上最一致、最主要的剩余瓶颈。

`vllm-nt` 的 fused fast path 已经贴近官方 `fused_rms_norm`，但官方 fused 本身仍明显慢于 torch native：

| case | shape | torch ms | current vllm-nt ms | official fused NT ms | current vs torch |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen_decode_like | 512x1024 | 0.0393 | 0.4505 | 0.4449 | 11.46x |
| qwen_large_m | 8192x1024 | 0.3072 | 7.2112 | 7.1871 | 23.48x |
| gpt2_like | 512x768 | 0.0348 | 0.4511 | 0.4519 | 12.96x |

结论：当前 `vllm-nt` RMSNorm 包装层已基本追平官方 fused 实现，剩余差距主要是 NineToothed RMSNorm kernel 在 MLU 上相对 torch native 仍慢很多。

### Qwen3 非 MM ablation

在 Qwen3 上，关闭 FA 和 MM 后仍很慢；进一步逐个关闭非 MM 算子后，RMSNorm 是决定性因素：

| case | total tok/s | % native | 说明 |
| --- | ---: | ---: | --- |
| native | 22188.87 | 100.00% | vLLM native |
| NT_All | 1647.39 | 7.42% | NT 全开 |
| NO_FA | 1648.48 | 7.43% | 关闭 FA，几乎无变化 |
| NO_FA_MM | 1672.01 | 7.54% | 关闭 FA+MM，仍很慢 |
| NO_RMS, keep MM | 17406.20 | 78.45% | 只关 RMSNorm，MM 仍开，吞吐大幅恢复 |
| NO_RMS_MM_FA | 22223.42 | 100.16% | 关闭 RMSNorm+MM+FA，基本回到 native |

结论：Qwen3 当前不是 FA 主导，也不是 MM 主导；在排除 FA 后，主要瓶颈是 RMSNorm。MM 是次要影响，只有在移除 RMSNorm 后才明显。

## 端到端结果

### Qwen3 / gpt2 非 eager 验证

MM 修复后，gpt2 的 NT 吞吐已明显恢复；Qwen3 仍受 RMSNorm 限制：

| model | native tok/s | NT_All tok/s | NO_FA tok/s | NO_FA_MM tok/s |
| --- | ---: | ---: | ---: | ---: |
| qwen3_0.6b | 22298.55 | 1113.04 | 1115.82 | 1132.78 |
| gpt2 | 46731.59 | 36441.27 | 36180.41 | 46129.27 |

### 7 模型 eager 吞吐矩阵

按 `/data02/jiangqiu/models/down_models.sh` 中存在的模型，开启 eager 后测试 6 种模式：`native`、`NT_All`、`NO_RMS`、`NO_MM`、`NO_FA`、`NO_RMS_MM_FA`。

| model | native tok/s | NT_All tok/s | best NT mode | best tok/s | best / native |
| --- | ---: | ---: | --- | ---: | ---: |
| Qwen2.5-7B-Instruct | 7538.25 | 4309.50 | NO_RMS_MM_FA | 5563.97 | 73.8% |
| gpt2 | 45935.69 | 35004.33 | NO_RMS_MM_FA | 35623.78 | 77.6% |
| DeepSeek-R1-Distill-Qwen-7B | 6552.77 | 3767.45 | NO_RMS | 4689.14 | 71.6% |
| Llama-2-7b-chat-hf | 5236.80 | 3085.83 | NO_RMS | 3780.82 | 72.2% |
| MiniCPM4.1-8B | 4179.73 | 739.00 | NO_RMS_MM_FA | 795.62 | 19.0% |
| glm-4-9b | 4312.44 | 2709.42 | NO_RMS | 3381.08 | 78.4% |
| Mistral-7B-Instruct-v0.3 | 6779.80 | 3970.97 | NO_RMS_MM_FA | 5198.06 | 76.7% |

结论：7 个模型里，最佳 NT 模式始终是 `NO_RMS` 或 `NO_RMS_MM_FA`。`NO_MM` 和 `NO_FA` 通常接近 `NT_All`，说明在当前 eager 测试口径下，它们不是主要剩余瓶颈。

## 总体分析

1. MM 问题已经定位并修复：之前慢几十倍主要是 `vllm-nt` 固定 block size 导致没有真正使用官方 autotune 路径。
2. 官方 MM 在 MLU 上相对 torch 只慢约 `1.06x` 几何平均，说明 NineToothed 在 MLU 上不是普遍不可用。
3. RMSNorm 是当前最稳定的剩余瓶颈：
   - Qwen3 ablation 显示单独关闭 RMSNorm 可将吞吐从约 `1.6k tok/s` 恢复到 `17k-22k tok/s` 区间。
   - 7 模型 eager 矩阵中最佳模式始终包含 `NO_RMS`。
4. 当前 `vllm-nt` RMSNorm fast path 已追平官方 fused RMSNorm，但官方 fused 本身仍比 torch native 慢 `11x-23x`，后续需要优化 NineToothed RMSNorm kernel 本身。
5. MiniCPM4.1-8B 是明显异常点：最佳 NT 模式仅 `19.0%` native，说明除 RMSNorm 外可能还有模型结构或其它算子的额外瓶颈，需要单独拆解。

## 下一步建议

1. 优先优化 RMSNorm kernel，而不是继续看 FA/MM。
2. 针对 RMSNorm 做更细的 kernel-level sweep：`M`、`N`、`num_warps`、block size、是否使用官方 plain vs fused 结构。
3. 对 MiniCPM4.1-8B 单独做 per-op ablation，因为它不符合其它 6 个模型的恢复比例。
4. 后续新增其它设备时，把报告放到 `reports/<device>/`，并在 `reports/README.md` 中登记。

## Source Reports

- `reports/mlu/multi_model_eager_throughput_summary.md`
- `reports/mlu/qwen3_non_mm_ablation.md`
- `reports/mlu/rms_norm_variant_benchmark.md`
- `reports/mlu/mm_hypothesis_benchmark_current.md`
- `reports/mlu/mm_hypothesis_benchmark_fixed_tuning.md`
- `reports/mlu/official_ninetoothed_examples_mm_mlu.md`
- `reports/mlu/nt_mlu_validation_report.md`
