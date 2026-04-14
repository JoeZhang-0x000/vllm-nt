# MLU 上 NT 算子开关、调优与测试记录

本文整理本轮在 MLU 机器上对 `vllm-nt` 做的开关支持、调优入口和测试结果。测试重点是先验证 NT 算子接管范围是否可控，再定位吞吐下降的主要来源，最后确认 `max_num_configs` 调优模式对 MatMul 的实际影响。

## 测试环境

| 项目 | 配置 |
| --- | --- |
| 设备 | MLU，测试时使用 `MLU_VISIBLE_DEVICES=1` |
| dtype | `bfloat16` |
| vLLM | `0.11.2` |
| vLLM 进程模式 | `VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| 模型 | `Qwen/Qwen3-0.6B`、`openai-community/gpt2` |
| 缺失模型 | `llama2_7b` 未在 HuggingFace / ModelScope 本地缓存中找到，未运行 |

完整原始报告见：

- `reports/mlu/nt_mlu_validation_report.md`
- `reports/mlu/mm_hypothesis_benchmark_current.md`
- `reports/mlu/mm_hypothesis_benchmark_fixed_tuning.md`
- `reports/mlu/nt_matmul_microbenchmark.md`
- `reports/mlu/official_ninetoothed_examples_mm_mlu.md`
- `reports/mlu/qwen3_non_mm_ablation.md`
- `reports/mlu/rms_norm_variant_benchmark.md`
- `reports/mlu/multi_model_eager_throughput_summary.md`

## 新增开关

本轮新增了 NT 算子接管范围的环境变量开关：

| 环境变量 | 语义 | 默认值 |
| --- | --- | --- |
| `VLLM_NT_ENABLE_ALL` | 总开关，关闭后所有 NT 算子都不接管 | `1` |
| `VLLM_NT_ENABLE_FA` | 控制 FA / PagedAttention / UnifiedAttention 相关路径 | `1` |
| `VLLM_NT_ENABLE_MM` | 控制 MatMul 和 LMHead 相关路径 | `1` |

典型使用方式：

```bash
VLLM_NT_ENABLE_ALL=0 python ...
VLLM_NT_ENABLE_ALL=1 VLLM_NT_ENABLE_FA=0 python ...
VLLM_NT_ENABLE_ALL=1 VLLM_NT_ENABLE_FA=0 VLLM_NT_ENABLE_MM=0 python ...
```

## `max_num_configs` 调优入口

当前支持两套全局 `max_num_configs` 模式，作用于所有通过 `_cached_make` 创建的 NT kernel：

| 模式 | 设置 | 含义 |
| --- | --- | --- |
| 快速模式 | `VLLM_NT_MAX_NUM_CONFIGS_MODE=quick` | `max_num_configs=2`，默认模式 |
| 调优模式 | `VLLM_NT_MAX_NUM_CONFIGS_MODE=tuning` | `max_num_configs=10` |
| 显式覆盖 | `VLLM_NT_MAX_NUM_CONFIGS=<int>` | 优先级高于 mode |

MatMul 专项测试脚本也支持命令行参数：

```bash
MLU_VISIBLE_DEVICES=1 python scripts/run_nt_matmul_microbenchmark.py \
  --dtype bfloat16 \
  --warmup 5 \
  --iters 30 \
  --max-num-configs 10 \
  --output reports/mlu/nt_matmul_microbenchmark.md
```

## 准确性测试

准确性测试使用 3 条 prompt，并且放在一个 batch 中推理，不拆成三次请求：

1. `Answer in one short sentence: what is the moon?`
2. `Answer with one word only: the capital of France is`
3. `Translate to English: 今天天气很好，我们一起去散步。`

结果摘要：

| 模型 | `NT 全关` vs `NT 全开` |
| --- | --- |
| `qwen3_0.6b` | 3 条输出一致 |
| `gpt2` | 3 条输出存在差异 |

需要注意，`gpt2` 本身输出质量较弱，但这里的结论不是评估文本质量，而是记录同一 batch prompt 在 `NT 全关` 与 `NT 全开` 下生成结果不一致。

## 离线吞吐测试

吞吐测试使用离线模式，并对比以下 4 组：

| 模式 | 含义 |
| --- | --- |
| `vllm_native` | 不加载 `vllm_nt`，只保留 MLU platform plugin |
| `nt_all_on` | NT 全开 |
| `nt_disable_fa` | NT 开启但关闭 FA |
| `nt_disable_fa_mm` | NT 开启但关闭 FA 和 MM |

初始吞吐摘要（该报告生成于 `max_num_configs` 支持前，实际等价于旧默认 `max_num_configs=1`）：

| 模型 | native total tok/s | NT 全开 total tok/s | 关闭 FA total tok/s | 关闭 FA+MM total tok/s | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `qwen3_0.6b` | `23879.08` | `463.12` | `463.15` | `509.86` | 关闭 FA 与 MM 后仍明显慢，说明还有其它 NT 算子也有较大开销 |
| `gpt2` | `50752.77` | `9761.92` | `9746.28` | `51504.21` | 关闭 FA+MM 后基本回到 native，说明主要瓶颈集中在 MM 路径 |

全局 `max_num_configs` 支持后，使用 `max_model_len=512` 和 `max_num_batched_tokens=512` 避免 Qwen profile 阶段触发 MLU grid 上限，并重跑 quick/tuning 两组吞吐：

| 模型 | max configs | native total tok/s | NT 全开 total tok/s | 关闭 FA total tok/s | 关闭 FA+MM total tok/s | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen3_0.6b` | `2` | `22280.29` | `865.49` | `864.92` | `1055.57` | quick 相比旧默认有提升，但仍远慢于 native |
| `qwen3_0.6b` | `10` | `22251.12` | `913.44` | `913.32` | `1134.32` | tuning 比 quick 继续小幅提升，但仍有数量级差距 |
| `gpt2` | `2` | `46681.50` | `9329.03` | `9337.66` | `45012.21` | 关闭 FA+MM 后接近 native，MM 仍是主要瓶颈 |
| `gpt2` | `10` | `46469.11` | `9333.07` | `9329.79` | `46151.35` | tuning 对 NT 全开/关闭 FA 基本无收益，对关闭 FA+MM 有小幅提升 |

如果不限制 `max_model_len/max_num_batched_tokens`，Qwen 在全局 `max_num_configs=2` 下会在 RMSNorm profile 阶段触发 `NineToothedGridError`：`Required: 163840, Hardware limit: 65535`。将两者降到 `1024` 时仍会遇到边界值 `Required: 65536, Hardware limit: 65535`，因此本轮完整吞吐补测采用 `512/512` 作为可运行口径。

## MatMul 专项基准

MatMul 专项基准只测 kernel 级路径，不加载完整 vLLM 引擎。对比路径包括：

| 路径 | 说明 |
| --- | --- |
| `vllm_mlu._mlu_ops.matmul` | MLU 原生 matmul baseline |
| `vllm_nt._ntops.torch.linear` | NT linear 包装路径 |
| `vllm_nt._ntops.torch.matmul` | NT matmul 包装路径 |

### `max_num_configs=1` 基线

| 模型 | shape | M | native ms | NT linear ms | NT linear gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt2` | `768x768` | `8192` | `0.0816` | `4.0607` | `49.74x` |
| `gpt2` | `768x3072` | `8192` | `0.2478` | `16.1515` | `65.18x` |
| `gpt2` | `3072x768` | `8192` | `0.2685` | `13.8215` | `51.48x` |
| `gpt2` | `768x50257` | `256` | `0.1880` | `8.2830` | `44.06x` |
| `qwen3_0.6b` | `1024x1024` | `8192` | `0.1332` | `6.9653` | `52.28x` |
| `qwen3_0.6b` | `1024x3072` | `8192` | `0.3390` | `20.7932` | `61.34x` |
| `qwen3_0.6b` | `1536x1024` | `8192` | `0.1804` | `9.7159` | `53.85x` |
| `qwen3_0.6b` | `1024x151936` | `256` | `0.5446` | `32.0151` | `58.78x` |

### `max_num_configs=10` 调优模式

| 模型 | shape | M | native ms | NT linear ms | NT linear gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt2` | `768x768` | `8192` | `0.0836` | `4.0548` | `48.51x` |
| `gpt2` | `768x3072` | `8192` | `0.2475` | `16.1455` | `65.23x` |
| `gpt2` | `3072x768` | `8192` | `0.2585` | `13.8148` | `53.43x` |
| `gpt2` | `768x50257` | `256` | `0.1870` | `8.2799` | `44.27x` |
| `qwen3_0.6b` | `1024x1024` | `8192` | `0.1327` | `6.9652` | `52.50x` |
| `qwen3_0.6b` | `1024x3072` | `8192` | `0.3385` | `20.8093` | `61.48x` |
| `qwen3_0.6b` | `1536x1024` | `8192` | `0.1793` | `9.7006` | `54.12x` |
| `qwen3_0.6b` | `1024x151936` | `256` | `0.5441` | `32.0259` | `58.86x` |

对比结论：`max_num_configs=10` 基本没有改善这批 MatMul shape。核心大 shape 的提升约为 `-0.1%` 到 `+0.2%`，可视为噪声范围。

## 当前结论

1. MLU 上现在已经可以通过环境变量控制 NT 是否接管全部算子、FA、MM。
2. `gpt2` 的整体吞吐瓶颈主要集中在 MM 路径，关闭 `FA+MM` 后吞吐恢复到 native 同量级。
3. `qwen3_0.6b` 关闭 `FA+MM` 后仍明显慢，说明除 MM 外还需要继续看 `RMSNorm`、`SiluAndMul`、`Embedding`、`RoPE`、`TopKTopP`、`RandomSample` 等路径。
4. MatMul 的 NT kernel 与 MLU 原生 `tmo.matmul` 相比仍有数量级差距，大 shape 下通常慢 `48x-66x`。
5. 将 `max_num_configs` 从 `1` 提升到 `10` 没有明显改善 MatMul microbenchmark 性能，说明当前 MatMul kernel 瓶颈不只是调优搜索数量不足。
6. 在完整 vLLM 吞吐里，全局 `max_num_configs=10` 对 Qwen 有小幅收益，但需要降低 profile token 上限才能避开 MLU grid limit。

## 后续建议

1. 短期默认不要在 MLU 上接管 MM，建议默认保留 `VLLM_NT_ENABLE_MM=0` 或仅在实验时开启。
2. 针对 MatMul，下一步应优先看 NT kernel 的 tile 策略、内存访问、`num_warps` 实际生效值，以及是否能直接复用或包装 MLU 原生 `tmo.matmul`。
3. 针对 `qwen3_0.6b`，继续按算子拆分测试非 MM 路径，优先排查 `RMSNorm`、`SiluAndMul` 和采样相关算子。
4. 如果后续补齐 `llama2_7b` 本地缓存，需要重新跑准确性、吞吐和 MatMul shape，避免报告只覆盖小模型。
