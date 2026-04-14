# MLU Result Roll-Up

本文收束本轮在 MLU 上完成的最终测试结果，只保留三类最终报告：

- `correctness_by_model.md`
- `throughput_matrix.md`
- `operator_benchmarks.md`

## 我们在 MLU 上做了什么

1. 为 `vllm-nt` 增加 NT 接管控制：
   - `VLLM_NT_ENABLE_ALL`
   - `VLLM_NT_ENABLE_FA`
   - `VLLM_NT_ENABLE_MM`
   - `VLLM_NT_DISABLE_OPS=...`
2. 修复了 `MatMul/Linear` 默认固定 `BLOCK_SIZE_M/N/K=64` 导致 MM 没走到官方 autotune 路径的问题。
3. 为常见 last-dim RMSNorm 增加 fused fast path，使 `vllm-nt` 当前实现对齐官方 `fused_rms_norm` 的表现。
4. 用最终版脚本重跑三类实验：
   - 推理正确性和 hit 追踪
   - 离线 eager 吞吐矩阵
   - 官方 NT 算子库 vs torch native 的算子级 benchmark

## 测试环境

| 项目 | 配置 |
| --- | --- |
| device | `MLU_VISIBLE_DEVICES=1` |
| dtype | `bfloat16` |
| vLLM multiprocessing | `VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| max configs | `VLLM_NT_MAX_NUM_CONFIGS=10` |
| profile limits | `max_model_len=512`，`max_num_batched_tokens=512` |
| 吞吐模式 | `enforce_eager=1` |
| 模型来源 | `/data02/jiangqiu/models/down_models.sh` 中存在的 7 个模型 |

## 正确性检验结果

所有模型都使用同一条 prompt：

`In one short sentence, explain what the moon is.`

主要结论：

- `Llama-2-7b-chat-hf`：`native` 与 `NT_ALL_ON` 输出一致，命中 `RMSNorm/SiluAndMul/MatMul/Embedding/LMHead/PagedAttention/RoPE/TopKTopP/RandomSample`
- `Qwen2.5-7B-Instruct`：`native` 与 `NT_ALL_ON` 输出一致，命中集合与 Llama 基本一致
- `MiniCPM4.1-8B`：`native` 与 `NT_ALL_ON` 输出一致，但没有命中 `RoPE`，命中 `RMSNorm/SiluAndMul/MatMul/Embedding/LMHead/PagedAttention/TopKTopP/RandomSample`
- `glm-4-9b`：`native` 与 `NT_ALL_ON` 输出一致，没有命中 `RoPE`
- `Mistral-7B-Instruct-v0.3`：`native` 与 `NT_ALL_ON` 输出一致，命中集合接近 Llama
- `gpt2`：`native` 与 `NT_ALL_ON` 输出不一致，命中的是 `GELU/LayerNorm/MatMul/Embedding/WPE/LMHead/PagedAttention/TopKTopP/RandomSample`
- `DeepSeek-R1-Distill-Qwen-7B`：`native` 与 `NT_ALL_ON` 输出一致，命中集合与 Qwen 类似

详细 prompt、输出和每个算子的 `hits` 见 `reports/mlu/correctness_by_model.md`。

## 吞吐测试结果

离线吞吐统一配置：

- `enforce_eager=1`
- `input_len=64`
- `output_len=64`
- `warmup_iters=1`
- `measure_iters=3`
- `max_model_len=512`
- `max_num_batched_tokens=512`
- `max_num_configs=10`

每个模型测试 6 种模式：

- `native`
- `NT_All`
- `NO_RMS`
- `NO_MM`
- `NO_FA`
- `NO_RMS_MM_FA`

关键结果：

| model | native tok/s | NT_All tok/s | best mode | best / native |
| --- | ---: | ---: | --- | ---: |
| `Llama-2-7b-chat-hf` | `5251.13` | `3108.28` | `NO_RMS_MM_FA` | `99.3%` |
| `Qwen2.5-7B-Instruct` | `7519.98` | `4009.07` | `NO_RMS_MM_FA` | `99.8%` |
| `MiniCPM4.1-8B` | `4166.13` | `749.86` | `NO_RMS_MM_FA` | `23.4%` |
| `glm-4-9b` | `4317.21` | `2732.99` | `NO_RMS_MM_FA` | `99.6%` |
| `Mistral-7B-Instruct-v0.3` | `6800.23` | `4020.62` | `NO_RMS_MM_FA` | `99.4%` |
| `gpt2` | `45921.82` | `35584.29` | `NO_RMS_MM_FA` | `97.2%` |
| `DeepSeek-R1-Distill-Qwen-7B` | `6587.16` | `3785.96` | `NO_RMS_MM_FA` | `99.5%` |

结论：

1. 对 7 个模型中的 6 个，`NO_RMS_MM_FA` 都几乎恢复到 native，说明当前剩余大头主要集中在 `RMSNorm`，其次是 `MM` 和 `FA` 的组合影响。
2. `NO_MM` 在多数模型上也能明显提升吞吐，说明 MM 修复后仍有残余开销，但已经不是之前几十倍那种级别。
3. `NO_FA` 单独关闭通常收益有限，说明在当前 MLU 环境下 FA 不是主导瓶颈。
4. `MiniCPM4.1-8B` 是明显异常点：即使 `NO_RMS_MM_FA` 也只有 `23.4%` native，需要单独拆解。

详细表格见 `reports/mlu/throughput_matrix.md`。

## 算子级结果

官方 `ninetoothed-examples` 与 `torch native` 的 MLU 对比如下：

### 相对接近 torch 的算子

- `MM`
  - `square_1024`: `1.48x`
  - `gpt2_mlp_up`: `2.31x`
  - `qwen_lm_head`: `2.53x`
- `AddMM`
  - `gpt2_fc_bias`: `1.60x`
  - `qwen_proj_bias`: `1.34x`

### 明显慢于 torch 的算子

- `RMSNorm`
  - `qwen_decode`: `4.91x`
  - `qwen_prefill`: `8.21x`
  - `llama_decode`: `2.45x`
- `FusedRMSNorm`
  - `qwen_decode`: `4.53x`
  - `qwen_prefill`: `9.78x`
  - `llama_decode`: `2.36x`
- `SiLU`
  - `gpt2_mlp`: `26.64x`
  - `qwen_mlp`: `32.27x`
- `SwiGLU`
  - `gpt2_like`: `20.95x`
  - `qwen_like`: `21.28x`

### 当前在 MLU/bfloat16 上失败的算子

- `Softmax`
  - `sampling_small_vocab`: `CompilationError`
  - `sampling_large_vocab`: `CompilationError`
  - `attn_scores`: `CompilationError`

详细表格见 `reports/mlu/operator_benchmarks.md`。

## 测试分析

1. **MM 已不再是主问题**
   - 官方 `MM/AddMM` 在 MLU 上只比 torch 慢约 `1.3x-2.5x`
   - 这与之前 `vllm-nt` 中 MM 慢几十倍的现象不同，说明集成层的大 bug 已经被修掉

2. **RMSNorm 是当前最稳定的瓶颈**
   - `NO_RMS` 或 `NO_RMS_MM_FA` 都能让多个模型显著恢复
   - 官方 `RMSNorm/FusedRMSNorm` 本身在 MLU 上就慢于 torch，说明问题已更多落在 NT kernel 自身

3. **SiLU / SwiGLU 也值得关注**
   - 从算子 benchmark 看，这两类激活在官方 NT 库里对 torch 明显偏慢
   - 但端到端上它们没有像 RMSNorm 那样成为首要瓶颈，可能是调用频次或占比没有 RMSNorm 高

4. **MiniCPM4.1-8B 需要单独分析**
   - 它的恢复比例远低于其它模型
   - 正确性追踪里它没有命中 `RoPE`
   - 它很可能还有模型结构或模型专用路径的问题，不能直接按其他模型的结论外推

## Final Files

- `reports/mlu/correctness_by_model.md`
- `reports/mlu/throughput_matrix.md`
- `reports/mlu/operator_benchmarks.md`
