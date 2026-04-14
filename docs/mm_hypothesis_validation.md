# MLU MatMul Hypothesis Validation

目标：解释为什么官方 `ninetoothed-examples` 的 `MM` 在本机 MLU 上只比 `torch.mm` 慢约 `1.1x-1.5x`，而 `vllm-nt` 中的 `MatMul/Linear` 在真实模型 shape 上会慢几十倍。

## 假设清单

- [x] H1: `vllm-nt` 当前 `MatMul/Linear` kernel 实现与官方 `ninetoothed-examples` `MM` kernel 不一致。
- [x] H2: `vllm-nt` 当前 `MatMul/Linear` 默认把 `BLOCK_SIZE_M/N/K` 固定为 `64`，导致九齿 autotune 实际没有在 MM 上发挥作用。
- [x] H3: `linear` 包装层或 bias 路径是主要开销来源，而不是 MM kernel 本身。
- [x] H4: 官方基准只覆盖 square GEMM，`vllm-nt` 的主要问题集中在真实模型里的矩形 / 极宽 shape。
- [x] H5: 修复 MM kernel 行为后，`vllm-nt` 的 MatMul microbenchmark 和端到端吞吐会显著改善。

## 验证记录

### H1
- 状态：已验证，成立。
- 结果：修复前 `current matmul vs official NT` 的几何平均慢速比是 `34.48x`（square）和 `28.39x`（model），说明同一台 MLU 上官方 MM 和 `vllm-nt` 当前 MM 行为明显不同。见 `reports/mm_hypothesis_benchmark_current.md`。

### H2
- 状态：已验证，成立。
- 结果：将 `vllm_nt/_ntops/kernels/matmul.py` 和 `vllm_nt/_ntops/kernels/linear.py` 从固定 `64/64/64` 改成官方 examples 同款 `ninetoothed.block_size()` 后，`current matmul vs official NT` 从 `34.48x/28.39x` 降到 `1.52x/1.39x`（square/model，`max_num_configs=10`），说明固定 tile 是主因。见 `reports/mm_hypothesis_benchmark_fixed_tuning.md`。

### H3
- 状态：已验证，不成立。
- 结果：修复前 `linear+bias vs linear` 的几何平均只多 `1.03x`，修复后也只有 `1.18x`，说明 bias/包装层不是几十倍退化的主因。见 `reports/mm_hypothesis_benchmark_current.md` 和 `reports/mm_hypothesis_benchmark_fixed_tuning.md`。

### H4
- 状态：已验证，部分不成立。
- 结果：问题不只出现在矩形 shape，因为修复前连 `1024/2048/4096` square GEMM 也比官方 MM 慢 `27.76x-41.50x`。但修复后矩形/极宽 shape 仍比官方 MM 慢约 `1.13x-1.62x`，所以 shape 仍会放大剩余差距。见 `reports/mm_hypothesis_benchmark_current.md` 与 `reports/mm_hypothesis_benchmark_fixed_tuning.md`。

### H5
- 状态：已验证，成立。
- 结果：修复后端到端吞吐明显改善。`gpt2 nt_all_on` 从 `9333.07` 提升到 `36441.27 tok/s`，接近 native `46731.59`；`qwen3 nt_all_on` 从 `913.44` 提升到 `1113.04 tok/s`，说明 MM 修复有效，但 Qwen 仍受其它 NT 算子限制。修复后的完整结果见 `reports/nt_mlu_validation_report.md`。

## 当前结论

- `vllm-nt` MM 大幅退化的主因已经定位到：`MatMul/Linear` 默认把 block size 固定成 `64/64/64`，没有真正走到官方 examples 那套可调 block size/autotune 路径。
- 修复后，当前 MM 已经回到与官方 MM 同量级：在模型 shape 上约慢 `1.39x`，不再是几十倍。
- `gpt2` 的端到端吞吐也随之大幅恢复，说明之前的大头瓶颈确实是 MM。
- `qwen3` 仍然明显慢于 native，后续应优先继续排查 `RMSNorm`、`SiluAndMul`、`Embedding`、`RoPE`、`TopKTopP`、`RandomSample` 等非 MM 路径。
