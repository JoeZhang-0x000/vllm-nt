# Benchmark Report Index

Reports are grouped by device backend so future CUDA, MLU, or other device runs can coexist without name collisions.

## Directory Layout

| path | device | purpose |
| --- | --- | --- |
| `reports/mlu/` | MLU | Current MLU tuning, ablation, and throughput reports |

## MLU Reports

| report | purpose |
| --- | --- |
| `reports/mlu/README.md` | MLU result roll-up: what was tested, operator-level findings, end-to-end results, and analysis |
| `reports/mlu/multi_model_eager_throughput_summary.md` | Final 7-model eager throughput summary across `native`, `NT_All`, `NO_RMS`, `NO_MM`, `NO_FA`, and `NO_RMS_MM_FA` |
| `reports/mlu/nt_mlu_validation_report.md` | Qwen3/gpt2 end-to-end validation after the MM fix |
| `reports/mlu/qwen3_non_mm_ablation.md` | Qwen3 non-MM ablation identifying `RMSNorm` as the dominant remaining bottleneck |
| `reports/mlu/rms_norm_variant_benchmark.md` | RMSNorm variant benchmark comparing torch native, current `vllm-nt`, and official NineToothed variants |
| `reports/mlu/mm_hypothesis_benchmark_current.md` | MM hypothesis benchmark before the block-size autotune fix |
| `reports/mlu/mm_hypothesis_benchmark_fixed_tuning.md` | MM hypothesis benchmark after the block-size autotune fix with `max_num_configs=10` |
| `reports/mlu/official_ninetoothed_examples_mm_mlu.md` | Official `ninetoothed-examples` MM benchmark adapted to local MLU |
| `reports/mlu/nt_matmul_microbenchmark.md` | Standalone NT MatMul microbenchmark report |

## Primary Entry Points

- For the complete MLU conclusion, start with `reports/mlu/README.md`.
- For the latest multi-model table only, see `reports/mlu/multi_model_eager_throughput_summary.md`.
- For the full MLU tuning narrative, see `docs/mlu_nt_tuning_and_benchmark.md`.
- For MM root-cause validation, see `docs/mm_hypothesis_validation.md`.
