# Benchmark Report Index

Reports are grouped by device backend so future CUDA, MLU, or other device runs can coexist without name collisions.

## Directory Layout

| path | device | purpose |
| --- | --- | --- |
| `reports/mlu/` | MLU | Final MLU validation, throughput, and operator benchmark reports |

## MLU Reports

| report | purpose |
| --- | --- |
| `reports/mlu/README.md` | Final MLU roll-up: what was done, key results, and analysis |
| `reports/mlu/correctness_by_model.md` | Per-model prompt, native/NT output, hit ops, and hit counts |
| `reports/mlu/throughput_matrix.md` | Per-model offline eager throughput matrix with explicit load configuration |
| `reports/mlu/operator_benchmarks.md` | Official NineToothed operator benchmarks vs torch native on current MLU |

## Primary Entry Points

- Start with `reports/mlu/README.md` for the final MLU conclusion.
- Use `reports/mlu/correctness_by_model.md` to trace which operators each model hits.
- Use `reports/mlu/throughput_matrix.md` for end-to-end throughput comparison.
- Use `reports/mlu/operator_benchmarks.md` for operator-level performance comparison.
