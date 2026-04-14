#!/usr/bin/env python3
"""Run qwen3 non-MM ablations on top of the MLU throughput benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_PREFIX = "RESULT_JSON="
QWEN3_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "qwen3_non_mm_ablation.md"


@dataclass(frozen=True)
class AblationCase:
    name: str
    mode: str
    disabled_ops: tuple[str, ...] = ()


CASES = [
    AblationCase("native", "vllm_native"),
    AblationCase("nt_all_on", "nt_all_on"),
    AblationCase("nt_disable_fa", "nt_disable_fa"),
    AblationCase("nt_disable_fa_mm", "nt_disable_fa_mm"),
    AblationCase("ablate_rmsnorm_keep_mm", "nt_disable_fa", ("RMSNorm",)),
    AblationCase("ablate_rmsnorm", "nt_disable_fa_mm", ("RMSNorm",)),
    AblationCase("ablate_silu_and_mul", "nt_disable_fa_mm", ("SiluAndMul",)),
    AblationCase("ablate_embedding", "nt_disable_fa_mm", ("Embedding",)),
    AblationCase("ablate_rope", "nt_disable_fa_mm", ("RoPE",)),
    AblationCase("ablate_topk_topp", "nt_disable_fa_mm", ("TopKTopP",)),
    AblationCase("ablate_random_sample", "nt_disable_fa_mm", ("RandomSample",)),
    AblationCase(
        "ablate_sampling",
        "nt_disable_fa_mm",
        ("TopKTopP", "RandomSample"),
    ),
]


def _run_case(args: argparse.Namespace, case: AblationCase) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    env["MLU_VISIBLE_DEVICES"] = args.mlu_visible_devices
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_NT_MAX_NUM_CONFIGS"] = str(args.max_num_configs)
    if case.disabled_ops:
        env["VLLM_NT_DISABLE_OPS"] = ",".join(case.disabled_ops)
    else:
        env.pop("VLLM_NT_DISABLE_OPS", None)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_nt_mlu_suite.py"),
        "throughput-child",
        "--mode",
        case.mode,
        "--model",
        QWEN3_MODEL,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--warmup-iters",
        str(args.warmup_iters),
        "--measure-iters",
        str(args.measure_iters),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    for stream in (completed.stdout, completed.stderr):
        for line in reversed(stream.splitlines()):
            if line.startswith(RESULT_PREFIX):
                payload = json.loads(line[len(RESULT_PREFIX) :])
                payload["case_name"] = case.name
                payload["disabled_ops"] = list(case.disabled_ops)
                return payload
    raise RuntimeError(completed.stdout + "\n" + completed.stderr)


def _render_report(args: argparse.Namespace, rows: list[dict]) -> str:
    by_name = {row["case_name"]: row for row in rows}
    native = float(by_name["native"]["throughput_total_tokens_per_second"])
    baseline = float(by_name["nt_disable_fa_mm"]["throughput_total_tokens_per_second"])

    lines = [
        "# Qwen3 Non-MM Ablation on MLU",
        "",
        "## Configuration",
        f"- model: `{QWEN3_MODEL}`",
        f"- dtype: `{args.dtype}`",
        f"- batch_size: `{args.batch_size}`",
        f"- input_len/output_len: `{args.input_len}` / `{args.output_len}`",
        f"- max_num_configs: `{args.max_num_configs}`",
        f"- max_model_len/max_num_batched_tokens: `{args.max_model_len}` / `{args.max_num_batched_tokens}`",
        f"- mlu_visible_devices: `{args.mlu_visible_devices}`",
        "",
        "## Throughput",
        "| case | mode | disabled_ops | total tok/s | delta vs nt_disable_fa_mm | % of native | hit_ops |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        total = float(row["throughput_total_tokens_per_second"])
        delta = total - baseline
        pct_native = total / native * 100
        hit_ops = ", ".join(row["usage_summary"]["hit_ops"]) if row.get("usage_summary") else "native"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_name"],
                    row["mode"],
                    ", ".join(row["disabled_ops"]) or "-",
                    f"{total:.2f}",
                    f"{delta:+.2f}",
                    f"{pct_native:.2f}%",
                    hit_ops or "-",
                ]
            )
            + " |"
        )

    ranked = sorted(
        [row for row in rows if row["case_name"].startswith("ablate_")],
        key=lambda row: float(row["throughput_total_tokens_per_second"]),
        reverse=True,
    )
    lines.extend([
        "",
        "## Reading",
        f"- `native`: `{native:.2f} tok/s`",
        f"- `nt_disable_fa_mm`: `{baseline:.2f} tok/s`，这代表在 FA 和 MM 都关闭后，剩余 NT 非 MM 路径的基线吞吐。",
    ])
    if ranked:
        top = ranked[0]
        lines.append(
            f"- 最大单项收益是 `{top['case_name']}`，吞吐 `{float(top['throughput_total_tokens_per_second']):.2f} tok/s`，相对 `nt_disable_fa_mm` 提升 `{float(top['throughput_total_tokens_per_second']) - baseline:+.2f} tok/s`。"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mlu-visible-devices", default="1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-configs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=3)
    args = parser.parse_args()

    rows = [_run_case(args, case) for case in CASES]
    report = _render_report(args, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
