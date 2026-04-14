#!/usr/bin/env python3
"""Run eager throughput matrix for models listed in down_models.sh."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = Path("/data02/jiangqiu/models")
DOWN_MODELS = MODELS_ROOT / "down_models.sh"
RESULT_PREFIX = "RESULT_JSON="
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "multi_model_eager_throughput_summary.md"
MODE_ORDER = (
    "native",
    "NT_All",
    "NO_RMS",
    "NO_MM",
    "NO_FA",
    "NO_RMS_MM_FA",
)
MODE_SPECS = {
    "native": {
        "mode": "vllm_native",
        "env": {},
    },
    "NT_All": {
        "mode": "nt_all_on",
        "env": {},
    },
    "NO_RMS": {
        "mode": "nt_all_on",
        "env": {"VLLM_NT_DISABLE_OPS": "RMSNorm"},
    },
    "NO_MM": {
        "mode": "nt_all_on",
        "env": {"VLLM_NT_ENABLE_MM": "0"},
    },
    "NO_FA": {
        "mode": "nt_all_on",
        "env": {"VLLM_NT_ENABLE_FA": "0"},
    },
    "NO_RMS_MM_FA": {
        "mode": "nt_all_on",
        "env": {
            "VLLM_NT_DISABLE_OPS": "RMSNorm",
            "VLLM_NT_ENABLE_MM": "0",
            "VLLM_NT_ENABLE_FA": "0",
        },
    },
}
DEFAULT_BATCH_SIZES = {
    "gpt2": 256,
    "Llama-2-7b-chat-hf": 128,
    "Qwen2.5-7B-Instruct": 128,
    "Mistral-7B-Instruct-v0.3": 128,
    "DeepSeek-R1-Distill-Qwen-7B": 96,
    "MiniCPM4.1-8B": 64,
    "glm-4-9b": 64,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    source_model: str


def _parse_model_list() -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for line in DOWN_MODELS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line)
        if "--local_dir" not in parts:
            continue
        local_dir = parts[parts.index("--local_dir") + 1]
        source_model = parts[parts.index("--model") + 1]
        model_path = MODELS_ROOT / local_dir
        if model_path.exists():
            models.append(ModelSpec(local_dir, model_path, source_model))
    return models


def _child_env(args: argparse.Namespace, extra_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(REPO_ROOT)
    )
    env["MLU_VISIBLE_DEVICES"] = args.mlu_visible_devices
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_NT_MAX_NUM_CONFIGS"] = str(args.max_num_configs)
    for key in ("VLLM_NT_DISABLE_OPS", "VLLM_NT_ENABLE_MM", "VLLM_NT_ENABLE_FA"):
        env.pop(key, None)
    env.update(extra_env)
    return env


def _run_case(args: argparse.Namespace, model: ModelSpec, mode_name: str) -> dict[str, Any]:
    spec = MODE_SPECS[mode_name]
    batch_size = DEFAULT_BATCH_SIZES[model.name]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_nt_mlu_suite.py"),
        "throughput-child",
        "--mode",
        spec["mode"],
        "--model",
        str(model.path),
        "--dtype",
        args.dtype,
        "--batch-size",
        str(batch_size),
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
        "--enforce-eager",
        "1",
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_child_env(args, spec["env"]),
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return {
            "model": model.name,
            "mode": mode_name,
            "status": "failed",
            "error": (completed.stderr or completed.stdout)[-4000:],
            "batch_size": batch_size,
        }
    for stream in (completed.stdout, completed.stderr):
        for line in reversed(stream.splitlines()):
            if line.startswith(RESULT_PREFIX):
                payload = json.loads(line[len(RESULT_PREFIX) :])
                payload.update(
                    {
                        "model": model.name,
                        "mode_name": mode_name,
                        "status": "ok",
                    }
                )
                return payload
    return {
        "model": model.name,
        "mode": mode_name,
        "status": "failed",
        "error": "No RESULT_JSON found",
        "batch_size": batch_size,
    }


def _fmt_result(row: dict[str, Any], key: str) -> str:
    if row.get("status") != "ok":
        return "FAIL"
    return f"{row[key]:.2f}"


def _conclusion(model_rows: dict[str, dict[str, Any]]) -> str:
    native = model_rows["native"]
    if native.get("status") != "ok":
        return "native failed"
    base = native["throughput_total_tokens_per_second"]
    comparisons = []
    for mode in MODE_ORDER[1:]:
        row = model_rows[mode]
        if row.get("status") == "ok":
            comparisons.append((mode, row["throughput_total_tokens_per_second"] / base))
    if not comparisons:
        return "no NT result"
    best_mode, best_ratio = max(comparisons, key=lambda item: item[1])
    return f"best {best_mode} ({best_ratio * 100:.1f}% native)"


def _render_report(args: argparse.Namespace, results: list[dict[str, Any]], models: list[ModelSpec]) -> str:
    grouped: dict[str, dict[str, Any]] = {model.name: {} for model in models}
    for row in results:
        grouped[row["model"]][row.get("mode_name", row.get("mode", "unknown"))] = row

    lines = [
        "# Multi-Model Eager Throughput Summary",
        "",
        "## Configuration",
        f"- models root: `{MODELS_ROOT}`",
        f"- model source list: `{DOWN_MODELS}`",
        f"- dtype: `{args.dtype}`",
        f"- enforce_eager: `1`",
        f"- max_num_configs: `{args.max_num_configs}`",
        f"- max_model_len/max_num_batched_tokens: `{args.max_model_len}` / `{args.max_num_batched_tokens}`",
        f"- input_len/output_len: `{args.input_len}` / `{args.output_len}`",
        f"- warmup/measure iters: `{args.warmup_iters}` / `{args.measure_iters}`",
        f"- mlu_visible_devices: `{args.mlu_visible_devices}`",
        "",
        "## Summary Table",
        "| model | batch | native tok/s | NT_All tok/s | NO_RMS tok/s | NO_MM tok/s | NO_FA tok/s | NO_RMS_MM_FA tok/s | conclusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in models:
        rows = grouped[model.name]
        batch = next(
            (str(rows[mode].get("batch_size", DEFAULT_BATCH_SIZES[model.name])) for mode in MODE_ORDER if mode in rows),
            str(DEFAULT_BATCH_SIZES[model.name]),
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    model.name,
                    batch,
                    _fmt_result(rows.get("native", {}), "throughput_total_tokens_per_second"),
                    _fmt_result(rows.get("NT_All", {}), "throughput_total_tokens_per_second"),
                    _fmt_result(rows.get("NO_RMS", {}), "throughput_total_tokens_per_second"),
                    _fmt_result(rows.get("NO_MM", {}), "throughput_total_tokens_per_second"),
                    _fmt_result(rows.get("NO_FA", {}), "throughput_total_tokens_per_second"),
                    _fmt_result(rows.get("NO_RMS_MM_FA", {}), "throughput_total_tokens_per_second"),
                    _conclusion(rows),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Failures"])
    failures = [row for row in results if row.get("status") != "ok"]
    if not failures:
        lines.append("- None")
    else:
        for row in failures:
            lines.append(f"- `{row['model']}` / `{row.get('mode_name', row.get('mode'))}`: {row['error'][:300]}")
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
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=3)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    models = _parse_model_list()
    if args.models:
        selected = set(args.models)
        models = [model for model in models if model.name in selected]
    results: list[dict[str, Any]] = []
    for model in models:
        for mode_name in MODE_ORDER:
            print(f"Running {model.name} {mode_name}...")
            results.append(_run_case(args, model, mode_name))

    report = _render_report(args, results, models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
