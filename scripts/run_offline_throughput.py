#!/usr/bin/env python3
"""Run offline eager throughput matrix across selected models on MLU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from _model_utils import DEFAULT_BATCH_SIZES, select_models


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "mlu" / "throughput_matrix.md"
RESULT_PREFIX = "RESULT_JSON="
MODE_ORDER = (
    "native",
    "NT_All",
    "NO_RMS",
    "NO_MM",
    "NO_FA",
    "NO_RMS_MM_FA",
)
MODE_ENVS = {
    "native": {
        "VLLM_PLUGINS": "mlu,mlu_hijack,lora_filesystem_resolver",
    },
    "NT_All": {
        "VLLM_NT_ENABLE_ALL": "1",
        "VLLM_NT_ENABLE_FA": "1",
        "VLLM_NT_ENABLE_MM": "1",
    },
    "NO_RMS": {
        "VLLM_NT_ENABLE_ALL": "1",
        "VLLM_NT_ENABLE_FA": "1",
        "VLLM_NT_ENABLE_MM": "1",
        "VLLM_NT_DISABLE_OPS": "RMSNorm",
    },
    "NO_MM": {
        "VLLM_NT_ENABLE_ALL": "1",
        "VLLM_NT_ENABLE_FA": "1",
        "VLLM_NT_ENABLE_MM": "0",
    },
    "NO_FA": {
        "VLLM_NT_ENABLE_ALL": "1",
        "VLLM_NT_ENABLE_FA": "0",
        "VLLM_NT_ENABLE_MM": "1",
    },
    "NO_RMS_MM_FA": {
        "VLLM_NT_ENABLE_ALL": "1",
        "VLLM_NT_ENABLE_FA": "0",
        "VLLM_NT_ENABLE_MM": "0",
        "VLLM_NT_DISABLE_OPS": "RMSNorm",
    },
}


def _child_env(args: argparse.Namespace, mode_name: str) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(REPO_ROOT)
    )
    env["MLU_VISIBLE_DEVICES"] = args.mlu_visible_devices
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_NT_MAX_NUM_CONFIGS"] = str(args.max_num_configs)
    for key in (
        "VLLM_PLUGINS",
        "VLLM_NT_ENABLE_ALL",
        "VLLM_NT_ENABLE_FA",
        "VLLM_NT_ENABLE_MM",
        "VLLM_NT_DISABLE_OPS",
    ):
        env.pop(key, None)
    env.update(MODE_ENVS[mode_name])
    return env


def _run_child(args: argparse.Namespace, model_name: str, model_path: str, mode_name: str) -> dict[str, Any]:
    batch_size = DEFAULT_BATCH_SIZES[model_name]
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "throughput-child",
        "--model",
        model_path,
        "--model-name",
        model_name,
        "--mode-name",
        mode_name,
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
        env=_child_env(args, mode_name),
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return {
            "model_name": model_name,
            "mode_name": mode_name,
            "status": "failed",
            "batch_size": batch_size,
            "error": (completed.stderr or completed.stdout)[-4000:],
        }
    for stream in (completed.stdout, completed.stderr):
        for line in reversed(stream.splitlines()):
            if line.startswith(RESULT_PREFIX):
                return json.loads(line[len(RESULT_PREFIX) :])
    return {
        "model_name": model_name,
        "mode_name": mode_name,
        "status": "failed",
        "batch_size": batch_size,
        "error": "No RESULT_JSON found",
    }


def _render_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["model_name"], {})[row["mode_name"]] = row

    lines = [
        "# Offline Throughput Matrix on MLU",
        "",
        "## Load Configuration",
        f"- dtype: `{args.dtype}`",
        "- enforce_eager: `1`",
        f"- input_len/output_len: `{args.input_len}` / `{args.output_len}`",
        f"- warmup/measure iters: `{args.warmup_iters}` / `{args.measure_iters}`",
        f"- max_model_len/max_num_batched_tokens: `{args.max_model_len}` / `{args.max_num_batched_tokens}`",
        f"- max_num_configs: `{args.max_num_configs}`",
        f"- mlu_visible_devices: `{args.mlu_visible_devices}`",
        "",
        "## Summary Table",
        "| model | batch | native tok/s | NT_All tok/s | NO_RMS tok/s | NO_MM tok/s | NO_FA tok/s | NO_RMS_MM_FA tok/s | conclusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model_name, mode_rows in grouped.items():
        native = mode_rows.get("native", {})
        if native.get("status") == "ok":
            base = native["throughput_total_tokens_per_second"]
            best_mode = None
            best_ratio = -1.0
            for mode_name in MODE_ORDER[1:]:
                row = mode_rows.get(mode_name, {})
                if row.get("status") == "ok":
                    ratio = row["throughput_total_tokens_per_second"] / base
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_mode = mode_name
            conclusion = f"best {best_mode} ({best_ratio * 100:.1f}% native)" if best_mode else "no NT result"
        else:
            conclusion = "native failed"

        def fmt(mode_name: str) -> str:
            row = mode_rows.get(mode_name, {})
            if row.get("status") != "ok":
                return "FAIL"
            return f"{row['throughput_total_tokens_per_second']:.2f}"

        batch = next((str(mode_rows[m].get("batch_size", DEFAULT_BATCH_SIZES[model_name])) for m in MODE_ORDER if m in mode_rows), str(DEFAULT_BATCH_SIZES[model_name]))
        lines.append(
            f"| `{model_name}` | `{batch}` | {fmt('native')} | {fmt('NT_All')} | {fmt('NO_RMS')} | {fmt('NO_MM')} | {fmt('NO_FA')} | {fmt('NO_RMS_MM_FA')} | {conclusion} |"
        )

    failures = [row for row in rows if row.get("status") != "ok"]
    lines.extend(["", "## Failures"])
    if not failures:
        lines.append("- None")
    else:
        for row in failures:
            lines.append(f"- `{row['model_name']}` / `{row['mode_name']}`: {row['error'][:300]}")
    return "\n".join(lines) + "\n"


def _run_parent(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for model in select_models(args.models):
        for mode_name in MODE_ORDER:
            print(f"Running {model.name} {mode_name}...")
            row = _run_child(args, model.name, str(model.path), mode_name)
            rows.append(row)
    report = _render_report(args, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")


def _run_throughput_child(args: argparse.Namespace) -> None:
    use_nt = args.mode_name != "native"
    if use_nt:
        import vllm_nt  # noqa: F401
        from vllm_nt.oot import get_usage_summary
    else:
        get_usage_summary = None
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        enforce_eager=bool(args.enforce_eager),
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.output_len,
        min_tokens=args.output_len,
        ignore_eos=True,
    )

    base_text = "Summarize this note in one sentence and keep the wording precise. "
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)

    prompt_tokens_sum = 0
    output_tokens_sum = 0
    elapsed = 0.0
    for iteration in range(args.warmup_iters + args.measure_iters):
        prompts = []
        for index in range(args.batch_size):
            prefix_ids = tokenizer.encode(f"Request {iteration * args.batch_size + index}: ", add_special_tokens=False)
            token_ids = list(prefix_ids)
            while len(token_ids) < args.input_len:
                token_ids.extend(base_ids)
            prompts.append(tokenizer.decode(token_ids[: args.input_len], clean_up_tokenization_spaces=False))
        prompt_tokens = sum(len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts)
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        delta = time.perf_counter() - start
        output_tokens = sum(len(getattr(output.outputs[0], "token_ids", []) or tokenizer.encode(output.outputs[0].text, add_special_tokens=False)) for output in outputs)
        if iteration >= args.warmup_iters:
            prompt_tokens_sum += prompt_tokens
            output_tokens_sum += output_tokens
            elapsed += delta

    payload = {
        "model_name": args.model_name,
        "mode_name": args.mode_name,
        "status": "ok",
        "batch_size": args.batch_size,
        "throughput_total_tokens_per_second": (prompt_tokens_sum + output_tokens_sum) / elapsed,
        "throughput_output_tokens_per_second": output_tokens_sum / elapsed,
        "usage_summary": get_usage_summary() if get_usage_summary is not None else None,
    }
    print(f"{RESULT_PREFIX}{json.dumps(payload, ensure_ascii=False)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--mlu-visible-devices", default="1")
    run_parser.add_argument("--dtype", default="bfloat16")
    run_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    run_parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    run_parser.add_argument("--max-model-len", type=int, default=512)
    run_parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    run_parser.add_argument("--max-num-configs", type=int, default=10)
    run_parser.add_argument("--input-len", type=int, default=64)
    run_parser.add_argument("--output-len", type=int, default=64)
    run_parser.add_argument("--warmup-iters", type=int, default=1)
    run_parser.add_argument("--measure-iters", type=int, default=3)
    run_parser.add_argument("--models", nargs="*", default=None)

    child = subparsers.add_parser("throughput-child")
    child.add_argument("--model", required=True)
    child.add_argument("--model-name", required=True)
    child.add_argument("--mode-name", choices=MODE_ORDER, required=True)
    child.add_argument("--dtype", default="bfloat16")
    child.add_argument("--batch-size", type=int, required=True)
    child.add_argument("--input-len", type=int, required=True)
    child.add_argument("--output-len", type=int, required=True)
    child.add_argument("--warmup-iters", type=int, default=1)
    child.add_argument("--measure-iters", type=int, default=3)
    child.add_argument("--tensor-parallel-size", type=int, default=1)
    child.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    child.add_argument("--max-model-len", type=int, default=512)
    child.add_argument("--max-num-batched-tokens", type=int, default=512)
    child.add_argument("--enforce-eager", type=int, choices=[0, 1], default=1)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        _run_parent(args)
        return 0
    if args.command == "throughput-child":
        _run_throughput_child(args)
        return 0
    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
