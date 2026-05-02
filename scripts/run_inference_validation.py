#!/usr/bin/env python3
"""Run per-model native and NT correctness validation on MLU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from _model_utils import DEFAULT_LONG_PROMPT, DEFAULT_PROMPT, select_models
except ModuleNotFoundError:
    from scripts._model_utils import DEFAULT_LONG_PROMPT, DEFAULT_PROMPT, select_models


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "mlu" / "correctness_by_model.md"
RESULT_PREFIX = "RESULT_JSON="
PROMPT_PROFILES = {
    "short": DEFAULT_PROMPT,
    "long": DEFAULT_LONG_PROMPT,
}
MODE_ENVS = {
    "native": {
        "VLLM_PLUGINS": "mlu,mlu_hijack,lora_filesystem_resolver",
    },
    "NT_ALL_ON": {
        "VLLM_NT_ENABLE_ALL": "1",
        "VLLM_NT_ENABLE_FA": "1",
        "VLLM_NT_ENABLE_MM": "1",
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
    env["VLLM_NT_ENABLE_STATS"] = "1"
    for key in (
        "VLLM_NT_ENABLE_ALL",
        "VLLM_NT_ENABLE_FA",
        "VLLM_NT_ENABLE_MM",
        "VLLM_NT_DISABLE_OPS",
        "VLLM_PLUGINS",
    ):
        env.pop(key, None)
    env.update(MODE_ENVS[mode_name])
    return env


def _run_child(args: argparse.Namespace, model_path: str, mode_name: str) -> dict[str, Any]:
    prompt = _resolve_prompt(args)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "generate-child",
        "--model",
        model_path,
        "--mode-name",
        mode_name,
        "--prompt",
        prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--batch-size",
        str(args.batch_size),
        "--dtype",
        args.dtype,
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
        check=True,
    )
    for stream in (completed.stdout, completed.stderr):
        for line in reversed(stream.splitlines()):
            if line.startswith(RESULT_PREFIX):
                return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(completed.stdout + "\n" + completed.stderr)


def _render_report(args: argparse.Namespace, rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["model_name"], {})[row["mode_name"]] = row

    lines = [
        "# Model Correctness and Hit Tracking on MLU",
        "",
        "## Configuration",
        f"- prompt_profile: `{args.prompt_profile}`",
        f"- prompt: `{_resolve_prompt(args)}`",
        f"- batch_size: `{args.batch_size}`",
        f"- max_tokens: `{args.max_tokens}`",
        f"- dtype: `{args.dtype}`",
        f"- enforce_eager: `1`",
        f"- max_num_configs: `{args.max_num_configs}`",
        f"- max_model_len/max_num_batched_tokens: `{args.max_model_len}` / `{args.max_num_batched_tokens}`",
        f"- mlu_visible_devices: `{args.mlu_visible_devices}`",
        "",
    ]
    for model_name, model_rows in grouped.items():
        native = model_rows["native"]
        nt = model_rows["NT_ALL_ON"]
        usage = nt.get("usage_summary") or {"hit_ops": [], "operators": {}}
        lines.extend(
            [
                f"## `{model_name}`",
                f"- model path: `{native['model_path']}`",
                f"- prompt: `{_resolve_prompt(args)}`",
                f"- native output: `{native['output']}`",
                f"- NT_ALL_ON output: `{nt['output']}`",
                f"- NT hit_ops: `{', '.join(usage['hit_ops']) or 'None'}`",
                "",
                "| operator | hits |",
                "| --- | ---: |",
            ]
        )
        for op_name, op_stats in usage.get("operators", {}).items():
            if op_stats.get("hits", 0) > 0:
                lines.append(f"| {op_name} | {op_stats['hits']} |")
        lines.append("")
    return "\n".join(lines)


def _resolve_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "prompt", None):
        return args.prompt
    profile = getattr(args, "prompt_profile", "short")
    try:
        return PROMPT_PROFILES[profile]
    except KeyError as exc:
        raise RuntimeError(f"unknown prompt profile: {profile}") from exc


def _run_parent(args: argparse.Namespace) -> None:
    rows = []
    for model in select_models(args.models):
        for mode_name in ("native", "NT_ALL_ON"):
            print(f"Running {model.name} {mode_name}...")
            row = _run_child(args, str(model.path), mode_name)
            row["model_name"] = model.name
            row["model_path"] = str(model.path)
            rows.append(row)
    report = _render_report(args, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote report to {args.output}")


def _run_generate_child(args: argparse.Namespace) -> None:
    use_nt = args.mode_name != "native"
    if use_nt:
        import vllm_nt  # noqa: F401
        from vllm_nt.oot import get_usage_summary
    else:
        get_usage_summary = None

    from vllm import LLM, SamplingParams

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
    prompt = _resolve_prompt(args)
    prompts = [prompt] * args.batch_size
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens),
    )
    payload = {
        "mode_name": args.mode_name,
        "model_path": args.model,
        "output": outputs[0].outputs[0].text,
        "batch_size": args.batch_size,
        "usage_summary": get_usage_summary() if get_usage_summary is not None else None,
    }
    print(f"{RESULT_PREFIX}{json.dumps(payload, ensure_ascii=False)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--mlu-visible-devices", default="1")
    run_parser.add_argument("--prompt", default=None)
    run_parser.add_argument("--prompt-profile", choices=sorted(PROMPT_PROFILES), default="short")
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument("--max-tokens", type=int, default=32)
    run_parser.add_argument("--dtype", default="bfloat16")
    run_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    run_parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    run_parser.add_argument("--max-model-len", type=int, default=512)
    run_parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    run_parser.add_argument("--max-num-configs", type=int, default=10)
    run_parser.add_argument("--models", nargs="*", default=None)

    child = subparsers.add_parser("generate-child")
    child.add_argument("--model", required=True)
    child.add_argument("--mode-name", choices=sorted(MODE_ENVS), required=True)
    child.add_argument("--prompt", default=None)
    child.add_argument("--prompt-profile", choices=sorted(PROMPT_PROFILES), default="short")
    child.add_argument("--batch-size", type=int, default=1)
    child.add_argument("--max-tokens", type=int, default=32)
    child.add_argument("--dtype", default="bfloat16")
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
    if args.command == "generate-child":
        _run_generate_child(args)
        return 0
    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
