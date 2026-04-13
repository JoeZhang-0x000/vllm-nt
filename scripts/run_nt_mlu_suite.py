#!/usr/bin/env python3
"""Run batched accuracy and offline throughput comparisons for vllm-nt on MLU."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
RESULT_PREFIX = "RESULT_JSON="
ACCURACY_PROMPTS = [
    "Answer in one short sentence: what is the moon?",
    "Answer with one word only: the capital of France is",
    "Translate to English: 今天天气很好，我们一起去散步。",
]
MODEL_PATTERNS = {
    "qwen3_0.6b": ["models--Qwen--Qwen3-0.6B"],
    "gpt2": ["models--openai-community--gpt2", "models--gpt2"],
    "llama2_7b": [
        "models--meta-llama--Llama-2-7b-hf",
        "models--*Llama-2-7b*",
        "models--*llama*2*7b*",
    ],
}
DEFAULT_BATCH_SIZES = {
    "qwen3_0.6b": 128,
    "gpt2": 256,
    "llama2_7b": 16,
}
ACCURACY_MODES = {
    "nt_all_off": {
        "VLLM_NT_ENABLE_ALL": "0",
    },
    "nt_all_on": {
        "VLLM_NT_ENABLE_ALL": "1",
        "VLLM_NT_ENABLE_FA": "1",
        "VLLM_NT_ENABLE_MM": "1",
    },
}
THROUGHPUT_MODES = {
    "vllm_native": {
        "VLLM_PLUGINS": "mlu,mlu_hijack,lora_filesystem_resolver",
    },
    "nt_all_on": {
        "VLLM_NT_ENABLE_ALL": "1",
        "VLLM_NT_ENABLE_FA": "1",
        "VLLM_NT_ENABLE_MM": "1",
    },
    "nt_disable_fa": {
        "VLLM_NT_ENABLE_ALL": "1",
        "VLLM_NT_ENABLE_FA": "0",
        "VLLM_NT_ENABLE_MM": "1",
    },
    "nt_disable_fa_mm": {
        "VLLM_NT_ENABLE_ALL": "1",
        "VLLM_NT_ENABLE_FA": "0",
        "VLLM_NT_ENABLE_MM": "0",
    },
}


@dataclass(frozen=True)
class ResolvedModel:
    key: str
    model_id: str
    snapshot_path: Path


def _snapshot_from_ref(ref_file: Path) -> Path | None:
    snapshot = ref_file.read_text(encoding="utf-8").strip()
    if not snapshot:
        return None
    candidate = ref_file.parent.parent / "snapshots" / snapshot
    return candidate if candidate.exists() else None


def resolve_models() -> tuple[list[ResolvedModel], dict[str, str]]:
    resolved: list[ResolvedModel] = []
    missing: dict[str, str] = {}
    for key, patterns in MODEL_PATTERNS.items():
        ref_files: list[Path] = []
        for pattern in patterns:
            ref_files.extend(sorted(HF_CACHE_ROOT.glob(f"{pattern}/refs/main")))
        for ref_file in ref_files:
            snapshot = _snapshot_from_ref(ref_file)
            if snapshot is not None:
                repo_dir = ref_file.parents[1].name
                model_id = repo_dir.removeprefix("models--").replace("--", "/")
                resolved.append(
                    ResolvedModel(
                        key=key,
                        model_id=model_id,
                        snapshot_path=snapshot,
                    )
                )
                break
        else:
            missing[key] = f"No cached snapshot found under {HF_CACHE_ROOT}"
    return resolved, missing


def _child_env(mode_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(REPO_ROOT)
    )
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.update(mode_env)
    return env


def _run_child(subcommand: str, *, mode: str, model: Path, extra_args: list[str]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        subcommand,
        "--mode",
        mode,
        "--model",
        str(model),
        *extra_args,
    ]
    mode_env = THROUGHPUT_MODES.get(mode) or ACCURACY_MODES.get(mode)
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_child_env(mode_env or {}),
        capture_output=True,
        text=True,
        check=True,
    )
    for stream in (completed.stdout, completed.stderr):
        for line in reversed(stream.splitlines()):
            if line.startswith(RESULT_PREFIX):
                return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(
        f"Failed to parse child result for {subcommand} {mode}.\n"
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )


def _load_vllm_runtime(use_nt: bool) -> tuple[Any, Any, Any | None]:
    usage_summary = None
    if use_nt:
        import vllm_nt  # noqa: F401
        from vllm_nt.oot import get_usage_summary

        usage_summary = get_usage_summary
    from vllm import LLM, SamplingParams

    return LLM, SamplingParams, usage_summary


def _apply_mode_env(mode: str) -> None:
    mode_env = THROUGHPUT_MODES.get(mode) or ACCURACY_MODES.get(mode)
    if mode_env is None:
        return
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.update(mode_env)


def _run_accuracy_child(args: argparse.Namespace) -> None:
    _apply_mode_env(args.mode)
    use_nt = args.mode != "vllm_native"
    LLM, SamplingParams, get_usage_summary = _load_vllm_runtime(use_nt)

    llm_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(ACCURACY_PROMPTS, sampling_params)
    result = {
        "mode": args.mode,
        "model": args.model,
        "prompts": ACCURACY_PROMPTS,
        "outputs": [output.outputs[0].text for output in outputs],
        "usage_summary": get_usage_summary() if get_usage_summary else None,
    }
    print(f"{RESULT_PREFIX}{json.dumps(result, ensure_ascii=False)}")


def _build_prompt_from_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)


def _build_synthetic_prompts(
    tokenizer: Any,
    *,
    batch_size: int,
    input_len: int,
    offset: int,
) -> list[str]:
    prompts: list[str] = []
    base_text = (
        "Summarize the following technical note in one sentence and keep the wording precise. "
    )
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    for index in range(batch_size):
        prefix = f"Request {offset + index}: "
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        token_ids = list(prefix_ids)
        while len(token_ids) < input_len:
            token_ids.extend(base_ids)
        prompts.append(_build_prompt_from_token_ids(tokenizer, token_ids[:input_len]))
    return prompts


def _count_output_tokens(output: Any, tokenizer: Any) -> int:
    candidate = output.outputs[0]
    token_ids = getattr(candidate, "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    return len(tokenizer.encode(candidate.text, add_special_tokens=False))


def _run_throughput_child(args: argparse.Namespace) -> None:
    _apply_mode_env(args.mode)
    use_nt = args.mode != "vllm_native"
    LLM, SamplingParams, get_usage_summary = _load_vllm_runtime(use_nt)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.output_len,
        min_tokens=args.output_len,
        ignore_eos=True,
    )

    prompt_tokens_per_iter: list[int] = []
    output_tokens_per_iter: list[int] = []
    elapsed_seconds: list[float] = []

    total_iters = args.warmup_iters + args.measure_iters
    for iteration in range(total_iters):
        prompts = _build_synthetic_prompts(
            tokenizer,
            batch_size=args.batch_size,
            input_len=args.input_len,
            offset=iteration * args.batch_size,
        )
        prompt_tokens = sum(
            len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
        )
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start_time
        output_tokens = sum(_count_output_tokens(output, tokenizer) for output in outputs)

        if iteration >= args.warmup_iters:
            prompt_tokens_per_iter.append(prompt_tokens)
            output_tokens_per_iter.append(output_tokens)
            elapsed_seconds.append(elapsed)

    total_prompt_tokens = sum(prompt_tokens_per_iter)
    total_output_tokens = sum(output_tokens_per_iter)
    total_elapsed = sum(elapsed_seconds)
    result = {
        "mode": args.mode,
        "model": args.model,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "measure_iters": args.measure_iters,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "mean_elapsed_seconds": statistics.mean(elapsed_seconds),
        "total_elapsed_seconds": total_elapsed,
        "throughput_total_tokens_per_second": (total_prompt_tokens + total_output_tokens)
        / total_elapsed,
        "throughput_output_tokens_per_second": total_output_tokens / total_elapsed,
        "usage_summary": get_usage_summary() if get_usage_summary else None,
    }
    print(f"{RESULT_PREFIX}{json.dumps(result, ensure_ascii=False)}")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _render_report(
    *,
    resolved: list[ResolvedModel],
    missing: dict[str, str],
    accuracy_results: dict[str, dict[str, Any]],
    throughput_results: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    configured_max_num_configs = os.environ.get("VLLM_NT_MAX_NUM_CONFIGS")
    configured_max_num_configs_mode = (
        "(explicit)"
        if configured_max_num_configs is not None
        else os.environ.get("VLLM_NT_MAX_NUM_CONFIGS_MODE", "quick")
    )
    lines = [
        "# NT MLU Validation Report",
        "",
        "## Configuration",
        f"- dtype: `{args.dtype}`",
        f"- tensor_parallel_size: `{args.tensor_parallel_size}`",
        f"- gpu_memory_utilization: `{args.gpu_memory_utilization}`",
        f"- max_num_configs: `{configured_max_num_configs or '(mode/default)'}`",
        f"- max_num_configs_mode: `{configured_max_num_configs_mode}`",
        f"- max_model_len: `{args.max_model_len or '(model default)'}`",
        f"- max_num_batched_tokens: `{args.max_num_batched_tokens or '(vLLM default)'}`",
        f"- mlu_visible_devices: `{os.environ.get('MLU_VISIBLE_DEVICES', '(default)')}`",
        "- v1_multiprocessing: `0`",
        f"- accuracy max tokens: `{args.accuracy_max_tokens}`",
        f"- throughput input/output lengths: `{args.throughput_input_len}` / `{args.throughput_output_len}`",
        f"- throughput warmup/measure iterations: `{args.warmup_iters}` / `{args.measure_iters}`",
        "",
        "## Accuracy Prompts",
    ]
    for index, prompt in enumerate(ACCURACY_PROMPTS, start=1):
        lines.append(f"{index}. {prompt}")
    lines.append("")

    if missing:
        lines.append("## Missing Models")
        for key, reason in missing.items():
            lines.append(f"- `{key}`: {reason}")
        lines.append("")

    lines.append("## Model Results")
    lines.append("")
    for model in resolved:
        lines.append(f"### `{model.key}`")
        lines.append(f"- model id: `{model.model_id}`")
        lines.append(f"- cache path: `{model.snapshot_path}`")
        lines.append("")
        lines.append("#### Accuracy")
        for mode in ("nt_all_off", "nt_all_on"):
            result = accuracy_results[model.key][mode]
            lines.append(f"- mode: `{mode}`")
            usage = result.get("usage_summary")
            if usage is not None:
                lines.append(f"- hit_ops: `{', '.join(usage['hit_ops']) or 'None'}`")
            for prompt, output in zip(result["prompts"], result["outputs"]):
                lines.append(f"- prompt: {prompt}")
                lines.append(f"- output: {output!r}")
            lines.append("")

        lines.append("#### Throughput")
        throughput_rows: list[list[str]] = []
        for mode in ("vllm_native", "nt_all_on", "nt_disable_fa", "nt_disable_fa_mm"):
            result = throughput_results[model.key][mode]
            usage = result.get("usage_summary")
            hit_ops = ", ".join(usage["hit_ops"]) if usage is not None else "native"
            throughput_rows.append(
                [
                    mode,
                    str(result["batch_size"]),
                    f"{result['throughput_total_tokens_per_second']:.2f}",
                    f"{result['throughput_output_tokens_per_second']:.2f}",
                    f"{result['mean_elapsed_seconds']:.4f}",
                    hit_ops or "None",
                ]
            )
        lines.extend(
            _markdown_table(
                [
                    "mode",
                    "batch",
                    "total tok/s",
                    "output tok/s",
                    "mean sec/iter",
                    "hit_ops",
                ],
                throughput_rows,
            )
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _run_suite(args: argparse.Namespace) -> None:
    if args.max_num_configs is not None:
        os.environ["VLLM_NT_MAX_NUM_CONFIGS"] = str(args.max_num_configs)
    elif args.max_num_configs_mode is not None:
        os.environ["VLLM_NT_MAX_NUM_CONFIGS_MODE"] = args.max_num_configs_mode

    resolved, missing = resolve_models()
    accuracy_results: dict[str, dict[str, Any]] = {}
    throughput_results: dict[str, dict[str, Any]] = {}

    for model in resolved:
        accuracy_results[model.key] = {}
        for mode in ACCURACY_MODES:
            accuracy_results[model.key][mode] = _run_child(
                "accuracy-child",
                mode=mode,
                model=Path(model.model_id),
                extra_args=[
                    "--dtype",
                    args.dtype,
                    "--max-tokens",
                    str(args.accuracy_max_tokens),
                    "--tensor-parallel-size",
                    str(args.tensor_parallel_size),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--max-model-len",
                    str(args.max_model_len or 0),
                    "--max-num-batched-tokens",
                    str(args.max_num_batched_tokens or 0),
                ],
            )

        throughput_results[model.key] = {}
        batch_size = DEFAULT_BATCH_SIZES[model.key]
        for mode in THROUGHPUT_MODES:
            throughput_results[model.key][mode] = _run_child(
                "throughput-child",
                mode=mode,
                model=Path(model.model_id),
                extra_args=[
                    "--dtype",
                    args.dtype,
                    "--batch-size",
                    str(batch_size),
                    "--input-len",
                    str(args.throughput_input_len),
                    "--output-len",
                    str(args.throughput_output_len),
                    "--warmup-iters",
                    str(args.warmup_iters),
                    "--measure-iters",
                    str(args.measure_iters),
                    "--tensor-parallel-size",
                    str(args.tensor_parallel_size),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--max-model-len",
                    str(args.max_model_len or 0),
                    "--max-num-batched-tokens",
                    str(args.max_num_batched_tokens or 0),
                ],
            )

    report = _render_report(
        resolved=resolved,
        missing=missing,
        accuracy_results=accuracy_results,
        throughput_results=throughput_results,
        args=args,
    )
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    suite_parser = subparsers.add_parser("run")
    suite_parser.add_argument("--output", default="reports/nt_mlu_validation_report.md")
    suite_parser.add_argument("--dtype", default="bfloat16")
    suite_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    suite_parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    suite_parser.add_argument("--max-model-len", type=int, default=None)
    suite_parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    suite_parser.add_argument("--accuracy-max-tokens", type=int, default=64)
    suite_parser.add_argument("--throughput-input-len", type=int, default=64)
    suite_parser.add_argument("--throughput-output-len", type=int, default=64)
    suite_parser.add_argument("--warmup-iters", type=int, default=1)
    suite_parser.add_argument("--measure-iters", type=int, default=3)
    suite_parser.add_argument("--max-num-configs", type=int, default=None)
    suite_parser.add_argument(
        "--max-num-configs-mode",
        choices=["quick", "tuning"],
        default=None,
    )

    for name in ("accuracy-child", "throughput-child"):
        child_parser = subparsers.add_parser(name)
        child_parser.add_argument("--mode", required=True)
        child_parser.add_argument("--model", required=True)
        child_parser.add_argument("--dtype", default="bfloat16")
        child_parser.add_argument("--tensor-parallel-size", type=int, default=1)
        child_parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
        child_parser.add_argument("--max-model-len", type=int, default=0)
        child_parser.add_argument("--max-num-batched-tokens", type=int, default=0)

        if name == "accuracy-child":
            child_parser.add_argument("--max-tokens", type=int, default=64)
        else:
            child_parser.add_argument("--batch-size", type=int, required=True)
            child_parser.add_argument("--input-len", type=int, required=True)
            child_parser.add_argument("--output-len", type=int, required=True)
            child_parser.add_argument("--warmup-iters", type=int, default=1)
            child_parser.add_argument("--measure-iters", type=int, default=3)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        _run_suite(args)
        return 0
    if args.command == "accuracy-child":
        _run_accuracy_child(args)
        return 0
    if args.command == "throughput-child":
        _run_throughput_child(args)
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
