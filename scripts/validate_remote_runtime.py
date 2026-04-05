#!/usr/bin/env python3
"""Remote runtime validation for vllm-nt.

This script is meant to run on a machine that has:
- a usable accelerator device
- vLLM installed
- a model path accessible to vLLM

It performs three checks:
1. Inspect candidate function patch targets and report which ones exist.
2. Run a real vLLM generation.
3. Print the vllm-nt usage summary and optionally assert that selected ops hit.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any


def _csv(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class TargetProbe:
    module_path: str
    attr_name: str
    exists: bool
    detail: str


def _probe_target(module_path: str, attr_name: str) -> TargetProbe:
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        return TargetProbe(module_path, attr_name, False, f"import failed: {exc}")

    if not hasattr(module, attr_name):
        return TargetProbe(module_path, attr_name, False, "attribute missing")

    target = getattr(module, attr_name)
    return TargetProbe(
        module_path,
        attr_name,
        True,
        f"resolved to {type(target).__name__}",
    )


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate vllm-nt on a remote runtime")
    parser.add_argument("--model", required=True, help="Model path or model id")
    parser.add_argument(
        "--prompt",
        default="Explain rotary embeddings in one sentence.",
        help="Prompt used for generation",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32, help="Number of generated tokens"
    )
    parser.add_argument(
        "--dtype", default="bfloat16", help="LLM dtype passed to vLLM"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="vLLM tensor_parallel_size"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization",
    )
    parser.add_argument(
        "--expect-hit",
        type=_csv,
        default=[],
        help="Comma-separated operator names expected in hit_ops",
    )
    parser.add_argument(
        "--expect-registered",
        type=_csv,
        default=[],
        help="Comma-separated operator names expected in registered_ops",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print usage summary JSON after generation",
    )
    args = parser.parse_args()

    import torch
    import vllm_nt  # noqa: F401
    from vllm import LLM, SamplingParams
    from vllm_nt.oot import _FUNCTION_PATCH_SPECS, get_usage_summary

    _print_header("Environment")
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if hasattr(torch, "mlu"):
        print(f"mlu_available={torch.mlu.is_available()}")

    _print_header("Function Patch Targets")
    seen: set[tuple[str, str]] = set()
    for spec in _FUNCTION_PATCH_SPECS:
        key = (spec.module_path, spec.attr_name)
        if key in seen:
            continue
        seen.add(key)
        probe = _probe_target(spec.module_path, spec.attr_name)
        print(json.dumps(asdict(probe), ensure_ascii=True))

    _print_header("Generation")
    print(f"model={args.model}")
    print(f"dtype={args.dtype}")
    print(f"tensor_parallel_size={args.tensor_parallel_size}")
    print(f"prompt={args.prompt!r}")

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(max_tokens=args.max_tokens)
    outputs = llm.generate([args.prompt], sampling_params)
    text = outputs[0].outputs[0].text
    print(f"completion={text!r}")

    summary = get_usage_summary()
    _print_header("Usage Summary")
    print(f"registered_ops={summary['registered_ops']}")
    print(f"hit_ops={summary['hit_ops']}")
    print(f"missed_ops={summary['missed_ops']}")

    if args.dump_json:
        _print_header("Usage Summary JSON")
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))

    registered = set(summary["registered_ops"])
    hit = set(summary["hit_ops"])

    missing_registered = [name for name in args.expect_registered if name not in registered]
    missing_hit = [name for name in args.expect_hit if name not in hit]

    if missing_registered:
        print(f"missing registered ops: {missing_registered}", file=sys.stderr)
    if missing_hit:
        print(f"missing hit ops: {missing_hit}", file=sys.stderr)

    return 1 if (missing_registered or missing_hit) else 0


if __name__ == "__main__":
    raise SystemExit(main())
