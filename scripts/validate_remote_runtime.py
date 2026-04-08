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
import os
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
    object_name: str | None
    attr_name: str
    exists: bool
    detail: str


def _probe_target(
    module_path: str, attr_name: str, object_name: str | None = None
) -> TargetProbe:
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        return TargetProbe(
            module_path, object_name, attr_name, False, f"import failed: {exc}"
        )

    target_obj = module
    if object_name is not None:
        if not hasattr(module, object_name):
            return TargetProbe(
                module_path, object_name, attr_name, False, "object missing"
            )
        target_obj = getattr(module, object_name)

    if not hasattr(target_obj, attr_name):
        return TargetProbe(
            module_path, object_name, attr_name, False, "attribute missing"
        )

    target = getattr(target_obj, attr_name)
    return TargetProbe(
        module_path,
        object_name,
        attr_name,
        True,
        f"resolved to {type(target).__name__}",
    )


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _probe_custom_op(qualified_name: str) -> dict[str, object]:
    try:
        namespace, op_name = qualified_name.split(".", 1)
        namespace_obj = getattr(__import__("torch").ops, namespace)
        exists = hasattr(namespace_obj, op_name)
        overload = getattr(namespace_obj, op_name, None)
        has_default = hasattr(overload, "default") if overload is not None else False
        return {
            "qualified_name": qualified_name,
            "exists": exists,
            "has_default": has_default,
            "detail": type(overload).__name__ if overload is not None else "missing",
        }
    except Exception as exc:
        return {
            "qualified_name": qualified_name,
            "exists": False,
            "has_default": False,
            "detail": f"probe failed: {exc}",
        }


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
    print(
        "experimental_forward_patch="
        f"{os.environ.get('VLLM_NT_ENABLE_EXPERIMENTAL_FORWARD_PATCH', '0')}"
    )
    print(
        "custom_op_rebind="
        f"{os.environ.get('VLLM_NT_ENABLE_CUSTOM_OP_REBIND', '0')}"
    )
    print(
        "custom_op_register_intercept="
        f"{os.environ.get('VLLM_NT_ENABLE_CUSTOM_OP_REGISTER_INTERCEPT', '0')}"
    )

    _print_header("Platform Diagnostics")
    try:
        from vllm.platforms import current_platform
        import vllm.envs as envs
        is_cuda_alike = current_platform.is_cuda_alike()
        is_cpu = current_platform.is_cpu()
        is_oot = current_platform.is_out_of_tree()
        use_v1 = envs.VLLM_USE_V1
        # Mirror Attention_MluHjack.__init__ logic
        base_use_direct_call = (not is_cuda_alike) and (not is_cpu)
        final_use_direct_call = False if (is_oot and use_v1) else base_use_direct_call
        print(f"platform={type(current_platform).__name__}")
        print(f"is_cuda_alike={is_cuda_alike}")
        print(f"is_cpu={is_cpu}")
        print(f"is_out_of_tree={is_oot}")
        print(f"VLLM_USE_V1={use_v1}")
        print(f"expected_use_direct_call={final_use_direct_call}")
        dispatch_key = getattr(current_platform, "dispatch_key", "unknown")
        print(f"dispatch_key={dispatch_key}")
    except Exception as exc:
        print(f"platform diagnostics failed: {exc}")

    _print_header("Custom Op Dispatch Tables")
    try:
        for op_fullname in (
            "vllm::unified_attention",
            "vllm::unified_attention_with_output",
        ):
            try:
                table = torch._C._dispatch_dump_table(op_fullname)
                print(f"--- {op_fullname} ---")
                print(table)
            except Exception as exc:
                print(f"{op_fullname}: dispatch_dump_table failed: {exc}")
    except Exception as exc:
        print(f"dispatch table probe failed: {exc}")

    _print_header("Custom Op Targets")
    for qualified_name in (
        "vllm.unified_attention",
        "vllm.unified_kv_cache_update",
        "vllm.unified_attention_with_output",
    ):
        print(json.dumps(_probe_custom_op(qualified_name), ensure_ascii=True))

    _print_header("Function Patch Targets")
    seen: set[tuple[str, str | None, str]] = set()
    for spec in _FUNCTION_PATCH_SPECS:
        key = (spec.module_path, spec.object_name, spec.attr_name)
        if key in seen:
            continue
        seen.add(key)
        probe = _probe_target(spec.module_path, spec.attr_name, spec.object_name)
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
