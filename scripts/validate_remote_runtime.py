#!/usr/bin/env python3
"""Validate vllm-nt registration and basic generation on a target runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from typing import Any


DEFAULT_PROMPT = "In one short sentence, explain what the moon is."


def _discover_plugin_entry_point() -> dict[str, Any]:
    entry_points = importlib.metadata.entry_points()
    if hasattr(entry_points, "select"):
        candidates = entry_points.select(group="vllm.general_plugins")
    else:
        candidates = entry_points.get("vllm.general_plugins", [])
    matches = [
        {
            "name": entry_point.name,
            "value": entry_point.value,
            "group": entry_point.group,
        }
        for entry_point in candidates
        if entry_point.name == "vllm_nt"
    ]
    return {
        "plugin_discoverable": bool(matches),
        "entry_points": matches,
    }


def _check_expected_registered(
    summary: dict[str, Any], expected_registered: set[str]
) -> list[str]:
    registered_ops = set(summary.get("registered_ops", []))
    return sorted(expected_registered - registered_ops)


def _run_generation(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("VLLM_NT_ENABLE_STATS", "1")

    import vllm_nt  # noqa: F401
    from vllm import LLM, SamplingParams
    from vllm_nt.oot import get_usage_summary

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        ),
    )
    summary = get_usage_summary()
    payload = {
        "plugin": _discover_plugin_entry_point(),
        "model": args.model,
        "prompt": args.prompt,
        "output": outputs[0].outputs[0].text,
        "usage_summary": summary,
    }
    expected_registered = {
        item.strip()
        for item in args.expect_registered.split(",")
        if item.strip()
    }
    missing = _check_expected_registered(summary, expected_registered)
    if missing:
        payload["missing_expected_registered"] = missing
    return payload


def _print_payload(payload: dict[str, Any], dump_json: bool) -> None:
    if dump_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    plugin = payload.get("plugin", {})
    print(f"plugin_discoverable: {plugin.get('plugin_discoverable')}")
    if "model" not in payload:
        return

    summary = payload["usage_summary"]
    print(f"model: {payload['model']}")
    print(f"output: {payload['output']}")
    print(f"registered_ops: {', '.join(summary.get('registered_ops', [])) or 'None'}")
    print(f"hit_ops: {', '.join(summary.get('hit_ops', [])) or 'None'}")
    print(f"missed_ops: {', '.join(summary.get('missed_ops', [])) or 'None'}")
    if payload.get("missing_expected_registered"):
        print(
            "missing_expected_registered: "
            + ", ".join(payload["missing_expected_registered"])
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Model path or Hugging Face model id")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--expect-registered", default="")
    parser.add_argument("--check-plugin-discovery", action="store_true")
    parser.add_argument("--dump-json", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.check_plugin_discovery and not args.model:
        payload = {"plugin": _discover_plugin_entry_point()}
        _print_payload(payload, args.dump_json)
        return 0 if payload["plugin"]["plugin_discoverable"] else 1

    if not args.model:
        parser.error("--model is required unless --check-plugin-discovery is used")

    payload = _run_generation(args)
    _print_payload(payload, args.dump_json)
    return 1 if payload.get("missing_expected_registered") else 0


if __name__ == "__main__":
    raise SystemExit(main())
