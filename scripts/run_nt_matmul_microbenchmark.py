#!/usr/bin/env python3
"""Benchmark NT MatMul kernels against the native MLU matmul path."""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
COMMON_M_VALUES = (1, 16, 128, 512, 1024, 4096, 8192)
LM_HEAD_M_VALUES = (1, 16, 128, 256)
MODEL_PATTERNS = {
    "qwen3_0.6b": ["models--Qwen--Qwen3-0.6B"],
    "gpt2": ["models--openai-community--gpt2", "models--gpt2"],
    "llama2_7b": [
        "models--meta-llama--Llama-2-7b-hf",
        "models--*Llama-2-7b*",
        "models--*llama*2*7b*",
    ],
}


@dataclass(frozen=True)
class MatMulCase:
    model: str
    name: str
    k: int
    n: int
    m_values: tuple[int, ...] = COMMON_M_VALUES


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    case: str
    m: int
    k: int
    n: int
    native_ms: float
    nt_linear_ms: float
    nt_matmul_ms: float
    native_tflops: float
    nt_linear_tflops: float
    nt_matmul_tflops: float
    nt_linear_slowdown: float
    nt_matmul_slowdown: float
    iters: int


CASES = [
    MatMulCase("gpt2", "fc_768_768", 768, 768),
    MatMulCase("gpt2", "mlp_up_768_3072", 768, 3072),
    MatMulCase("gpt2", "mlp_down_3072_768", 3072, 768),
    MatMulCase("gpt2", "lm_head_768_50257", 768, 50257, LM_HEAD_M_VALUES),
    MatMulCase("qwen3_0.6b", "attn_1024_1024", 1024, 1024),
    MatMulCase("qwen3_0.6b", "mlp_up_1024_3072", 1024, 3072),
    MatMulCase("qwen3_0.6b", "mlp_down_1536_1024", 1536, 1024),
    MatMulCase("qwen3_0.6b", "lm_head_1024_151936", 1024, 151936, LM_HEAD_M_VALUES),
    MatMulCase("llama2_7b", "attn_4096_4096", 4096, 4096),
    MatMulCase("llama2_7b", "mlp_up_4096_11008", 4096, 11008),
    MatMulCase("llama2_7b", "mlp_down_11008_4096", 11008, 4096),
    MatMulCase("llama2_7b", "lm_head_4096_32000", 4096, 32000, LM_HEAD_M_VALUES),
]


def _sync() -> None:
    torch.mlu.synchronize()


def _snapshot_from_ref(ref_file: Path) -> Path | None:
    snapshot = ref_file.read_text(encoding="utf-8").strip()
    if not snapshot:
        return None
    candidate = ref_file.parent.parent / "snapshots" / snapshot
    return candidate if candidate.exists() else None


def _find_cached_models() -> tuple[set[str], dict[str, str]]:
    found: set[str] = set()
    missing: dict[str, str] = {}
    for model, patterns in MODEL_PATTERNS.items():
        for pattern in patterns:
            for ref_file in sorted(HF_CACHE_ROOT.glob(f"{pattern}/refs/main")):
                if _snapshot_from_ref(ref_file) is not None:
                    found.add(model)
                    break
            if model in found:
                break
        if model not in found:
            missing[model] = f"No cached snapshot found under {HF_CACHE_ROOT}"
    return found, missing


def _estimate_iters(m: int, k: int, n: int, *, max_iters: int) -> int:
    flops = 2 * m * k * n
    if flops >= 2_000_000_000:
        return min(max_iters, 5)
    if flops >= 500_000_000:
        return min(max_iters, 10)
    if flops >= 100_000_000:
        return min(max_iters, 20)
    return max_iters


def _bench(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = fn()
    _sync()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        output = fn()
        _sync()
        samples.append(time.perf_counter() - start)
    assert output is not None
    return statistics.median(samples), output


def _tflops(m: int, k: int, n: int, seconds: float) -> float:
    if seconds <= 0:
        return math.inf
    return (2 * m * k * n) / seconds / 1e12


def run_benchmarks(args: argparse.Namespace) -> tuple[list[BenchmarkResult], dict[str, str]]:
    from vllm_mlu import _mlu_ops as mlu_ops
    from vllm_nt._ntops.torch import linear as nt_linear
    from vllm_nt._ntops.torch import matmul as nt_matmul
    from vllm_nt._ntops.torch.utils import (
        get_default_max_num_configs,
        set_default_max_num_configs,
        set_default_max_num_configs_mode,
    )

    if not hasattr(torch, "mlu") or not torch.mlu.is_available():
        raise RuntimeError("MLU is not available in this environment")

    if args.max_num_configs is not None:
        set_default_max_num_configs(args.max_num_configs)
    elif args.max_num_configs_mode is not None:
        set_default_max_num_configs_mode(args.max_num_configs_mode)
    args.effective_max_num_configs = get_default_max_num_configs()

    found_models, missing = _find_cached_models()
    selected_models = set(args.models)
    runnable_models = selected_models & found_models
    for model in selected_models - found_models:
        missing.setdefault(model, f"No cached snapshot found under {HF_CACHE_ROOT}")

    dtype = DTYPE_BY_NAME[args.dtype]
    results: list[BenchmarkResult] = []

    for case in CASES:
        if case.model not in runnable_models:
            continue
        for m in case.m_values:
            x = torch.randn((m, case.k), device="mlu", dtype=dtype)
            weight = torch.randn((case.n, case.k), device="mlu", dtype=dtype)
            weight_t = weight.t().contiguous()
            iters = _estimate_iters(m, case.k, case.n, max_iters=args.iters)

            native_seconds, native_output = _bench(
                lambda: mlu_ops.matmul(x, weight, None, None, "none", 1.0, 0.0),
                warmup=args.warmup,
                iters=iters,
            )
            nt_linear_seconds, nt_linear_output = _bench(
                lambda: nt_linear(x, weight, None),
                warmup=args.warmup,
                iters=iters,
            )
            nt_matmul_seconds, nt_matmul_output = _bench(
                lambda: nt_matmul(x, weight_t),
                warmup=args.warmup,
                iters=iters,
            )

            torch.testing.assert_close(
                nt_linear_output,
                native_output,
                rtol=args.rtol,
                atol=args.atol,
            )
            torch.testing.assert_close(
                nt_matmul_output,
                native_output,
                rtol=args.rtol,
                atol=args.atol,
            )

            results.append(
                BenchmarkResult(
                    model=case.model,
                    case=case.name,
                    m=m,
                    k=case.k,
                    n=case.n,
                    native_ms=native_seconds * 1000,
                    nt_linear_ms=nt_linear_seconds * 1000,
                    nt_matmul_ms=nt_matmul_seconds * 1000,
                    native_tflops=_tflops(m, case.k, case.n, native_seconds),
                    nt_linear_tflops=_tflops(m, case.k, case.n, nt_linear_seconds),
                    nt_matmul_tflops=_tflops(m, case.k, case.n, nt_matmul_seconds),
                    nt_linear_slowdown=nt_linear_seconds / native_seconds,
                    nt_matmul_slowdown=nt_matmul_seconds / native_seconds,
                    iters=iters,
                )
            )
            print(
                f"{case.model} {case.name} M={m}: "
                f"native={native_seconds * 1000:.4f}ms "
                f"nt_linear={nt_linear_seconds * 1000:.4f}ms "
                f"nt_matmul={nt_matmul_seconds * 1000:.4f}ms"
            )

            del x, weight, weight_t, native_output, nt_linear_output, nt_matmul_output
            gc.collect()
            torch.mlu.empty_cache()

    return results, {k: v for k, v in missing.items() if k in selected_models}


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render_report(
    results: list[BenchmarkResult],
    missing: dict[str, str],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# NT MatMul Microbenchmark",
        "",
        "## Configuration",
        f"- dtype: `{args.dtype}`",
        f"- warmup: `{args.warmup}`",
        f"- max iters: `{args.iters}`",
        f"- max_num_configs: `{args.effective_max_num_configs}`",
        f"- max_num_configs_mode: `{args.max_num_configs_mode or '(explicit/default)'}`",
        f"- rtol/atol: `{args.rtol}` / `{args.atol}`",
        f"- mlu_visible_devices: `{os.environ.get('MLU_VISIBLE_DEVICES', '(default)')}`",
        "- native path: `vllm_mlu._mlu_ops.matmul`",
        "- NT linear path: `vllm_nt._ntops.torch.linear`",
        "- NT matmul path: `vllm_nt._ntops.torch.matmul`",
        "",
    ]
    if missing:
        lines.append("## Missing Models")
        for model, reason in sorted(missing.items()):
            lines.append(f"- `{model}`: {reason}")
        lines.append("")

    lines.append("## Results")
    for model in args.models:
        model_results = [result for result in results if result.model == model]
        if not model_results:
            continue
        lines.extend(["", f"### `{model}`"])
        rows = [
            [
                result.case,
                str(result.m),
                str(result.k),
                str(result.n),
                f"{result.native_ms:.4f}",
                f"{result.nt_linear_ms:.4f}",
                f"{result.nt_matmul_ms:.4f}",
                f"{result.nt_linear_slowdown:.2f}x",
                f"{result.nt_matmul_slowdown:.2f}x",
                f"{result.native_tflops:.3f}",
                f"{result.nt_linear_tflops:.3f}",
                f"{result.nt_matmul_tflops:.3f}",
                str(result.iters),
            ]
            for result in model_results
        ]
        lines.extend(
            _table(
                [
                    "case",
                    "M",
                    "K",
                    "N",
                    "native ms",
                    "NT linear ms",
                    "NT matmul ms",
                    "NT linear slowdown",
                    "NT matmul slowdown",
                    "native TFLOPS",
                    "NT linear TFLOPS",
                    "NT matmul TFLOPS",
                    "iters",
                ],
                rows,
            )
        )

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/mlu/nt_matmul_microbenchmark.md")
    parser.add_argument("--dtype", choices=sorted(DTYPE_BY_NAME), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--max-num-configs", type=int, default=None)
    parser.add_argument(
        "--max-num-configs-mode",
        choices=["quick", "tuning"],
        default=None,
    )
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_PATTERNS),
        default=["gpt2", "qwen3_0.6b", "llama2_7b"],
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.effective_max_num_configs = args.max_num_configs
    results, missing = run_benchmarks(args)
    report = render_report(results, missing, args)
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
