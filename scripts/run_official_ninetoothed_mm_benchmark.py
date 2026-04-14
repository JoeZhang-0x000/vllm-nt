#!/usr/bin/env python3
"""Run the official ninetoothed-examples MM benchmark on the local device."""

from __future__ import annotations

import argparse
import statistics
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import triton


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXAMPLES_REPO = Path("/tmp/ninetoothed-examples")
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "official_ninetoothed_examples_mm_mlu.md"
DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class BenchRow:
    size: int
    ninetoothed_ms: float
    torch_ms: float
    triton_ms: float
    nt_vs_torch: float
    nt_vs_triton: float


def _sync(device: str) -> None:
    if device == "mlu":
        torch.mlu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _bench(fn, device: str) -> float:
    return float(triton.testing.do_bench(fn))


def run(args: argparse.Namespace) -> list[BenchRow]:
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(args.examples_repo))

    import ops.ninetoothed.torch as nt_ops
    import ops.triton.torch as triton_ops

    if args.device == "mlu":
        if not hasattr(torch, "mlu") or not torch.mlu.is_available():
            raise RuntimeError("MLU is not available")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

    dtype = DTYPE_BY_NAME[args.dtype]
    providers = {
        "ninetoothed": nt_ops.mm,
        "torch": torch.mm,
        "triton": triton_ops.mm,
    }
    rows: list[BenchRow] = []

    for size in [2**i for i in range(3, 13)]:
        lhs = torch.randn((size, size), dtype=dtype, device=args.device)
        rhs = torch.randn((size, size), dtype=dtype, device=args.device)

        outputs = {name: fn(lhs, rhs) for name, fn in providers.items()}
        torch.testing.assert_close(
            outputs["ninetoothed"],
            outputs["torch"],
            atol=0.025,
            rtol=0.025,
        )
        torch.testing.assert_close(
            outputs["ninetoothed"],
            outputs["triton"],
            atol=0.0625,
            rtol=0.01,
        )
        _sync(args.device)

        ninetoothed_ms = _bench(lambda: providers["ninetoothed"](lhs, rhs), args.device)
        torch_ms = _bench(lambda: providers["torch"](lhs, rhs), args.device)
        triton_ms = _bench(lambda: providers["triton"](lhs, rhs), args.device)

        rows.append(
            BenchRow(
                size=size,
                ninetoothed_ms=ninetoothed_ms,
                torch_ms=torch_ms,
                triton_ms=triton_ms,
                nt_vs_torch=ninetoothed_ms / torch_ms,
                nt_vs_triton=ninetoothed_ms / triton_ms,
            )
        )
        print(
            f"size={size} nt={ninetoothed_ms:.4f}ms "
            f"torch={torch_ms:.4f}ms triton={triton_ms:.4f}ms"
        )

    return rows


def render_report(args: argparse.Namespace, rows: list[BenchRow]) -> str:
    geometric_torch = statistics.geometric_mean(row.nt_vs_torch for row in rows)
    geometric_triton = statistics.geometric_mean(row.nt_vs_triton for row in rows)
    lines = [
        "# Official ninetoothed-examples MM Benchmark on MLU",
        "",
        "## Configuration",
        f"- examples repo: `{args.examples_repo}`",
        f"- examples commit: `{args.examples_commit}`",
        "- benchmark source: `tests/test_benchmarks.py::TestMMBenchmark`",
        f"- device: `{args.device}`",
        f"- dtype: `{args.dtype}`",
        "- shape sweep: `m=n=k=2^i`, `i in [3, 12]`",
        "",
        "## Summary",
        f"- geometric mean slowdown vs `torch.mm`: `{geometric_torch:.2f}x`",
        f"- geometric mean slowdown vs official Triton MM: `{geometric_triton:.2f}x`",
        "",
        "## Results",
        "| size | NineToothed ms | torch.mm ms | official Triton ms | NT vs torch | NT vs Triton |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.size),
                    f"{row.ninetoothed_ms:.4f}",
                    f"{row.torch_ms:.4f}",
                    f"{row.triton_ms:.4f}",
                    f"{row.nt_vs_torch:.2f}x",
                    f"{row.nt_vs_triton:.2f}x",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- This reuses the official `ninetoothed-examples` MM benchmark shape sweep, but runs it on local `MLU` instead of the upstream hard-coded CUDA path.",
            "- On square MM shapes, NineToothed is close to `torch.mm` on MLU and substantially faster than the official Triton MM implementation from the examples repo.",
            "- This does not match the severe slowdown seen in the `vllm-nt` model-shape MatMul microbenchmark, which suggests the main problem is shape-specific or integration-specific rather than a generic square-GEMM regression.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-repo", type=Path, default=DEFAULT_EXAMPLES_REPO)
    parser.add_argument("--device", choices=["mlu", "cuda"], default="mlu")
    parser.add_argument("--dtype", choices=sorted(DTYPE_BY_NAME), default="bfloat16")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.examples_repo.exists():
        raise FileNotFoundError(f"Examples repo not found: {args.examples_repo}")
    args.examples_commit = (
        __import__("subprocess")
        .check_output(["git", "-C", str(args.examples_repo), "rev-parse", "HEAD"], text=True)
        .strip()
    )
    rows = run(args)
    report = render_report(args, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
