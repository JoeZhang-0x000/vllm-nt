#!/usr/bin/env python3
"""Compare official ninetoothed MM with vllm-nt MM/Linear on MLU."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXAMPLES_REPO = Path("/tmp/ninetoothed-examples")
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "mm_hypothesis_benchmark_current.md"
DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class Case:
    name: str
    m: int
    k: int
    n: int


SQUARE_CASES = [
    Case("square_1024", 1024, 1024, 1024),
    Case("square_2048", 2048, 2048, 2048),
    Case("square_4096", 4096, 4096, 4096),
]

MODEL_CASES = [
    Case("gpt2_fc", 8192, 768, 768),
    Case("gpt2_mlp_up", 8192, 768, 3072),
    Case("gpt2_mlp_down", 8192, 3072, 768),
    Case("gpt2_lm_head", 256, 768, 50257),
    Case("qwen3_attn", 8192, 1024, 1024),
    Case("qwen3_mlp_up", 8192, 1024, 3072),
    Case("qwen3_mlp_down", 8192, 1536, 1024),
    Case("qwen3_lm_head", 256, 1024, 151936),
]


@dataclass(frozen=True)
class Row:
    case: str
    m: int
    k: int
    n: int
    official_nt_ms: float
    current_matmul_ms: float
    current_linear_ms: float
    current_linear_bias_ms: float
    torch_ms: float


def _bench(fn) -> float:
    return float(triton.testing.do_bench(fn))


def _load_impls(examples_repo: Path):
    sys.path.insert(0, str(examples_repo))
    import ops.ninetoothed.torch as official_nt_ops
    from vllm_nt._ntops.torch.linear import linear as current_linear
    from vllm_nt._ntops.torch.matmul import matmul as current_matmul

    return official_nt_ops.mm, current_matmul, current_linear


def _run_case(case: Case, *, dtype: torch.dtype, device: str, official_mm, current_matmul, current_linear) -> Row:
    lhs = torch.randn((case.m, case.k), dtype=dtype, device=device)
    rhs = torch.randn((case.k, case.n), dtype=dtype, device=device)
    weight = rhs.t().contiguous()
    bias = torch.randn((case.n,), dtype=dtype, device=device)

    official_out = official_mm(lhs, rhs)
    matmul_out = current_matmul(lhs, rhs)
    linear_out = current_linear(lhs, weight, None)
    linear_bias_out = current_linear(lhs, weight, bias)
    torch_out = torch.mm(lhs, rhs)

    torch.testing.assert_close(official_out, torch_out, atol=0.025, rtol=0.025)
    torch.testing.assert_close(matmul_out, torch_out, atol=0.025, rtol=0.025)
    torch.testing.assert_close(linear_out, torch_out, atol=0.025, rtol=0.025)
    torch.testing.assert_close(linear_bias_out, torch_out + bias.view(1, -1), atol=0.025, rtol=0.025)

    return Row(
        case=case.name,
        m=case.m,
        k=case.k,
        n=case.n,
        official_nt_ms=_bench(lambda: official_mm(lhs, rhs)),
        current_matmul_ms=_bench(lambda: current_matmul(lhs, rhs)),
        current_linear_ms=_bench(lambda: current_linear(lhs, weight, None)),
        current_linear_bias_ms=_bench(lambda: current_linear(lhs, weight, bias)),
        torch_ms=_bench(lambda: torch.mm(lhs, rhs)),
    )


def _render_table(rows: list[Row]) -> list[str]:
    lines = [
        "| case | M | K | N | official NT ms | current matmul ms | current linear ms | current linear+bias ms | torch.mm ms | current matmul vs official | current matmul vs torch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.case,
                    str(row.m),
                    str(row.k),
                    str(row.n),
                    f"{row.official_nt_ms:.4f}",
                    f"{row.current_matmul_ms:.4f}",
                    f"{row.current_linear_ms:.4f}",
                    f"{row.current_linear_bias_ms:.4f}",
                    f"{row.torch_ms:.4f}",
                    f"{row.current_matmul_ms / row.official_nt_ms:.2f}x",
                    f"{row.current_matmul_ms / row.torch_ms:.2f}x",
                ]
            )
            + " |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-repo", type=Path, default=DEFAULT_EXAMPLES_REPO)
    parser.add_argument("--device", choices=["mlu", "cuda"], default="mlu")
    parser.add_argument("--dtype", choices=sorted(DTYPE_BY_NAME), default="bfloat16")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.device == "mlu" and (not hasattr(torch, "mlu") or not torch.mlu.is_available()):
        raise RuntimeError("MLU is not available")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    official_mm, current_matmul, current_linear = _load_impls(args.examples_repo)
    dtype = DTYPE_BY_NAME[args.dtype]

    square_rows = [
        _run_case(case, dtype=dtype, device=args.device, official_mm=official_mm, current_matmul=current_matmul, current_linear=current_linear)
        for case in SQUARE_CASES
    ]
    model_rows = [
        _run_case(case, dtype=dtype, device=args.device, official_mm=official_mm, current_matmul=current_matmul, current_linear=current_linear)
        for case in MODEL_CASES
    ]

    examples_commit = subprocess.check_output(
        ["git", "-C", str(args.examples_repo), "rev-parse", "HEAD"], text=True
    ).strip()

    lines = [
        "# MM Hypothesis Benchmark (Current vllm-nt)",
        "",
        "## Configuration",
        f"- examples repo: `{args.examples_repo}`",
        f"- examples commit: `{examples_commit}`",
        f"- device: `{args.device}`",
        f"- dtype: `{args.dtype}`",
        "",
        "## Square Cases",
        *_render_table(square_rows),
        "",
        "## Model Cases",
        *_render_table(model_rows),
        "",
        "## Summary",
        f"- geometric mean current matmul vs official NT (square): `{statistics.geometric_mean(row.current_matmul_ms / row.official_nt_ms for row in square_rows):.2f}x`",
        f"- geometric mean current matmul vs official NT (model): `{statistics.geometric_mean(row.current_matmul_ms / row.official_nt_ms for row in model_rows):.2f}x`",
        f"- geometric mean current linear+bias vs current linear (model): `{statistics.geometric_mean(row.current_linear_bias_ms / row.current_linear_ms for row in model_rows):.2f}x`",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
