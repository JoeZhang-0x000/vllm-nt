#!/usr/bin/env python3
"""Run official ninetoothed operator benchmarks against torch native on MLU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXAMPLES_REPO = Path("/tmp/ninetoothed-examples")
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "mlu" / "operator_benchmarks.md"


def _bench(fn) -> float:
    return float(triton.testing.do_bench(fn))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-repo", type=Path, default=DEFAULT_EXAMPLES_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sys.path.insert(0, str(args.examples_repo))
    import ops.ninetoothed.torch as nt_ops

    device = "mlu"
    dtype = torch.bfloat16
    rows: list[tuple[str, str, str, str, str]] = []

    def record(operator: str, case: str, nt_fn, torch_fn) -> None:
        try:
            nt_ms = _bench(nt_fn)
            torch_ms = _bench(torch_fn)
            rows.append((operator, case, f"{nt_ms:.4f}", f"{torch_ms:.4f}", f"{nt_ms / torch_ms:.2f}x"))
        except Exception as exc:  # pragma: no cover - benchmark environment specific
            rows.append((operator, case, "FAIL", "FAIL", str(exc).splitlines()[0][:120]))

    # MM
    for name, m, k, n in [
        ("square_1024", 1024, 1024, 1024),
        ("gpt2_mlp_up", 8192, 768, 3072),
        ("qwen_lm_head", 256, 1024, 151936),
    ]:
        a = torch.randn((m, k), device=device, dtype=dtype)
        b = torch.randn((k, n), device=device, dtype=dtype)
        record("MM", name, lambda: nt_ops.mm(a, b), lambda: torch.mm(a, b))

    # AddMM
    for name, m, k, n in [
        ("gpt2_fc_bias", 8192, 768, 768),
        ("qwen_proj_bias", 8192, 1024, 1024),
    ]:
        inp = torch.randn((m, n), device=device, dtype=dtype)
        a = torch.randn((m, k), device=device, dtype=dtype)
        b = torch.randn((k, n), device=device, dtype=dtype)
        record("AddMM", name, lambda: nt_ops.addmm(inp, a, b), lambda: torch.addmm(inp, a, b))

    # RMSNorm
    for name, m, n in [
        ("qwen_decode", 512, 1024),
        ("qwen_prefill", 4096, 1024),
        ("llama_decode", 512, 4096),
    ]:
        x = torch.randn((m, n), device=device, dtype=dtype)
        record("RMSNorm", name, lambda: nt_ops.rms_norm(x), lambda: F.rms_norm(x, x.shape[-1:]))

    # FusedRMSNorm
    for name, m, n in [
        ("qwen_decode", 512, 1024),
        ("qwen_prefill", 4096, 1024),
        ("llama_decode", 512, 4096),
    ]:
        x = torch.randn((m, n), device=device, dtype=dtype)
        w = torch.randn((n,), device=device, dtype=dtype)
        record("FusedRMSNorm", name, lambda: nt_ops.fused_rms_norm(x, w, 1e-5), lambda: F.rms_norm(x, x.shape[-1:], w, 1e-5))

    # SiLU
    for name, shape in [
        ("gpt2_mlp", (8192, 3072)),
        ("qwen_mlp", (8192, 4096)),
    ]:
        x = torch.randn(shape, device=device, dtype=dtype)
        record("SiLU", name, lambda: nt_ops.silu(x), lambda: F.silu(x))

    # SwiGLU
    for name, shape in [
        ("gpt2_like", (8192, 3072)),
        ("qwen_like", (8192, 4096)),
    ]:
        a = torch.randn(shape, device=device, dtype=dtype)
        b = torch.randn(shape, device=device, dtype=dtype)
        record("SwiGLU", name, lambda: nt_ops.swiglu(a, b), lambda: F.silu(a) * b)

    # Softmax
    for name, shape in [
        ("sampling_small_vocab", (256, 32000)),
        ("sampling_large_vocab", (256, 152064)),
        ("attn_scores", (4096, 128)),
    ]:
        x = torch.randn(shape, device=device, dtype=dtype)
        record("Softmax", name, lambda: nt_ops.softmax(x), lambda: torch.softmax(x, dim=-1))

    lines = [
        "# Official Ninetoothed Operator Benchmarks on MLU",
        "",
        "## Configuration",
        f"- examples repo: `{args.examples_repo}`",
        "- device: `MLU`",
        "- dtype: `bfloat16`",
        "- compared against: `torch native`",
        "",
        "## Results",
        "| operator | case | NT ms | torch ms | slowdown |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for operator, case, nt_ms, torch_ms, slowdown in rows:
        lines.append(f"| {operator} | {case} | {nt_ms} | {torch_ms} | {slowdown} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
