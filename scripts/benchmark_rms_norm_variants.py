#!/usr/bin/env python3
"""Benchmark current and official RMSNorm variants on MLU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXAMPLES_REPO = Path("/tmp/ninetoothed-examples")
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "rms_norm_variant_benchmark.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-repo", type=Path, default=DEFAULT_EXAMPLES_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    sys.path.insert(0, str(args.examples_repo))
    import ops.ninetoothed.torch as official_nt
    from vllm_nt._ntops.torch.rms_norm import rms_norm as current_rms_norm

    cases = [
        ("qwen_decode_like", 512, 1024),
        ("qwen_large_m", 8192, 1024),
        ("gpt2_like", 512, 768),
    ]
    rows = []
    print("case,variant,ms")
    for name, m, n in cases:
        x = torch.randn((m, n), device="mlu", dtype=torch.bfloat16)
        w = torch.randn((n,), device="mlu", dtype=torch.bfloat16)
        eps = torch.finfo(x.dtype).eps
        refs = {
            "torch": lambda: F.rms_norm(x, (n,), w, eps),
            "current": lambda: current_rms_norm(x, n, w, eps),
            "official_fused": lambda: official_nt.fused_rms_norm(x, w, eps),
            "official_plain": lambda: official_nt.rms_norm(x, eps),
        }
        outs = {key: fn() for key, fn in refs.items()}
        torch.testing.assert_close(outs["current"], outs["torch"], atol=0.03125, rtol=0.02)
        torch.testing.assert_close(outs["official_fused"], outs["torch"], atol=0.03125, rtol=0.02)
        for key, fn in refs.items():
            ms = float(triton.testing.do_bench(fn))
            print(f"{name},{key},{ms:.4f}")
            rows.append((name, m, n, key, ms))

    by_case = {}
    for case, m, n, variant, ms in rows:
        by_case.setdefault((case, m, n), {})[variant] = ms

    lines = [
        "# RMSNorm Variant Benchmark on MLU",
        "",
        "## Configuration",
        f"- examples repo: `{args.examples_repo}`",
        "- dtype: `bfloat16`",
        "- device: `MLU`",
        "",
        "## Results",
        "| case | M | N | torch ms | current vllm-nt ms | official fused NT ms | official plain NT ms | current vs torch | current vs official fused |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (case, m, n), values in by_case.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    str(m),
                    str(n),
                    f"{values['torch']:.4f}",
                    f"{values['current']:.4f}",
                    f"{values['official_fused']:.4f}",
                    f"{values['official_plain']:.4f}",
                    f"{values['current'] / values['torch']:.2f}x",
                    f"{values['current'] / values['official_fused']:.2f}x",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "- The current `vllm-nt` RMSNorm fast path now matches the official `fused_rms_norm` implementation closely.",
            "- The official fused implementation is still far slower than `torch.nn.functional.rms_norm` on these MLU shapes, so RMSNorm remains a backend/kernel-level bottleneck rather than only a `vllm-nt` wrapper issue.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
