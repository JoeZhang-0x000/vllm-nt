"""Minimal vllm-nt inference example.

Usage:
    python example.py --model /path/to/your/model
    python example.py --model /path/to/your/model --prompt "Hello, world!"
    python example.py --model /path/to/your/model --max-tokens 64 --dtype float16
"""

import argparse
import os

DEFAULT_PROMPT = "What is the capital of France?"


def main():
    parser = argparse.ArgumentParser(description="vllm-nt inference example")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Input prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=32, help="Max tokens to generate"
    )
    parser.add_argument(
        "--dtype", default="bfloat16", help="Model dtype (default: bfloat16)"
    )
    args = parser.parse_args()

    # vllm-nt registers automatically via entry_point,
    # but explicit import ensures it works even without pip install
    os.environ.setdefault("VLLM_NT_ENABLE_STATS", "1")
    import vllm_nt  # noqa: F401
    from vllm_nt.oot import maybe_print_usage_summary

    from vllm import LLM, SamplingParams

    print(f"Model:  {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Dtype:  {args.dtype}")
    print("-" * 40)

    llm = LLM(model=args.model, dtype=args.dtype)
    sampling_params = SamplingParams(max_tokens=args.max_tokens)

    outputs = llm.generate([args.prompt], sampling_params)

    for output in outputs:
        print(output.outputs[0].text)

    maybe_print_usage_summary()


if __name__ == "__main__":
    main()
