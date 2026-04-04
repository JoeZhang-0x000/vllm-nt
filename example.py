"""Minimal vllm-nt inference example.

Usage:
    python example.py --model /path/to/your/model
    python example.py --model /path/to/your/model --prompt "Hello, world!"
    python example.py --model /path/to/your/model --max-tokens 64 --dtype float16
"""

import argparse

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
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable graph capture / compiled execution to simplify debugging",
    )
    args = parser.parse_args()

    # vllm-nt registers automatically via entry_point,
    # but explicit import ensures it works even without pip install
    import vllm_nt  # noqa: F401
    from vllm_nt.oot import maybe_print_usage_summary

    from vllm import LLM, SamplingParams

    print(f"Model:  {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Dtype:  {args.dtype}")
    print("-" * 40)

    llm = LLM(model=args.model, dtype=args.dtype, enforce_eager=args.enforce_eager)
    sampling_params = SamplingParams(max_tokens=args.max_tokens)

    outputs = llm.generate([args.prompt], sampling_params)

    for output in outputs:
        print(output.outputs[0].text)

    maybe_print_usage_summary()


if __name__ == "__main__":
    main()
