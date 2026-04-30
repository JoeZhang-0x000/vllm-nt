# vllm-nt

NineToothed operator plugin for vLLM.

## vLLM CLI Usage

Install `vllm-nt` into the same Python environment that runs vLLM:

```bash
pip install -e .
```

Run the stock vLLM server with the plugin enabled:

```bash
VLLM_PLUGINS=vllm_nt \
VLLM_NT_ENABLE_STATS=1 \
vllm serve /path/to/model
```

If the target runtime also needs vendor plugins, keep them in the comma-separated
list and add `vllm_nt`, for example:

```bash
VLLM_PLUGINS=mlu,mlu_hijack,lora_filesystem_resolver,vllm_nt \
VLLM_NT_ENABLE_STATS=1 \
vllm serve /path/to/model
```

`vllm-nt` is registered through vLLM's `vllm.general_plugins` entry point. The
plugin only installs the NT operator hooks; it does not replace or wrap vLLM's
OpenAI API server, engine lifecycle, or CLI argument parsing.

If the installed vLLM build cannot discover general plugins, use explicit import
as a fallback in a small Python entrypoint before constructing `vllm.LLM`.
`example.py` shows that fallback path.

## Runtime Validation

On the target accelerator machine:

```bash
python scripts/validate_remote_runtime.py \
  --model /path/to/model \
  --dtype bfloat16 \
  --max-tokens 32 \
  --expect-registered RMSNorm,SiluAndMul,MatMul,Embedding \
  --dump-json
```

To check whether the package metadata exposes the vLLM plugin entry point:

```bash
python scripts/validate_remote_runtime.py --check-plugin-discovery --dump-json
```

The runtime validation prints registered operators, hit operators, missed
operators, and the generated text. Attention, RoPE, and SDPA hits depend on the
exact vLLM backend path selected by the target runtime.
