# Remote Validation

This document covers the accelerator-backed validation that cannot be completed in the local workspace.

## What Is Already Done Locally

- Vendored Ninetoothed kernels and torch wrappers for:
  - paged attention prefill
  - paged attention decode
  - KV cache store
  - RoPE
  - SDPA
- Added `FunctionPatchSpec` scaffolding for function/backend patch targets
- Added local tests for wrappers and patch scaffolding

## What Still Requires Remote Validation

- Real vLLM hook reachability on the target runtime
- Real `block_table` / `cache_seqlens` / `slot_mapping` layout compatibility
- Real accelerator execution for attention, KV cache, and RoPE paths
- Whether standalone RoPE fallback is reached or bypassed by a fused backend

## Recommended Environment

- Machine with CUDA, MLU, or MUSA available
- Python environment that can import:
  - `torch`
  - `vllm`
  - `vllm_nt`
- A small model for iteration speed

Known MetaX validation host:

```bash
ssh -p 30467 root@ssh.v5000-prod-gw.nhss.zhejianglab.com
```

On that host, use `/opt/conda/bin/python` for non-interactive commands; the default
SSH `PATH` may not include `python`.

Recommended MetaX environment:

```bash
export MACA_HOME=/opt/maca-3.5.3
export MACA_PATH=/opt/maca-3.5.3
export LD_LIBRARY_PATH=/root/.infini/lib:$MACA_PATH/lib:$MACA_PATH/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/root/vllm-nt:${PYTHONPATH:-}
```

When validating attention routing with `vllm_metax`, disable vLLM-NT's base OOT
operator registrations to avoid duplicate names with the platform plugin:

```bash
export VLLM_NT_DISABLE_OPS=RMSNorm,SiluAndMul,MulAndSilu,GeluAndMul,GemmaRMSNorm
```

## Setup

From the remote machine:

```bash
cd /path/to/vllm-nt
git fetch
git checkout feat/nt-attention-rope-sdpa
pip install -e .
```

If `vllm_nt` is not discoverable through vLLM's plugin mechanism, keep using explicit import through the script below.

## InfiniCore Attention Configs

Native InfiniCore paged attention:

```bash
export VLLM_NT_BACKEND_CONFIG=/path/to/vllm-nt/vllm_nt/configs/pa-infinicore.yaml
```

InfiniCore Flash-Attn wrapper path (`mha_varlen` / `mha_kvcache`):

```bash
export VLLM_NT_BACKEND_CONFIG=/path/to/vllm-nt/vllm_nt/configs/pa-infinicore-flash-attn.yaml
```

Both configs keep `StoreKVCache` on `infinicore`; only prefill/decode attention
routing differs.

The Flash-Attn config has been smoke-tested with:

```bash
/mnt/geogpt-doc-new/default/xb/qwen3-0.6B
```

Expected signal for a short decode-heavy prompt:

```text
PagedAttentionDecode: backend=infinicore-flash-attn hits > 0 failures=0
```

For the Flash-Attn wrapper, also run a longer prompt so prefill is forced through
the same route:

```bash
python scripts/run_inference_validation.py run \
  --models qwen3-0.6B \
  --prompt-profile long \
  --batch-size 2 \
  --max-tokens 64 \
  --dtype bfloat16
```

Pass condition:

```text
PagedAttentionPrefill: backend=infinicore-flash-attn hits > 0 failures=0
PagedAttentionDecode: backend=infinicore-flash-attn hits > 0 failures=0
```

If generation succeeds but `PagedAttentionPrefill` remains at `hits=0`, the run
only validates decode. In that case, keep increasing request shape coverage
instead of treating the Flash-Attn prefill adapter as validated.

## Minimal Runtime Validation

Run:

```bash
python scripts/validate_remote_runtime.py \
  --model /path/to/model \
  --dtype bfloat16 \
  --max-tokens 32 \
  --dump-json
```

If you want a stricter check:

```bash
python scripts/validate_remote_runtime.py \
  --model /path/to/model \
  --dtype bfloat16 \
  --max-tokens 32 \
  --expect-registered RMSNorm,SiluAndMul,MatMul,Embedding \
  --dump-json
```

Do not require `PagedAttentionPrefill`, `PagedAttentionDecode`, `RoPE`, or `SDPA` hits on the first run unless you have already confirmed the hook path exists on your exact vLLM build.

## What To Look For

### 1. Function Patch Target Probe

The script prints candidate patch targets like:

```text
{"module_path": "...", "attr_name": "...", "exists": true, "detail": "..."}
```

You want to know:

- which `unified_attention_2d` path actually exists
- whether an `sdpa` op module exists
- whether `RotaryEmbedding.forward_oot` is present in your vLLM version

### 2. Usage Summary

Focus on:

- `registered_ops`
- `hit_ops`
- `missed_ops`

Interpretation:

- `RMSNorm`, `SiluAndMul`, `MatMul`, `Embedding` should be the baseline expectation
- `PagedAttentionPrefill` and `PagedAttentionDecode` only matter if your runtime uses the patched unified attention entry
- `RoPE` hit means the fallback/helper path was actually reached
- `SDPA` hit means the internal SDPA fallback path was exercised

### 3. RoPE Reality Check

If:

- attention hits appear
- but `RoPE` never appears in `hit_ops`

that can still be a valid outcome if your backend fuses RoPE into attention and bypasses standalone RoPE dispatch.

## Suggested Validation Matrix

Run at least these scenarios:

1. Single short prompt
   - confirms plugin import and basic generation

2. Longer prompt with multiple prompt tokens
   - improves odds of exercising prefill-related attention paths

3. Multi-request batch
   - helps verify jagged/prefill path behavior

4. Decode-heavy generation with larger `max_tokens`
   - improves odds of exercising paged decode path

## Failure Triage

### No function targets exist

Meaning:

- the current vLLM build uses different module paths

Action:

- inspect your installed `vllm` package for attention op entry points
- extend `_FUNCTION_PATCH_SPECS` with the real module path

### Targets exist but no paged attention hits

Meaning:

- patch installed but path is not exercised
- or backend uses a different attention kernel path

Action:

- inspect backend selection logs
- verify which attention backend your runtime chose

### RoPE still never hits

Meaning:

- likely fused backend behavior

Action:

- treat standalone RoPE hit as optional unless your backend is known to route through `forward_oot`

### Shape/layout mismatch errors

Meaning:

- `block_table`, `cache_seqlens`, or `slot_mapping` semantics differ from the assumptions in this branch

Action:

- dump the runtime tensor shapes and dtypes
- compare them with the expectations embedded in:
  - `vllm_nt/_ntops/torch/attention.py`
  - `vllm_nt/_ntops/torch/kv_cache.py`

## Files Relevant To Remote Debugging

- `scripts/validate_remote_runtime.py`
- `vllm_nt/_ntops/patching.py`
- `vllm_nt/_ntops/oot_support.py`
- `vllm_nt/_ntops/torch/attention.py`
- `vllm_nt/_ntops/torch/kv_cache.py`
- `vllm_nt/_ntops/torch/rotary_emb.py`
- `vllm_nt/_ntops/torch/sdpa.py`
