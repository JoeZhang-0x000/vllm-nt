# Official Ninetoothed Operator Benchmarks on MLU

## Configuration
- examples repo: `/tmp/ninetoothed-examples`
- device: `MLU`
- dtype: `bfloat16`
- compared against: `torch native`

## Results
| operator | case | NT ms | torch ms | slowdown |
| --- | --- | ---: | ---: | ---: |
| MM | square_1024 | 0.0309 | 0.0208 | 1.48x |
| MM | gpt2_mlp_up | 0.6021 | 0.2608 | 2.31x |
| MM | qwen_lm_head | 1.3516 | 0.5350 | 2.53x |
| AddMM | gpt2_fc_bias | 0.3025 | 0.1892 | 1.60x |
| AddMM | qwen_proj_bias | 0.3770 | 0.2815 | 1.34x |
| RMSNorm | qwen_decode | 0.4368 | 0.0889 | 4.91x |
| RMSNorm | qwen_prefill | 2.8942 | 0.3526 | 8.21x |
| RMSNorm | llama_decode | 0.4818 | 0.1965 | 2.45x |
| FusedRMSNorm | qwen_decode | 0.5459 | 0.1206 | 4.53x |
| FusedRMSNorm | qwen_prefill | 3.6531 | 0.3736 | 9.78x |
| FusedRMSNorm | llama_decode | 0.5859 | 0.2486 | 2.36x |
| SiLU | gpt2_mlp | 10.2759 | 0.3857 | 26.64x |
| SiLU | qwen_mlp | 14.3878 | 0.4458 | 32.27x |
| SwiGLU | gpt2_like | 11.0535 | 0.5277 | 20.95x |
| SwiGLU | qwen_like | 14.5357 | 0.6831 | 21.28x |
| Softmax | sampling_small_vocab | FAIL | FAIL | CompilationError: at 11:16: |
| Softmax | sampling_large_vocab | FAIL | FAIL | CompilationError: at 11:16: |
| Softmax | attn_scores | FAIL | FAIL | CompilationError: at 11:16: |
