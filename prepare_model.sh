#!/usr/bin/env bash
set -euo pipefail

modelscope download --model shakechen/Llama-2-7b-chat-hf --local_dir Llama-2-7b-chat-hf
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir Qwen2.5-7B-Instruct
modelscope download --model OpenBMB/MiniCPM4.1-8B --local_dir MiniCPM4.1-8B
modelscope download --model ZhipuAI/glm-4-9b --local_dir glm-4-9b
modelscope download --model LLM-Research/Mistral-7B-Instruct-v0.3 --local_dir Mistral-7B-Instruct-v0.3
modelscope download --model openai-community/gpt2 --local_dir gpt2
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local_dir DeepSeek-R1-Distill-Qwen-7B
