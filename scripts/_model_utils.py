from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path


MODELS_ROOT = Path("/data02/jiangqiu/models")
DOWN_MODELS = MODELS_ROOT / "down_models.sh"
DEFAULT_PROMPT = "In one short sentence, explain what the moon is."
DEFAULT_LONG_PROMPT = (
    "You are validating an attention backend. "
    "Write a clear, technical response in three short paragraphs. "
    "First, explain why prefill and decode are different execution phases in a "
    "decoder-only transformer. "
    "Second, describe what can go wrong if the prompt length is large enough to "
    "exercise the prefill path but the implementation still assumes decode-only "
    "shapes. "
    "Third, summarize how you would confirm that both prefill and decode are "
    "actually being exercised during a runtime smoke test. "
    "Be specific about tensor shapes, cache metadata, and routing signals. "
    "Do not use bullet points."
)
DEFAULT_BATCH_SIZES = {
    "gpt2": 256,
    "Llama-2-7b-chat-hf": 128,
    "Qwen2.5-7B-Instruct": 128,
    "Mistral-7B-Instruct-v0.3": 128,
    "DeepSeek-R1-Distill-Qwen-7B": 96,
    "MiniCPM4.1-8B": 64,
    "glm-4-9b": 64,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    source_model: str


def parse_models() -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for line in DOWN_MODELS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line)
        if "--local_dir" not in parts or "--model" not in parts:
            continue
        local_dir = parts[parts.index("--local_dir") + 1]
        source_model = parts[parts.index("--model") + 1]
        model_path = MODELS_ROOT / local_dir
        if model_path.exists():
            models.append(ModelSpec(local_dir, model_path, source_model))
    return models


def select_models(selected_names: list[str] | None) -> list[ModelSpec]:
    models = parse_models()
    if not selected_names:
        return models
    selected = set(selected_names)
    return [model for model in models if model.name in selected]
