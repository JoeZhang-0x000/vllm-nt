from __future__ import annotations

from types import SimpleNamespace

from scripts.run_inference_validation import _resolve_prompt


def test_resolve_prompt_prefers_explicit_prompt():
    args = SimpleNamespace(prompt="custom", prompt_profile="long")

    assert _resolve_prompt(args) == "custom"


def test_resolve_prompt_uses_profile_when_prompt_missing():
    args = SimpleNamespace(prompt=None, prompt_profile="long")

    assert _resolve_prompt(args).startswith("You are validating an attention backend.")
