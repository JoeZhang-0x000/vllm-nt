"""Package metadata tests for vLLM CLI plugin integration."""

from pathlib import Path


def test_vllm_general_plugin_entry_point_is_declared():
    pyproject = (
        Path(__file__).resolve().parent.parent / "pyproject.toml"
    ).read_text(encoding="utf-8")

    assert '[project.entry-points."vllm.general_plugins"]' in pyproject
    assert 'vllm_nt = "vllm_nt:register"' in pyproject
