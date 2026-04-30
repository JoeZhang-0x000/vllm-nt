from __future__ import annotations

import importlib

import pytest


def _reload_config(monkeypatch, path):
    monkeypatch.setenv("VLLM_NT_BACKEND_CONFIG", str(path))
    import vllm_nt._ntops.config as config

    config.reset_backend_config_cache()
    return config


def test_default_config_loads_hybrid_default(monkeypatch):
    monkeypatch.delenv("VLLM_NT_BACKEND_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_INFINI_PATCH_CONFIG", raising=False)
    import vllm_nt._ntops.config as config

    config.reset_backend_config_cache()

    cfg = config.load_backend_config()

    assert cfg.path is not None
    assert cfg.path.endswith("vllm_nt/configs/hybrid-default.yaml")
    assert config.config_for("RMSNorm").backend == "infinicore"
    assert config.config_for("GemmaRMSNorm").backend == "infinicore"
    assert config.config_for("SiluAndMul").backend == "ninetoothed"
    assert config.config_for("ApplyRotaryEmb").backend == "ninetoothed"
    assert config.config_for("Embedding").backend == "ninetoothed"
    assert config.config_for("TopKTopP").backend == "ninetoothed"
    assert config.config_for("RandomSample").backend == "ninetoothed"
    assert config.config_for("MatMul").backend == "original"
    assert config.config_for("LMHead").backend == "original"
    assert config.config_for("StoreKVCache").backend == "original"
    assert config.config_for("PagedAttentionPrefill").backend == "original"
    assert config.config_for("PagedAttentionDecode").backend == "original"
    assert config.config_for("SDPA").backend == "original"


def test_config_loads_defaults_and_string_ops(tmp_path, monkeypatch):
    cfg_path = tmp_path / "backend.yaml"
    cfg_path.write_text(
        """
version: 1
defaults:
  backend: original
  fallback_backend: metax
ops:
  RMSNorm: infinicore
""",
        encoding="utf-8",
    )
    config = _reload_config(monkeypatch, cfg_path)

    cfg = config.load_backend_config()

    assert cfg.default_backend == "original"
    assert cfg.default_fallback_backend == "original"
    assert config.config_for("RMSNorm").backend == "infinicore"
    assert config.config_for("UnknownOp").backend == "original"


def test_config_rejects_unknown_backend(tmp_path, monkeypatch):
    cfg_path = tmp_path / "backend.yaml"
    cfg_path.write_text("version: 1\nops:\n  RMSNorm: cuda\n", encoding="utf-8")
    config = _reload_config(monkeypatch, cfg_path)

    with pytest.raises(RuntimeError, match="unsupported backend"):
        config.load_backend_config()


def test_legacy_config_env_is_supported(tmp_path, monkeypatch):
    cfg_path = tmp_path / "legacy.yaml"
    cfg_path.write_text("version: 1\nops:\n  MatMul: ninetoothed\n", encoding="utf-8")
    monkeypatch.delenv("VLLM_NT_BACKEND_CONFIG", raising=False)
    monkeypatch.setenv("VLLM_INFINI_PATCH_CONFIG", str(cfg_path))
    import vllm_nt._ntops.config as config

    config.reset_backend_config_cache()

    assert config.config_for("MatMul").backend == "ninetoothed"


def test_router_falls_back_and_disables_first_failure(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    cfg_path = tmp_path / "backend.yaml"
    cfg_path.write_text(
        """
version: 1
defaults:
  backend: original
  fallback_backend: original
ops:
  MatMul:
    backend: infinicore
    fallback_backend: ninetoothed
    disable_backend_on_first_failure: true
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_NT_BACKEND_CONFIG", str(cfg_path))

    import vllm_nt._ntops.config as config
    import vllm_nt._ntops.backends as backends

    config.reset_backend_config_cache()
    backends.reset_backend_state()

    result = backends.route(
        "MatMul",
        lambda: "original",
        call_infinicore=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        call_ninetoothed=lambda: "nt",
    )

    assert result == "nt"
    assert backends.active_backend("MatMul") == "ninetoothed"
    stats = backends.backend_stats("MatMul")
    assert stats["infinicore"].attempts == 1
    assert stats["infinicore"].failures == 1


def test_disable_ops_forces_fallback(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    cfg_path = tmp_path / "backend.yaml"
    cfg_path.write_text(
        """
version: 1
ops:
  Embedding:
    backend: ninetoothed
    fallback_backend: original
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_NT_BACKEND_CONFIG", str(cfg_path))
    monkeypatch.setenv("VLLM_NT_DISABLE_OPS", "Embedding")

    import vllm_nt._ntops.config as config
    import vllm_nt._ntops.backends as backends

    config.reset_backend_config_cache()
    backends.reset_backend_state()
    importlib.reload(backends)

    assert backends.active_backend("Embedding") == "original"
