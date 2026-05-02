from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

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
    assert config.config_for("RoPE").backend == "ninetoothed"
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


def test_config_accepts_infinicore_flash_attn_backend(tmp_path, monkeypatch):
    cfg_path = tmp_path / "backend.yaml"
    cfg_path.write_text(
        """
version: 1
ops:
  PagedAttentionPrefill:
    backend: infinicore-flash-attn
  PagedAttentionDecode:
    backend: infinicore-flash-attn
""",
        encoding="utf-8",
    )
    config = _reload_config(monkeypatch, cfg_path)

    assert config.config_for("PagedAttentionPrefill").backend == "infinicore-flash-attn"
    assert config.config_for("PagedAttentionDecode").backend == "infinicore-flash-attn"


def test_all_infinicore_nt_fa2_config_only_switches_attention(monkeypatch):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "vllm_nt"
        / "configs"
        / "all-infinicore-nt-fa2.yaml"
    )
    config = _reload_config(monkeypatch, config_path)

    assert config.config_for("RMSNorm").backend == "infinicore"
    assert config.config_for("MatMul").backend == "infinicore"
    assert config.config_for("SiluAndMul").backend == "ninetoothed"
    assert config.config_for("StoreKVCache").backend == "infinicore"
    assert config.config_for("PagedAttentionPrefill").backend == "infinicore-flash-attn"
    assert config.config_for("PagedAttentionDecode").backend == "infinicore-flash-attn"


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
    monkeypatch.setenv("VLLM_NT_ENABLE_STATS", "1")

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


def test_router_dispatches_infinicore_flash_attn_backend(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    cfg_path = tmp_path / "backend.yaml"
    cfg_path.write_text(
        """
version: 1
ops:
  PagedAttentionPrefill:
    backend: infinicore-flash-attn
    fallback_backend: original
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_NT_BACKEND_CONFIG", str(cfg_path))
    monkeypatch.setenv("VLLM_NT_ENABLE_STATS", "1")

    import vllm_nt._ntops.config as config
    import vllm_nt._ntops.backends as backends

    config.reset_backend_config_cache()
    backends.reset_backend_state()

    result = backends.route(
        "PagedAttentionPrefill",
        lambda: "original",
        call_infinicore=lambda: "paged",
        call_infinicore_flash_attn=lambda: "flash",
    )

    assert result == "flash"
    stats = backends.backend_stats("PagedAttentionPrefill")
    assert stats["infinicore-flash-attn"].attempts == 1
    assert stats["infinicore-flash-attn"].hits == 1


def test_router_stats_disabled_by_default_but_fallback_still_disables_backend(
    tmp_path, monkeypatch
):
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
    monkeypatch.delenv("VLLM_NT_ENABLE_STATS", raising=False)

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
    assert backends.backend_stats("MatMul") == {}


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


def test_rope_uses_apply_rotary_emb_backend_alias(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    cfg_path = tmp_path / "backend.yaml"
    cfg_path.write_text(
        """
version: 1
defaults:
  backend: original
  fallback_backend: original
ops:
  ApplyRotaryEmb:
    backend: ninetoothed
    fallback_backend: original
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_NT_BACKEND_CONFIG", str(cfg_path))

    import vllm_nt._ntops.config as config
    import vllm_nt._ntops.backends as backends

    config.reset_backend_config_cache()
    backends.reset_backend_state()

    assert backends.configured_backend("RoPE") == "ninetoothed"
    assert backends.active_backend("RoPE") == "ninetoothed"
    assert backends.fallback_backend("RoPE") == "original"


@pytest.mark.parametrize("disabled_name", ["RoPE", "ApplyRotaryEmb"])
def test_disable_ops_accepts_rope_aliases(tmp_path, monkeypatch, disabled_name):
    pytest.importorskip("torch")
    cfg_path = tmp_path / "backend.yaml"
    cfg_path.write_text(
        """
version: 1
ops:
  ApplyRotaryEmb:
    backend: ninetoothed
    fallback_backend: original
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_NT_BACKEND_CONFIG", str(cfg_path))
    monkeypatch.setenv("VLLM_NT_DISABLE_OPS", disabled_name)

    import vllm_nt._ntops.config as config
    import vllm_nt._ntops.backends as backends

    config.reset_backend_config_cache()
    backends.reset_backend_state()

    assert backends.active_backend("RoPE") == "original"
    assert "RoPE" in backends.disabled_ops()
    assert "ApplyRotaryEmb" in backends.disabled_ops()


def test_explicit_config_path_missing_raises(tmp_path, monkeypatch):
    missing = tmp_path / "nonexistent.yaml"
    monkeypatch.setenv("VLLM_NT_BACKEND_CONFIG", str(missing))
    import vllm_nt._ntops.config as config

    config.reset_backend_config_cache()

    with pytest.raises(RuntimeError, match="backend config file not found"):
        config.load_backend_config()


def test_flash_attn_prefill_adapter_calls_mha_varlen(monkeypatch):
    torch = pytest.importorskip("torch")
    import vllm_nt._ntops.backends as backends

    calls = []

    def mha_varlen(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "infinicore", SimpleNamespace(mha_varlen=mha_varlen))
    monkeypatch.setattr(backends, "as_infini", lambda tensor: tensor)
    monkeypatch.setattr(backends, "as_infini_strided", lambda tensor: tensor)

    class DummyImpl:
        alibi_slopes = None
        scale = 0.125

    class DummyMetadata:
        num_decode_tokens = 1
        num_actual_tokens = 4
        prefill_query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
        prefill_block_table = torch.tensor([[0, 1]], dtype=torch.int32)
        cu_prefix_kv_lens = torch.tensor([0, 3], dtype=torch.int32)
        prefill_max_seq_len = 3

    query = torch.zeros((4, 2, 8))
    key = torch.zeros((4, 1, 8))
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 5, 16, 1, 8))

    backends.flash_attn_prefill_infinicore(
        DummyImpl(), query, key, kv_cache, DummyMetadata(), output
    )

    args, kwargs = calls[0]
    assert args[0].shape == (3, 2, 8)
    assert args[1].shape == (5, 16, 1, 8)
    assert torch.equal(args[3], DummyMetadata.prefill_query_start_loc)
    assert torch.equal(args[4], torch.tensor([0, 3], dtype=torch.int32))
    assert args[6] == 3
    assert args[7] == 3
    assert args[9] == 0.125
    assert kwargs["out"].shape == (3, 2, 8)


def test_flash_attn_prefill_adapter_uses_total_kv_len_for_max_k(monkeypatch):
    torch = pytest.importorskip("torch")
    import vllm_nt._ntops.backends as backends

    calls = []

    def mha_varlen(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "infinicore", SimpleNamespace(mha_varlen=mha_varlen))
    monkeypatch.setattr(backends, "as_infini", lambda tensor: tensor)
    monkeypatch.setattr(backends, "as_infini_strided", lambda tensor: tensor)

    class DummyImpl:
        alibi_slopes = None
        scale = 0.125

    class DummyMetadata:
        num_decode_tokens = 0
        num_actual_tokens = 5
        prefill_query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
        prefill_block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        cu_prefix_kv_lens = torch.tensor([0, 6, 10], dtype=torch.int32)

    query = torch.zeros((5, 2, 8))
    key = torch.zeros((5, 1, 8))
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 5, 16, 1, 8))

    backends.flash_attn_prefill_infinicore(
        DummyImpl(), query, key, kv_cache, DummyMetadata(), output
    )

    args, _ = calls[0]
    assert args[6] == 3
    assert args[7] == 6


def test_flash_attn_decode_adapter_calls_mha_kvcache(monkeypatch):
    torch = pytest.importorskip("torch")
    import vllm_nt._ntops.backends as backends

    calls = []

    def mha_kvcache(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "infinicore", SimpleNamespace(mha_kvcache=mha_kvcache))
    monkeypatch.setattr(backends, "as_infini", lambda tensor: tensor)
    monkeypatch.setattr(backends, "as_infini_strided", lambda tensor: tensor)

    class DummyImpl:
        alibi_slopes = None
        scale = 0.25

    class DummyMetadata:
        num_decode_tokens = 2
        num_decodes = 2
        decode_seq_lens = torch.tensor([4, 5], dtype=torch.int32)
        decode_block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    query = torch.zeros((2, 4, 8))
    key = torch.zeros((2, 1, 8))
    output = torch.zeros_like(query)
    kv_cache = torch.zeros((2, 5, 1, 16, 8))

    backends.flash_attn_decode_infinicore(
        DummyImpl(), query, key, kv_cache, DummyMetadata(), output
    )

    args, kwargs = calls[0]
    assert args[0].shape == (2, 1, 4, 8)
    assert args[1].shape == (5, 16, 1, 8)
    assert torch.equal(args[3], DummyMetadata.decode_seq_lens)
    assert torch.equal(args[4], DummyMetadata.decode_block_table)
    assert args[6] == 0.25
    assert kwargs["out"].shape == (2, 1, 4, 8)
