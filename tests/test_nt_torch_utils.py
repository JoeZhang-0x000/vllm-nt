import pytest


def _reset_max_num_configs(utils, monkeypatch):
    monkeypatch.delenv("VLLM_NT_MAX_NUM_CONFIGS", raising=False)
    monkeypatch.delenv("VLLM_NT_MAX_NUM_CONFIGS_MODE", raising=False)
    utils.set_default_max_num_configs(None)


def test_default_max_num_configs_uses_quick_mode(monkeypatch):
    from vllm_nt._ntops.torch import utils

    _reset_max_num_configs(utils, monkeypatch)

    assert utils.get_default_max_num_configs() == 2


def test_default_max_num_configs_supports_tuning_mode(monkeypatch):
    from vllm_nt._ntops.torch import utils

    _reset_max_num_configs(utils, monkeypatch)
    monkeypatch.setenv("VLLM_NT_MAX_NUM_CONFIGS_MODE", "tuning")

    assert utils.get_default_max_num_configs() == 10


def test_max_num_configs_env_overrides_mode(monkeypatch):
    from vllm_nt._ntops.torch import utils

    _reset_max_num_configs(utils, monkeypatch)
    monkeypatch.setenv("VLLM_NT_MAX_NUM_CONFIGS_MODE", "tuning")
    monkeypatch.setenv("VLLM_NT_MAX_NUM_CONFIGS", "4")

    assert utils.get_default_max_num_configs() == 4


def test_set_default_max_num_configs_mode(monkeypatch):
    from vllm_nt._ntops.torch import utils

    _reset_max_num_configs(utils, monkeypatch)
    utils.set_default_max_num_configs_mode("tune")

    assert utils.get_default_max_num_configs() == 10


def test_invalid_max_num_configs_rejected(monkeypatch):
    from vllm_nt._ntops.torch import utils

    _reset_max_num_configs(utils, monkeypatch)

    with pytest.raises(ValueError):
        utils.set_default_max_num_configs(0)


def test_cached_make_uses_effective_max_num_configs(monkeypatch):
    from vllm_nt._ntops.torch import utils

    _reset_max_num_configs(utils, monkeypatch)
    seen_max_num_configs = []

    def fake_make(*args, **kwargs):
        seen_max_num_configs.append(kwargs["max_num_configs"])
        return object()

    def premake():
        return object(), object(), ()

    monkeypatch.setattr(utils.ninetoothed, "make", fake_make)

    monkeypatch.setenv("VLLM_NT_MAX_NUM_CONFIGS_MODE", "quick")
    utils._cached_make(premake)

    monkeypatch.setenv("VLLM_NT_MAX_NUM_CONFIGS_MODE", "tuning")
    utils._cached_make(premake)

    assert seen_max_num_configs == [2, 10]
