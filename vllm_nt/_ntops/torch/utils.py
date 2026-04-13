import functools
import os

import ninetoothed


_MAX_NUM_CONFIGS_ENV = "VLLM_NT_MAX_NUM_CONFIGS"
_MAX_NUM_CONFIGS_MODE_ENV = "VLLM_NT_MAX_NUM_CONFIGS_MODE"
_MAX_NUM_CONFIGS_MODES = {
    "quick": 2,
    "tuning": 10,
}
_MAX_NUM_CONFIGS_MODE_ALIASES = {
    "quick": "quick",
    "fast": "quick",
    "tune": "tuning",
    "tuning": "tuning",
}


class _CachedMakeDefaultConfig:
    def __init__(self, num_warps=None, num_stages=None, max_num_configs=None):
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.max_num_configs = max_num_configs


_cached_make_default_config = _CachedMakeDefaultConfig(max_num_configs=None)


def _parse_positive_int(value, name):
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def _max_num_configs_from_env():
    value = os.environ.get(_MAX_NUM_CONFIGS_ENV)
    if value is not None:
        return _parse_positive_int(value, _MAX_NUM_CONFIGS_ENV)

    mode = os.environ.get(_MAX_NUM_CONFIGS_MODE_ENV, "quick").strip().lower()
    mode = _MAX_NUM_CONFIGS_MODE_ALIASES.get(mode, mode)
    if mode not in _MAX_NUM_CONFIGS_MODES:
        supported = ", ".join(sorted(_MAX_NUM_CONFIGS_MODES))
        raise ValueError(
            f"{_MAX_NUM_CONFIGS_MODE_ENV} must be one of {supported}, got {mode!r}"
        )
    return _MAX_NUM_CONFIGS_MODES[mode]


def get_default_num_warps():
    return _cached_make_default_config.num_warps


def set_default_num_warps(num_warps):
    _cached_make_default_config.num_warps = num_warps
    clear_make_cache()


def get_default_num_stages():
    return _cached_make_default_config.num_stages


def set_default_num_stages(num_stages):
    _cached_make_default_config.num_stages = num_stages
    clear_make_cache()


def get_default_max_num_configs():
    if _cached_make_default_config.max_num_configs is not None:
        return _cached_make_default_config.max_num_configs
    return _max_num_configs_from_env()


def set_default_max_num_configs(max_num_configs):
    if max_num_configs is None:
        _cached_make_default_config.max_num_configs = None
    else:
        _cached_make_default_config.max_num_configs = _parse_positive_int(
            max_num_configs, "max_num_configs"
        )
    clear_make_cache()


def set_default_max_num_configs_mode(mode):
    mode = _MAX_NUM_CONFIGS_MODE_ALIASES.get(mode.strip().lower(), mode.strip().lower())
    if mode not in _MAX_NUM_CONFIGS_MODES:
        supported = ", ".join(sorted(_MAX_NUM_CONFIGS_MODES))
        raise ValueError(f"mode must be one of {supported}, got {mode!r}")
    set_default_max_num_configs(_MAX_NUM_CONFIGS_MODES[mode])


def _cached_make(
    premake, *args, num_warps=None, num_stages=None, max_num_configs=None, **keywords
):
    if num_warps is None:
        num_warps = _cached_make_default_config.num_warps

    if num_stages is None:
        num_stages = _cached_make_default_config.num_stages

    if max_num_configs is None:
        max_num_configs = get_default_max_num_configs()

    return _cached_make_impl(
        premake,
        *args,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
        **keywords,
    )


@functools.cache
def _cached_make_impl(
    premake, *args, num_warps=None, num_stages=None, max_num_configs=None, **keywords
):
    return ninetoothed.make(
        *premake(*args, **keywords),
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )


def clear_make_cache():
    _cached_make_impl.cache_clear()
