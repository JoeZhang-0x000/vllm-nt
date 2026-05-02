from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_VALID_BACKENDS = {
    "original",
    "metax",
    "ninetoothed",
    "infinicore",
    "infinicore-flash-attn",
}
_CONFIG_ENV = "VLLM_NT_BACKEND_CONFIG"
_LEGACY_CONFIG_ENV = "VLLM_INFINI_PATCH_CONFIG"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hybrid-default.yaml"
_OP_ALIASES = {
    "RoPE": "ApplyRotaryEmb",
}
_CONFIG: "BackendConfig | None" = None


@dataclass(frozen=True)
class OpConfig:
    backend: str
    fallback_backend: str
    disable_backend_on_first_failure: bool


@dataclass(frozen=True)
class BackendConfig:
    version: int
    default_backend: str
    default_fallback_backend: str
    default_disable_backend_on_first_failure: bool
    ops: dict[str, OpConfig]
    path: str | None = None


def _canonical_backend(value: str) -> str:
    value = value.strip().lower()
    return "original" if value == "metax" else value


def _validate_backend(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{field} must be a string backend name")
    raw = value.strip().lower()
    if raw not in _VALID_BACKENDS:
        raise RuntimeError(
            f"{field} has unsupported backend {raw!r}; expected one of {sorted(_VALID_BACKENDS)}"
        )
    return _canonical_backend(raw)


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        data = _load_simple_yaml(text)
    if not isinstance(data, dict):
        raise RuntimeError(f"backend config {path} must contain a YAML mapping")
    return data


def _load_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise RuntimeError("simple backend YAML parser only supports mappings")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return result


def _parse_scalar(value: str) -> object:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _config_path() -> tuple[Path, bool]:
    env_val = os.environ.get(_CONFIG_ENV) or os.environ.get(_LEGACY_CONFIG_ENV)
    if env_val:
        return Path(env_val), True
    return _DEFAULT_CONFIG_PATH, False


def load_backend_config() -> BackendConfig:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    path, from_env = _config_path()
    if from_env and not path.exists():
        raise RuntimeError(
            f"backend config file not found: {path} "
            f"(set via {'VLLM_NT_BACKEND_CONFIG' if os.environ.get(_CONFIG_ENV) else 'VLLM_INFINI_PATCH_CONFIG'})"
        )
    raw = _load_yaml(path) if path.exists() else {}

    version = int(raw.get("version", 1))
    if version != 1:
        raise RuntimeError(f"unsupported backend config version {version}; expected 1")

    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise RuntimeError("backend config defaults must be a mapping")

    default_backend = _validate_backend(
        defaults.get("backend", "original"), field="defaults.backend"
    )
    default_fallback = _validate_backend(
        defaults.get("fallback_backend", "original"),
        field="defaults.fallback_backend",
    )
    default_disable = bool(defaults.get("disable_backend_on_first_failure", True))

    raw_ops = raw.get("ops") or {}
    if not isinstance(raw_ops, dict):
        raise RuntimeError("backend config ops must be a mapping")

    ops: dict[str, OpConfig] = {}
    for op_name, op_raw in raw_ops.items():
        if not isinstance(op_name, str):
            raise RuntimeError("backend config op names must be strings")
        if op_raw is None:
            op_raw = {}
        if isinstance(op_raw, str):
            op_raw = {"backend": op_raw}
        if not isinstance(op_raw, dict):
            raise RuntimeError(f"backend config for {op_name} must be a mapping or backend string")
        ops[op_name] = OpConfig(
            backend=_validate_backend(
                op_raw.get("backend", default_backend), field=f"ops.{op_name}.backend"
            ),
            fallback_backend=_validate_backend(
                op_raw.get("fallback_backend", default_fallback),
                field=f"ops.{op_name}.fallback_backend",
            ),
            disable_backend_on_first_failure=bool(
                op_raw.get("disable_backend_on_first_failure", default_disable)
            ),
        )

    _CONFIG = BackendConfig(
        version=version,
        default_backend=default_backend,
        default_fallback_backend=default_fallback,
        default_disable_backend_on_first_failure=default_disable,
        ops=ops,
        path=str(path) if path.exists() else None,
    )
    return _CONFIG


def canonical_op_name(op_name: str) -> str:
    return _OP_ALIASES.get(op_name, op_name)


def op_name_variants(op_name: str) -> set[str]:
    canonical = canonical_op_name(op_name)
    aliases = {alias for alias, target in _OP_ALIASES.items() if target == canonical}
    return {canonical, *aliases}


def config_for(op_name: str) -> OpConfig:
    config = load_backend_config()
    op_name = canonical_op_name(op_name)
    return config.ops.get(
        op_name,
        OpConfig(
            backend=config.default_backend,
            fallback_backend=config.default_fallback_backend,
            disable_backend_on_first_failure=config.default_disable_backend_on_first_failure,
        ),
    )


def reset_backend_config_cache() -> None:
    global _CONFIG
    _CONFIG = None
