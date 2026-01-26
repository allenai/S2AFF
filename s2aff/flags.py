import os
import warnings
from typing import Optional


_STAGE_ENV_WARNED = False


def _normalize_pipeline(value: Optional[str]):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"", "auto", "default"}:
        return None
    if value in {"python", "py", "legacy"}:
        return "python"
    if value in {"rust", "rs", "fast"}:
        return "rust"
    return None


def _warn_stage_env():
    global _STAGE_ENV_WARNED
    if _STAGE_ENV_WARNED:
        return
    warnings.warn(
        "S2AFF_STAGE1_VARIANT/S2AFF_STAGE2_VARIANT are deprecated. "
        "Use S2AFF_PIPELINE=python|rust instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _STAGE_ENV_WARNED = True


def _rust_available() -> bool:
    try:
        from s2aff.rust_backend import rust_available
    except Exception:
        return False
    try:
        return rust_available()
    except Exception:
        return False


def _normalize_variant(value, default, allowed):
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in allowed:
        return value
    return default


def get_stage1_variant(default=None):
    pipeline = _normalize_pipeline(os.getenv("S2AFF_PIPELINE"))
    rust_ok = _rust_available()
    if pipeline == "python":
        return "v1"
    if pipeline == "rust":
        if not rust_ok:
            warnings.warn(
                "S2AFF_PIPELINE=rust requested but Rust backend unavailable; using Python pipeline.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "v1"
        return "v7"
    if default is None:
        default = "v7" if rust_ok else "v1"
    stage_env = os.getenv("S2AFF_STAGE1_VARIANT")
    if stage_env is not None and str(stage_env).strip() != "":
        _warn_stage_env()
    value = _normalize_variant(stage_env, default, {"v1", "v7"})
    if value == "v7" and not rust_ok:
        return "v1"
    return value


def get_stage2_variant(default=None):
    pipeline = _normalize_pipeline(os.getenv("S2AFF_PIPELINE"))
    rust_ok = _rust_available()
    if pipeline == "python":
        return "v1"
    if pipeline == "rust":
        if not rust_ok:
            warnings.warn(
                "S2AFF_PIPELINE=rust requested but Rust backend unavailable; using Python pipeline.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "v1"
        return "v3"
    if default is None:
        default = "v3" if rust_ok else "v1"
    stage_env = os.getenv("S2AFF_STAGE2_VARIANT")
    if stage_env is not None and str(stage_env).strip() != "":
        _warn_stage_env()
    value = _normalize_variant(stage_env, default, {"v1", "v3"})
    if value == "v3" and not rust_ok:
        return "v1"
    return value
