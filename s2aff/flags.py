import os
import warnings
from typing import Optional


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


def _rust_available() -> bool:
    try:
        from s2aff.rust_backend import rust_available
    except Exception:
        return False
    try:
        return rust_available()
    except Exception:
        return False


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
    if default == "v7" and not rust_ok:
        return "v1"
    return default


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
    if default == "v3" and not rust_ok:
        return "v1"
    return default
