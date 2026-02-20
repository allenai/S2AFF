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


def get_stage1_pipeline(default=None):
    pipeline = _normalize_pipeline(os.getenv("S2AFF_PIPELINE"))
    rust_ok = _rust_available()
    if pipeline == "python":
        return "python"
    if pipeline == "rust":
        if not rust_ok:
            warnings.warn(
                "S2AFF_PIPELINE=rust requested but Rust backend unavailable; using Python pipeline.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "python"
        return "rust"
    if default is None:
        default = "rust" if rust_ok else "python"
    if default == "rust" and not rust_ok:
        return "python"
    return default


def get_stage2_pipeline(default=None):
    pipeline = _normalize_pipeline(os.getenv("S2AFF_PIPELINE"))
    rust_ok = _rust_available()
    if pipeline == "python":
        return "python"
    if pipeline == "rust":
        if not rust_ok:
            warnings.warn(
                "S2AFF_PIPELINE=rust requested but Rust backend unavailable; using Python pipeline.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "python"
        return "rust"
    if default is None:
        default = "rust" if rust_ok else "python"
    if default == "rust" and not rust_ok:
        return "python"
    return default
