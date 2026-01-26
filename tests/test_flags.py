import warnings

import s2aff.flags as flags


def test_normalize_pipeline_variants():
    assert flags._normalize_pipeline(None) is None
    assert flags._normalize_pipeline("") is None
    assert flags._normalize_pipeline("AUTO") is None
    assert flags._normalize_pipeline("python") == "python"
    assert flags._normalize_pipeline("Py") == "python"
    assert flags._normalize_pipeline("legacy") == "python"
    assert flags._normalize_pipeline("rust") == "rust"
    assert flags._normalize_pipeline("RS") == "rust"
    assert flags._normalize_pipeline("fast") == "rust"
    assert flags._normalize_pipeline("unknown") is None


def test_normalize_variant_accepts_allowed_and_default():
    assert flags._normalize_variant(None, "v1", {"v1", "v7"}) == "v1"
    assert flags._normalize_variant("v7", "v1", {"v1", "v7"}) == "v7"
    assert flags._normalize_variant("V7", "v1", {"v1", "v7"}) == "v7"
    assert flags._normalize_variant("bad", "v1", {"v1", "v7"}) == "v1"


def test_get_stage1_variant_pipeline_python(monkeypatch):
    monkeypatch.setenv("S2AFF_PIPELINE", "python")
    monkeypatch.setattr(flags, "_rust_available", lambda: True)
    monkeypatch.setattr(flags, "_STAGE_ENV_WARNED", False)

    assert flags.get_stage1_variant() == "v1"


def test_get_stage1_variant_pipeline_rust_unavailable_warns(monkeypatch):
    monkeypatch.setenv("S2AFF_PIPELINE", "rust")
    monkeypatch.setattr(flags, "_rust_available", lambda: False)
    monkeypatch.setattr(flags, "_STAGE_ENV_WARNED", False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert flags.get_stage1_variant() == "v1"

    assert any("S2AFF_PIPELINE=rust requested" in str(w.message) for w in captured)


def test_get_stage2_variant_deprecated_env_warns_once(monkeypatch):
    monkeypatch.delenv("S2AFF_PIPELINE", raising=False)
    monkeypatch.setenv("S2AFF_STAGE2_VARIANT", "v3")
    monkeypatch.setattr(flags, "_rust_available", lambda: True)
    monkeypatch.setattr(flags, "_STAGE_ENV_WARNED", False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert flags.get_stage2_variant(default="v1") == "v3"

    assert any("deprecated" in str(w.message) for w in captured)

    with warnings.catch_warnings(record=True) as captured_again:
        warnings.simplefilter("always")
        assert flags.get_stage2_variant(default="v1") == "v3"

    assert not captured_again


def test_get_stage2_variant_falls_back_when_rust_unavailable(monkeypatch):
    monkeypatch.delenv("S2AFF_PIPELINE", raising=False)
    monkeypatch.setenv("S2AFF_STAGE2_VARIANT", "v3")
    monkeypatch.setattr(flags, "_rust_available", lambda: False)
    monkeypatch.setattr(flags, "_STAGE_ENV_WARNED", False)

    assert flags.get_stage2_variant(default="v3") == "v1"
