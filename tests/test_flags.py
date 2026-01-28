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


def test_get_stage1_pipeline_python(monkeypatch):
    monkeypatch.setenv("S2AFF_PIPELINE", "python")
    monkeypatch.setattr(flags, "_rust_available", lambda: True)

    assert flags.get_stage1_pipeline() == "python"


def test_get_stage1_pipeline_rust_unavailable_warns(monkeypatch):
    monkeypatch.setenv("S2AFF_PIPELINE", "rust")
    monkeypatch.setattr(flags, "_rust_available", lambda: False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert flags.get_stage1_pipeline() == "python"

    assert any("S2AFF_PIPELINE=rust requested" in str(w.message) for w in captured)


def test_get_stage2_pipeline_python(monkeypatch):
    monkeypatch.setenv("S2AFF_PIPELINE", "python")
    monkeypatch.setattr(flags, "_rust_available", lambda: True)

    assert flags.get_stage2_pipeline() == "python"


def test_get_stage2_pipeline_rust_unavailable_warns(monkeypatch):
    monkeypatch.setenv("S2AFF_PIPELINE", "rust")
    monkeypatch.setattr(flags, "_rust_available", lambda: False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert flags.get_stage2_pipeline() == "python"

    assert any("S2AFF_PIPELINE=rust requested" in str(w.message) for w in captured)
