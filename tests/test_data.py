from s2aff.data import load_gold_affiliation_annotations
import s2aff.file_cache as file_cache


def _make_sample_csv(path):
    content = (
        "original_affiliation,labels\n"
        "\"Aff 1\",\"['https://ror.org/123456']\"\n"
        "\"Aff 2\",\"['GRID:1234']\"\n"
    )
    path.write_text(content)
    return path


def test_load_gold_affiliation_annotations_filters_non_ror(tmp_path, monkeypatch):
    csv_path = _make_sample_csv(tmp_path / "gold.csv")
    monkeypatch.setattr(file_cache, "cached_path", lambda value: str(csv_path))

    df = load_gold_affiliation_annotations(gold_path=str(csv_path))

    assert len(df) == 1
    assert df.iloc[0]["original_affiliation"] == "Aff 1"
    assert all("ror.org" in label for label in df.iloc[0]["labels"])


def test_load_gold_affiliation_annotations_keep_non_ror(tmp_path, monkeypatch):
    csv_path = _make_sample_csv(tmp_path / "gold.csv")
    monkeypatch.setattr(file_cache, "cached_path", lambda value: str(csv_path))

    df = load_gold_affiliation_annotations(gold_path=str(csv_path), keep_non_ror_gold=True)

    assert len(df) == 2
    assert isinstance(df.iloc[0]["labels"], list)
