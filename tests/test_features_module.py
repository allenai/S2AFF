import numpy as np
import pytest

from s2aff.features import (
    FEATURE_NAMES,
    find_query_ngrams_in_text,
    make_feature_names_and_constraints,
    make_lightgbm_features,
    parse_ror_entry_into_single_string_lightgbm,
)


class DummyIndex:
    def __init__(self):
        self.ror_dict = {
            "https://ror.org/123456": {
                "name": "Example University",
                "aliases": ["Example U"],
                "labels": [{"label": "Ex Univ"}],
                "acronyms": ["EU"],
                "addresses": [
                    {
                        "city": "Metropolis",
                        "state": "Example State",
                        "state_code": "EX-ES",
                        "country_geonames_id": "42",
                        "country_code": "EX",
                    }
                ],
                "works_count": 25,
            }
        }
        self.country_codes_dict = {"42": ["Exampleland", "EX", "EXA", "EXF"]}
        self.country_by_iso2 = {"US": ["United States", "US", "USA", "US"]}


class DummyLM:
    def __init__(self):
        self.lookup = {
            "qwertyuiop": -100.0,
            "example": -1.0,
            "university": -1.5,
            "example university": -0.5,
            "metropolis": -2.0,
        }

    def score(self, text, eos=False, bos=False):
        return self.lookup.get(text.lower(), -10.0)


def test_make_feature_names_and_constraints_alignment():
    feats, constraints = make_feature_names_and_constraints()
    constraint_tokens = constraints.split(",")

    assert len(feats) == len(constraint_tokens)
    assert feats[0] == "works_count"
    assert constraint_tokens[0] == "1"
    for field in ["names", "acronyms", "city", "state", "country"]:
        idx = list(feats).index(f"{field}_frac_of_query_matched_in_text")
        assert constraint_tokens[idx] == "1"


def test_find_query_ngrams_in_text_respects_boundaries():
    spans, tokens = find_query_ngrams_in_text(
        "Example University",
        "The Example University Library",
    )
    assert spans and tokens == ["Example University"]

    no_spans, no_tokens = find_query_ngrams_in_text(
        "Uni",
        "Example University",
        use_word_boundaries=True,
    )
    assert no_spans == [] and no_tokens == []


def test_parse_ror_entry_into_single_string_lightgbm_handles_non_ror():
    index = DummyIndex()
    non_ror = parse_ror_entry_into_single_string_lightgbm("Not a ROR", index)
    assert non_ror["names"] == ["Not a ROR"]
    assert non_ror["country"] == []


def test_parse_ror_entry_into_single_string_lightgbm_deduplicates_country():
    index = DummyIndex()
    index.ror_dict["https://ror.org/123456"]["addresses"][0]["country_geonames_id"] = "1"
    index.country_codes_dict["1"] = ["China", "CN", "CHN", "CH"]
    entry = parse_ror_entry_into_single_string_lightgbm("https://ror.org/123456", index)
    assert "pr" in entry["country"] and "prc" in entry["country"]
    assert entry["works_count"] == 25


def test_parse_ror_entry_into_single_string_lightgbm_falls_back_to_iso2():
    index = DummyIndex()
    addr = index.ror_dict["https://ror.org/123456"]["addresses"][0]
    addr["country_geonames_id"] = None
    addr["country_code"] = "US"

    entry = parse_ror_entry_into_single_string_lightgbm("https://ror.org/123456", index)
    assert "united states" in entry["country"]


def test_parse_ror_entry_into_single_string_lightgbm_handles_missing_country_info():
    index = DummyIndex()
    addr = index.ror_dict["https://ror.org/123456"]["addresses"][0]
    addr["country_geonames_id"] = None
    addr["country_code"] = None

    entry = parse_ror_entry_into_single_string_lightgbm("https://ror.org/123456", index)
    assert entry["country"] == []


def test_make_lightgbm_features_matches_feature_template():
    index = DummyIndex()
    lm = DummyLM()
    entry = parse_ror_entry_into_single_string_lightgbm("https://ror.org/123456", index)
    features = make_lightgbm_features("example university", entry, lm)
    assert len(features) == len(FEATURE_NAMES)
    assert features[0] == np.round(np.log1p(25))
    assert features[-5] >= 1.0
    assert np.isnan(features[-1])


def test_make_lightgbm_features_returns_all_nans_for_empty_query():
    features = make_lightgbm_features("", {"works_count": 0}, DummyLM())
    assert len(features) == len(FEATURE_NAMES)
    assert all(np.isnan(features))


def test_find_query_ngrams_in_text_handles_empty_and_non_strings():
    spans, tokens = find_query_ngrams_in_text("", "anything here")
    assert spans == [] and tokens == []

    spans, tokens = find_query_ngrams_in_text("query", ["not a string"])
    assert spans == [] and tokens == []

    spans, tokens = find_query_ngrams_in_text([1, 2], "text")
    assert spans == [] and tokens == []
