from collections import Counter

import numpy as np
import pytest

from s2aff.ror import (
    RORIndex,
    coerce_v2_to_v1like,
    get_special_tokens_dict,
    normalize_geoname_id,
    parse_ror_entry_into_single_string,
    sum_list_of_list_of_tuples,
)


def make_stub_index():
    index = RORIndex.__new__(RORIndex)
    index.grid_to_ror = {
        "grid.1234.5": "https://ror.org/ror-1",
        "grid.1234.56": "https://ror.org/ror-2",
    }
    index.isni_to_ror = {
        "0000000123456789": "https://ror.org/ror-3",
        "0000111122223333": "https://ror.org/ror-4",
    }
    index.ror_dict = {
        "https://ror.org/123456": {
            "name": "Example University",
            "aliases": [],
            "labels": [],
            "acronyms": [],
            "addresses": [
                {
                    "city": "City",
                    "state": "",
                    "state_code": None,
                    "country_geonames_id": None,
                    "country_code": "US",
                }
            ],
            "wikipedia_page": ["https://en.wikipedia.org/wiki/Example"],
        }
    }
    index.country_codes_dict = {"123": ["Exampleland", "EX", "EXA", "EXF"]}
    index.country_by_iso2 = {"US": ["United States", "US", "USA", "US"]}
    return index


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (" 123 ", "123"),
        (456, "456"),
        (456.0, "456"),
        (456.5, "456.5"),
    ],
)
def test_normalize_geoname_id_handles_various_types(value, expected):
    assert normalize_geoname_id(value) == expected


def test_coerce_v2_to_v1like_converts_fields():
    record = {
        "id": "https://ror.org/123456",
        "names": [
            {"value": "Example University", "types": ["ror_display"]},
            {"value": "EU", "types": ["acronym"]},
            {"value": "Example U", "types": ["alias"]},
            {"value": "Example Label", "types": ["label"]},
        ],
        "locations": [
            {
                "geonames_details": {
                    "name": "City",
                    "country_name": "Exampleland",
                    "country_code": "EX",
                    "country_subdivision_name": "State",
                    "country_subdivision_code": "EX-ES",
                }
            }
        ],
        "external_ids": {
            "GRID": {"preferred": "grid.1234.5"},
            "ISNI": {"all": ["0000 0001 2345 6789"]},
        },
    }
    converted = coerce_v2_to_v1like(record)
    assert converted["name"] == "Example University"
    assert "Example U" in converted["aliases"]
    assert converted["labels"][0]["label"] == "Example Label"
    assert converted["addresses"][0]["city"] == "City"
    assert converted["addresses"][0]["country_code"] == "EX"


def test_parse_ror_entry_into_single_string_uses_iso_fallback():
    index = make_stub_index()
    tokens = get_special_tokens_dict()
    text = parse_ror_entry_into_single_string(
        index.ror_dict["https://ror.org/123456"],
        index.country_codes_dict,
        index.country_by_iso2,
        tokens,
        use_separator_tokens=False,
        use_wiki=False,
    )
    assert "United States" in text
    assert "; City" in text


def test_extract_grid_and_map_to_ror_prefers_unique_match():
    index = make_stub_index()
    assert (
        index.extract_grid_and_map_to_ror("grid.1234.5 and more")
        == "https://ror.org/ror-1"
    )
    assert index.extract_grid_and_map_to_ror("grid.1234.56") is None


def test_extract_isni_and_map_to_ror_single_match():
    index = make_stub_index()
    assert (
        index.extract_isni_and_map_to_ror("ISNI 0000 0001 2345 6789")
        == "https://ror.org/ror-3"
    )
    assert index.extract_isni_and_map_to_ror("ISNI 0000 0001 2345 6789 and 0000 1111 2222 3333") is None


def test_sum_list_of_list_of_tuples_supports_log_and_linear():
    inputs = [
        [("a", 1.0), ("b", 2.0)],
        [("a", 2.0)],
    ]
    assert sum_list_of_list_of_tuples(inputs) == {"a": 3.0, "b": 2.0}
    logged = sum_list_of_list_of_tuples(inputs, use_log=True)
    assert pytest.approx(logged["a"], rel=1e-6) == np.log1p(1.0) + np.log1p(2.0)
    assert pytest.approx(logged["b"], rel=1e-6) == np.log1p(2.0)
