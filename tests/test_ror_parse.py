from s2aff.ror import get_special_tokens_dict, parse_ror_entry_into_single_string


def _base_entry():
    return {
        "name": "Example University",
        "aliases": ["Example U"],
        "labels": [{"label": "Univ. Example"}],
        "acronyms": ["EU"],
        "addresses": [
            {
                "city": "Metropolis",
                "state": "Example State",
                "state_code": "EX-ES",
                "country_geonames_id": "123456",
                "country_code": "EX",
            }
        ],
        "wikipedia_page": ["https://en.wikipedia.org/wiki/Example_University"],
    }


def _tokens():
    return get_special_tokens_dict()


def test_parse_uses_geoname_ids_for_country_metadata():
    entry = _base_entry()
    country_codes = {
        "123456": ["Exampleland", "EX", "EXA", "EXF"],
    }
    country_by_iso2 = {}

    result = parse_ror_entry_into_single_string(
        entry,
        country_codes,
        country_by_iso2,
        _tokens(),
        use_separator_tokens=False,
        use_wiki=False,
    )

    assert "exampleland" in result.lower()


def test_parse_falls_back_to_iso2_when_geoname_missing():
    entry = _base_entry()
    entry["addresses"][0]["country_geonames_id"] = None
    entry["addresses"][0]["country_code"] = "US"

    country_codes = {}
    country_by_iso2 = {"US": ["United States", "US", "USA", "US"]}

    result = parse_ror_entry_into_single_string(
        entry,
        country_codes,
        country_by_iso2,
        _tokens(),
        use_separator_tokens=False,
        use_wiki=False,
    )

    assert "united states" in result.lower()


def test_parse_normalizes_integer_geoname_id():
    entry = _base_entry()
    entry["addresses"][0]["country_geonames_id"] = 123456

    country_codes = {
        "123456": ["Exampleland", "EX", "EXA", "EXF"],
    }

    result = parse_ror_entry_into_single_string(
        entry,
        country_codes,
        {},
        _tokens(),
        use_separator_tokens=False,
        use_wiki=False,
    )

    assert "exampleland" in result.lower()
