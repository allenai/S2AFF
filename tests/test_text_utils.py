import numpy as np
import pytest

from s2aff.text import fix_text, normalize_geoname_id


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("Example's Lab", "Examples Lab"),
        ("Café & Science Dept.", "Cafe and Science Dept"),
        ("   Leading   and trailing   ", "Leading and trailing"),
        ("#TAB#Value!", "Value"),
        (None, ""),
    ],
)
def test_fix_text_normalizes_strings(input_text, expected):
    assert fix_text(input_text) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (" 12345 ", "12345"),
        ("", None),
        (12345, "12345"),
        (np.int64(67890), "67890"),
        (1234.0, "1234"),
        (1234.5, "1234.5"),
    ],
)
def test_normalize_geoname_id_converts_numeric_variants(value, expected):
    assert normalize_geoname_id(value) == expected


def test_normalize_geoname_id_handles_nan():
    assert normalize_geoname_id(np.nan) is None
