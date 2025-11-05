from collections import Counter
from types import SimpleNamespace

from s2aff.model import parse_ner_prediction


def make_dummy_index(name_lookup=None, address_counts=None):
    if name_lookup is None:
        name_lookup = {"child lab": {"https://ror.org/child-lab"}}
    if address_counts is None:
        address_counts = {"main street": 2}
    return SimpleNamespace(
        ror_name_direct_lookup=name_lookup,
        ror_address_counter=Counter(address_counts),
    )


def test_parse_ner_prediction_handles_split_token():
    ner_prediction = [
        {"Alpha": "B-MAIN"},
        {"Institute": "I-MAIN"},
        {"and": "B-SPLIT"},
        {"Beta": "I-MAIN"},
    ]
    main, child, address, early = parse_ner_prediction(ner_prediction, make_dummy_index())
    assert main == ["Alpha Institute"]
    assert child == []
    assert address == []
    assert early == []


def test_parse_ner_prediction_promotes_certainly_main_child_and_collects_candidates():
    ner_prediction = [
        {"Alpha": "B-MAIN"},
        {"Institute": "I-MAIN"},
        {"Child": "B-CHILD"},
        {"Lab": "I-CHILD"},
        {"City": "B-ADDRESS"},
        {"USA": "I-ADDRESS"},
    ]
    main, child, address, early = parse_ner_prediction(ner_prediction, make_dummy_index())
    assert "Alpha Institute" in main
    assert child == ["Child Lab"]
    assert address == ["City USA"]
    assert early == ["https://ror.org/child-lab"]


def test_parse_ner_prediction_moves_child_to_main_when_none_detected():
    ner_prediction = [
        {"Satellite": "B-CHILD"},
        {"Campus": "I-CHILD"},
    ]
    main, child, address, early = parse_ner_prediction(ner_prediction, make_dummy_index())
    assert main == ["Satellite Campus"]
    assert child == []
    assert address == []
    assert early == []


def test_parse_ner_prediction_promotes_address_only_tokens_to_main():
    ner_prediction = [
        {"12345": "B-ADDRESS"},
        {"Main": "I-ADDRESS"},
        {"Street": "I-ADDRESS"},
    ]
    main, child, address, early = parse_ner_prediction(ner_prediction, make_dummy_index())
    assert main == ["12345 Main Street"]
    assert child == []
    assert address == []
    assert early == []


def test_parse_ner_prediction_removes_main_if_flagged_as_address():
    ner_prediction = [
        {"Alpha": "B-MAIN"},
        {"Campus": "I-MAIN"},
        {"Science": "B-CHILD"},
        {"Center": "I-CHILD"},
    ]
    index = make_dummy_index(name_lookup={}, address_counts={"alpha campus": 1})
    main, child, address, early = parse_ner_prediction(ner_prediction, index)
    assert "Alpha Campus" not in main
    assert "Science Center" in main
    assert child == ["Alpha Campus"]
    assert address == ["Alpha Campus"]
    assert early == []


def test_parse_ner_prediction_handles_multiple_main_segments():
    ner_prediction = [
        {"Alpha": "B-MAIN"},
        {"Institute": "I-MAIN"},
        {"Beta": "B-MAIN"},
        {"University": "I-MAIN"},
    ]
    main, child, address, early = parse_ner_prediction(ner_prediction, make_dummy_index())
    assert "Alpha Institute" in main
    assert "Beta University" in main
    assert child == []
    assert address == []
    assert early == []
