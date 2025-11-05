import numpy as np
import pytest

from blingfire import text_to_words
from s2aff.features import FEATURE_NAMES
from s2aff.model import NERPredictor, PairwiseRORLightGBMReranker
from s2aff.text import fix_text

_FEATURE_NAMES = list(FEATURE_NAMES)


class _RecordingNERModel:
    def __init__(self):
        self.calls = []

    def predict(self, texts):
        self.calls.append(list(texts))
        outputs = []
        for text in texts:
            tokens = [token for token in text.split() if token]
            outputs.append([{token: "I-MAIN"} for token in tokens])
        return outputs, None


def test_ner_predictor_predict_delegates_and_normalizes():
    model = _RecordingNERModel()
    predictor = NERPredictor(model_path=None, use_cuda=False, model=model)

    raw_text = "Example University, Seattle, WA"
    result = predictor.predict(raw_text)

    expected = text_to_words(fix_text(raw_text))
    assert model.calls == [[expected]]
    assert isinstance(result, list)
    assert result and result[0] == {"Example": "I-MAIN"}


def test_ner_predictor_predict_requires_model():
    predictor = NERPredictor(model_path=None, model_type="roberta", use_cuda=False, model=None)
    with pytest.raises(ValueError):
        predictor.predict("Any text")


class _FakeBooster:
    def predict(self, data, num_threads=0):
        name_ind = _FEATURE_NAMES.index("names_frac_of_query_matched_in_text")
        acronym_ind = _FEATURE_NAMES.index("acronyms_frac_of_query_matched_in_text")
        return np.array([row[name_ind] + row[acronym_ind] for row in data], dtype=float)


class _FakeLanguageModel:
    def score(self, text, bos=False, eos=False):
        return -len(text.split())


def _fake_parse_ror_entry(ror_id, index):
    return index.feature_templates[ror_id]


def _fake_make_lightgbm_features(query, entry, language_model):
    features = np.zeros(len(_FEATURE_NAMES), dtype=float)
    features[_FEATURE_NAMES.index("names_frac_of_query_matched_in_text")] = entry.get("name_match", 0.0)
    features[_FEATURE_NAMES.index("acronyms_frac_of_query_matched_in_text")] = entry.get("acronym_match", 0.0)
    features[_FEATURE_NAMES.index("city_frac_of_query_matched_in_text")] = entry.get("city_match", 0.0)
    return features


def test_pairwise_reranker_uses_model_scores(monkeypatch):
    class TinyIndex:
        def __init__(self):
            self.ror_dict = {"https://ror.org/1": {}, "https://ror.org/2": {}}
            self.feature_templates = {
                "https://ror.org/1": {"name_match": 0.2, "acronym_match": 0.1, "city_match": 0.0},
                "https://ror.org/2": {"name_match": 0.8, "acronym_match": 0.0, "city_match": 1.0},
            }

    index = TinyIndex()
    monkeypatch.setattr(
        "s2aff.model.parse_ror_entry_into_single_string_lightgbm",
        _fake_parse_ror_entry,
    )
    monkeypatch.setattr(
        "s2aff.model.make_lightgbm_features",
        _fake_make_lightgbm_features,
    )

    reranker = PairwiseRORLightGBMReranker(
        index,
        model_path=None,
        kenlm_model_path=None,
        num_threads=0,
        booster=_FakeBooster(),
        language_model=_FakeLanguageModel(),
    )

    candidates, scores = reranker.predict(
        "Example University",
        ["https://ror.org/1", "https://ror.org/2"],
        [0.4, 0.3],
    )

    assert list(candidates) == ["https://ror.org/2", "https://ror.org/1"]
    assert pytest.approx(scores[0], rel=1e-6) == 0.85
    assert pytest.approx(scores[1], rel=1e-6) == 0.3
