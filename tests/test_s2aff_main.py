import s2aff
from s2aff import S2AFF


class DummyNERPredictor:
    def __init__(self, prediction):
        self.prediction = prediction
        self.last_input = None

    def predict(self, raw_affiliations):
        self.last_input = list(raw_affiliations)
        return [self.prediction for _ in raw_affiliations]


class DummyRORIndex:
    def __init__(self, candidates, scores, display_name="Example University"):
        self._candidates = candidates
        self._scores = scores
        self.ror_dict = {candidates[0]: {"name": display_name}} if candidates else {}

    def extract_ror(self, _):
        return None

    def extract_grid_and_map_to_ror(self, _):
        return None

    def extract_isni_and_map_to_ror(self, _):
        return None

    def get_candidates_from_main_affiliation(self, main, address, early_candidates):
        return list(self._candidates), list(self._scores)


class RecordingPairwiseModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def predict(self, raw_affiliation, candidates, scores):
        self.calls.append((raw_affiliation, list(candidates), list(scores)))
        return self.response


class NoCallPairwiseModel:
    def predict(self, *args, **kwargs):
        raise AssertionError("Pairwise model should not be called")


def test_s2aff_predict_short_circuits_when_id_found(monkeypatch):
    ner_prediction = [{"Example": "B-MAIN"}]
    ner = DummyNERPredictor(ner_prediction)

    class ShortCircuitRORIndex:
        def __init__(self):
            self.ror_dict = {"https://ror.org/extracted": {"name": "Extracted Org"}}

        def extract_ror(self, _):
            return "https://ror.org/extracted"

        def extract_grid_and_map_to_ror(self, _):
            return None

        def extract_isni_and_map_to_ror(self, _):
            return None

        def get_candidates_from_main_affiliation(self, *args, **kwargs):
            raise AssertionError("Should not fetch candidates when ID is extracted")

    pairwise = NoCallPairwiseModel()
    monkeypatch.setattr(
        s2aff, "parse_ner_prediction", lambda *_: (["Main"], [], [], [])
    )

    model = S2AFF(ner, ShortCircuitRORIndex(), pairwise)

    outputs = model.predict(["Affiliation with https://ror.org/extracted embedded"])
    result = outputs[0]
    assert result["stage1_candidates"] == ["https://ror.org/extracted"]
    assert result["stage1_scores"] == [1.0]
    assert result["stage2_candidates"] == ["https://ror.org/extracted"]
    assert result["stage2_scores"] == [1.0]


def test_s2aff_predict_converts_string_and_reranks(monkeypatch):
    ner_prediction = [{"Example": "B-MAIN"}]
    ner = DummyNERPredictor(ner_prediction)
    ror_index = DummyRORIndex(
        candidates=["https://ror.org/1", "https://ror.org/2"],
        scores=[0.8, 0.2],
    )
    pairwise = RecordingPairwiseModel(
        (["https://ror.org/1"], [0.95])
    )
    monkeypatch.setattr(
        s2aff, "parse_ner_prediction", lambda *_: (["Main"], ["Child"], ["Address"], ["early"])
    )

    model = S2AFF(ner, ror_index, pairwise, number_of_top_candidates_to_return=2, look_for_extractable_ids=False)

    outputs = model.predict("Raw affiliation text")

    assert ner.last_input == ["Raw affiliation text"]
    assert pairwise.calls and pairwise.calls[0][1] == ["https://ror.org/1", "https://ror.org/2"]
    assert outputs[0]["stage2_candidates"] == ["https://ror.org/1"]
    assert outputs[0]["stage1_candidates"] == ["https://ror.org/1", "https://ror.org/2"]


def test_s2aff_predict_handles_no_candidates(monkeypatch):
    ner_prediction = [{"Example": "B-MAIN"}]
    ner = DummyNERPredictor(ner_prediction)
    ror_index = DummyRORIndex(candidates=[], scores=[])
    pairwise = NoCallPairwiseModel()
    monkeypatch.setattr(
        s2aff, "parse_ner_prediction", lambda *_: (["Main"], [], [], [])
    )

    model = S2AFF(ner, ror_index, pairwise, look_for_extractable_ids=False)

    outputs = model.predict(["Raw affiliation text"])

    assert outputs[0]["stage2_candidates"] == ["NO_CANDIDATES_FOUND"]
    assert outputs[0]["stage2_scores"] == [0.0]


def test_s2aff_predict_emits_no_ror_for_low_single_score(monkeypatch):
    ner_prediction = [{"Example": "B-MAIN"}]
    ner = DummyNERPredictor(ner_prediction)
    ror_index = DummyRORIndex(candidates=["https://ror.org/solo"], scores=[0.1])
    pairwise = NoCallPairwiseModel()
    monkeypatch.setattr(
        s2aff, "parse_ner_prediction", lambda *_: (["Main"], [], [], [])
    )

    model = S2AFF(ner, ror_index, pairwise, look_for_extractable_ids=False, pairwise_model_threshold=0.3)

    outputs = model.predict(["Raw affiliation text"])
    result = outputs[0]

    assert result["stage1_candidates"] == ["https://ror.org/solo"]
    assert result["stage1_scores"] == [0.1]
    assert result["stage2_candidates"] == ["NO_ROR_FOUND"]
    assert result["stage2_scores"] == [0.0]
