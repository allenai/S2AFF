import s2aff
from s2aff import S2AFF


class DummyNERPredictor:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, raw_affiliations):
        return [self.prediction for _ in raw_affiliations]


class DummyBatchRORIndex:
    def __init__(self):
        self.ror_dict = {
            "ror_early": {"name": "Early Org"},
            "ror_a": {"name": "Org A"},
            "ror_b": {"name": "Org B"},
        }

    def extract_ror(self, raw_affiliation):
        if "EARLY" in raw_affiliation:
            return "ror_early"
        return None

    def extract_grid_and_map_to_ror(self, _):
        return None

    def extract_isni_and_map_to_ror(self, _):
        return None

    def get_candidates_from_main_affiliation_rust_batch(
        self, mains, addresses, early_candidates_list
    ):
        # first non-early item has candidates, second has none
        return [["ror_a", "ror_b"], []], [[0.6, 0.4], []]


class DummyBatchPairwiseModel:
    def __init__(self):
        self.calls = []

    def batch_predict_with_rust_backend(self, affiliations, candidates_list, scores_list):
        self.calls.append(
            (list(affiliations), [list(c) for c in candidates_list], [list(s) for s in scores_list])
        )
        return [["ror_b", "ror_a"]], [[0.9, 0.2]]


def test_s2aff_batch_predict_rust_backend(monkeypatch):
    monkeypatch.setattr(
        s2aff, "parse_ner_prediction", lambda *_: (["Main"], ["Child"], ["Address"], ["early"])
    )
    monkeypatch.setattr(s2aff, "rust_available", lambda: True)

    ner = DummyNERPredictor([{"Example": "B-MAIN"}])
    ror_index = DummyBatchRORIndex()
    pairwise = DummyBatchPairwiseModel()

    model = S2AFF(
        ner,
        ror_index,
        pairwise,
        pipeline="rust",
        look_for_extractable_ids=True,
    )

    raw_affiliations = [
        "EARLY affiliation with ror id",
        "Standard affiliation",
        "No candidates affiliation",
    ]

    outputs = model.predict(raw_affiliations)

    assert pairwise.calls
    assert pairwise.calls[0][0] == ["Standard affiliation"]
    assert outputs[0]["stage2_candidates"] == ["ror_early"]
    assert outputs[0]["stage2_scores"] == [1.0]
    assert outputs[1]["stage2_candidates"] == ["ror_b", "ror_a"]
    assert outputs[1]["stage2_scores"] == [0.9, 0.2]
    assert outputs[2]["stage2_candidates"] == ["NO_CANDIDATES_FOUND"]
    assert outputs[2]["stage2_scores"] == [0.0]


def test_s2aff_init_falls_back_without_rust(monkeypatch):
    monkeypatch.setattr(s2aff, "rust_available", lambda: False)

    model = S2AFF(
        DummyNERPredictor([{"Example": "B-MAIN"}]),
        DummyBatchRORIndex(),
        DummyBatchPairwiseModel(),
        pipeline="rust",
    )

    assert model.stage1_pipeline == "python"
    assert model.stage2_pipeline == "python"
