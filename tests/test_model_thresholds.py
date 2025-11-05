from collections import Counter

from s2aff import S2AFF


class StubNerPredictor:
    def __init__(self):
        self.prediction = [
            {"Alpha": "B-MAIN"},
            {"University": "I-MAIN"},
        ]

    def predict(self, affiliations):
        return [self.prediction for _ in affiliations]


class StubRorIndex:
    def __init__(self, candidates, scores):
        self._candidates = candidates
        self._scores = scores
        self.calls = 0
        self.ror_name_direct_lookup = {}
        self.ror_address_counter = Counter()
        self.ror_dict = {
            candidate: {
                "name": f"Name for {candidate}",
                "aliases": [],
                "labels": [],
                "acronyms": [],
                "addresses": [
                    {
                        "city": "",
                        "state": "",
                        "state_code": None,
                        "country_geonames_id": None,
                        "country_code": "US",
                    }
                ],
                "works_count": 0,
            }
            for candidate in candidates
        }

    def get_candidates_from_main_affiliation(self, main, address, early):
        self.calls += 1
        return list(self._candidates), list(self._scores)

    # Extraction helpers are invoked only when look_for_extractable_ids is True,
    # but we provide them for completeness.
    def extract_ror(self, raw_affiliation):
        return None

    def extract_grid_and_map_to_ror(self, raw_affiliation):
        return None

    def extract_isni_and_map_to_ror(self, raw_affiliation):
        return None


class StubPairwiseModel:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def predict(self, raw_affiliation, candidates, scores):
        self.calls.append((raw_affiliation, list(candidates), list(scores)))
        return self.output


def test_s2aff_returns_no_ror_when_single_candidate_score_is_low():
    ner = StubNerPredictor()
    ror_index = StubRorIndex(["https://ror.org/1"], [0.1])
    pairwise = StubPairwiseModel(output=(["https://ror.org/1"], [0.1]))

    model = S2AFF(
        ner_predictor=ner,
        ror_index=ror_index,
        pairwise_model=pairwise,
        look_for_extractable_ids=False,
        pairwise_model_threshold=0.3,
    )

    result = model.predict(["Alpha University"])[0]

    assert result["stage2_candidates"] == ["NO_ROR_FOUND"]
    assert result["stage2_scores"] == [0.0]
    assert pairwise.calls == []


def test_s2aff_threshold_applies_after_reranking():
    ner = StubNerPredictor()
    candidates = ["https://ror.org/1", "https://ror.org/2"]
    ror_index = StubRorIndex(candidates, [0.6, 0.5])
    pairwise = StubPairwiseModel(output=(candidates, [0.1, 0.05]))

    model = S2AFF(
        ner_predictor=ner,
        ror_index=ror_index,
        pairwise_model=pairwise,
        look_for_extractable_ids=False,
        pairwise_model_threshold=0.3,
        pairwise_model_delta_threshold=0.2,
    )

    result = model.predict(["Alpha University"])[0]

    assert result["stage2_candidates"] == ["NO_ROR_FOUND"]
    assert result["stage2_scores"] == [0.0]
    assert len(pairwise.calls) == 1
    called_raw, called_candidates, called_scores = pairwise.calls[0]
    assert called_raw == "Alpha University"
    assert called_candidates == candidates
    assert called_scores == [0.6, 0.5]
