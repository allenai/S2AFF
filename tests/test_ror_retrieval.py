import s2aff.ror as ror_module


def make_stub_index(monkeypatch, word_results, ngram_results, insert_index=1, address_index=None):
    index = ror_module.RORIndex.__new__(ror_module.RORIndex)
    index.word_multiplier = 1
    index.max_intersection_denominator = True
    index.idf_weight = None
    index.ns = {3}
    index.score_based_early_cutoff = 0.0
    index.insert_early_candidates_ind = insert_index
    index.reinsert_cutoff_frac = 1.1
    index.address_index = address_index or {"full_index": {}, "city_index": {}}
    index.ror_address_counter = {}
    index.word_index = {}
    index.ngram_index = {}
    index.ror_dict = {}

    def fake_word_nns(candidate, word_index, **kwargs):
        return word_results

    def fake_ngram_nns(candidate, ngram_index, **kwargs):
        return ngram_results

    monkeypatch.setattr(ror_module, "jaccard_word_nns", fake_word_nns)
    monkeypatch.setattr(ror_module, "jaccard_ngram_nns", fake_ngram_nns)

    return index


def test_get_candidates_inserts_early_candidates(monkeypatch):
    word_results = [("ror1__NAME", 0.6), ("ror2__NAME", 0.5)]
    ngram_results = [("ror1__NAME", 0.2)]
    index = make_stub_index(monkeypatch, word_results, ngram_results, insert_index=1)

    candidates, scores = ror_module.RORIndex.get_candidates_from_main_affiliation(
        index, "Example Institute", "", ["ror3"]
    )

    assert candidates[:3] == ("ror1", "ror3", "ror2")
    assert scores[1] == -0.1


def test_get_candidates_reinserts_high_scoring_removed_candidates(monkeypatch):
    word_results = [("ror1__NAME", 1.0), ("ror2__NAME", 0.8)]
    ngram_results = []
    address_index = {"full_index": {"paris": {"ror2"}}, "city_index": {}}
    index = make_stub_index(monkeypatch, word_results, ngram_results, insert_index=None, address_index=address_index)

    candidates, scores = ror_module.RORIndex.get_candidates_from_main_affiliation(
        index, "Example Institute", "Paris", []
    )

    assert candidates[0] == "ror2"
    assert "ror1" in candidates
    assert scores[candidates.index("ror1")] == -0.15
