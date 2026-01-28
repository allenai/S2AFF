import numpy as np
import pytest

from s2aff.features import build_query_context, make_lightgbm_features_with_query_context
from s2aff.rust_backend import get_rust_backend, rust_available
from s2aff.text import fix_text


def _assert_rank_parity(py_candidates, py_scores, rust_candidates, rust_scores, atol=1e-9):
    assert len(py_candidates) == len(rust_candidates)
    assert set(py_candidates) == set(rust_candidates)

    py_map = dict(zip(py_candidates, py_scores))
    rust_map = dict(zip(rust_candidates, rust_scores))
    for key, score in py_map.items():
        assert np.isclose(score, rust_map[key], atol=atol, rtol=1e-6)

    rust_index = {c: i for i, c in enumerate(rust_candidates)}
    for i in range(len(py_candidates) - 1):
        if py_scores[i] > py_scores[i + 1] + atol:
            assert rust_index[py_candidates[i]] < rust_index[py_candidates[i + 1]]


@pytest.mark.requires_models
def test_rust_stage1_parity(ror_index):
    if not rust_available():
        pytest.skip("rust backend not available")
    rust_backend = get_rust_backend(ror_index)
    if rust_backend is None:
        pytest.skip("rust backend not available")

    samples = [
        ("University of California", "Berkeley CA USA", []),
        ("Massachusetts Institute of Technology", "Cambridge MA", []),
        ("Stanford University", "Stanford CA", []),
        ("University of Oxford", "Oxford UK", []),
    ]

    for main, address, early in samples:
        py_candidates, py_scores = ror_index.get_candidates_from_main_affiliation_python(
            main, address, early
        )
        rust_candidates, rust_scores = rust_backend.get_candidates_from_main_affiliation_rust(
            [main], address, early
        )
        _assert_rank_parity(py_candidates, py_scores, rust_candidates, rust_scores)


@pytest.mark.requires_models
def test_rust_stage1_batch_parity(ror_index):
    if not rust_available():
        pytest.skip("rust backend not available")
    rust_backend = get_rust_backend(ror_index)
    if rust_backend is None:
        pytest.skip("rust backend not available")

    samples = [
        ("University of California", "Berkeley CA USA", []),
        ("Massachusetts Institute of Technology", "Cambridge MA", []),
        ("Stanford University", "Stanford CA", []),
        ("University of Oxford", "Oxford UK", []),
    ]

    mains = [s[0] for s in samples]
    addresses = [s[1] for s in samples]
    early_candidates_list = [s[2] for s in samples]

    rust_candidates_list, rust_scores_list = rust_backend.get_candidates_from_main_affiliation_rust_batch(
        mains, addresses, early_candidates_list
    )

    for (main, address, early), rust_candidates, rust_scores in zip(
        samples, rust_candidates_list, rust_scores_list
    ):
        py_candidates, py_scores = ror_index.get_candidates_from_main_affiliation_python(
            main, address, early
        )
        _assert_rank_parity(py_candidates, py_scores, rust_candidates, rust_scores)


@pytest.mark.requires_models
def test_rust_stage2_feature_parity(ror_index, pairwise_model):
    if not rust_available():
        pytest.skip("rust backend not available")
    rust_backend = get_rust_backend(ror_index)
    if rust_backend is None:
        pytest.skip("rust backend not available")

    query = "Department of Physics, University of California Berkeley"
    fixed_query = fix_text(query).lower().replace(",", "")
    query_context = build_query_context(fixed_query)

    candidate_ids = list(ror_index.ror_dict.keys())[:5]

    py_features = []
    for cand in candidate_ids:
        ror_entry = pairwise_model._get_cached_ror_entry(cand)
        py_features.append(
            make_lightgbm_features_with_query_context(query_context, ror_entry, pairwise_model.lm)
        )

    X_all, _ = rust_backend.build_lightgbm_features_with_query_context_batch(
        pairwise_model.lm, [query_context], [candidate_ids], pairwise_model._get_cached_ror_entry
    )
    rust_features = X_all.tolist() if len(X_all) else []

    assert len(py_features) == len(rust_features)
    for a, b in zip(py_features, rust_features):
        assert np.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
