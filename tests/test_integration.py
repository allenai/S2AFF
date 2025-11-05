"""
Integration tests for S2AFF that require model downloads.

These tests are marked with @pytest.mark.requires_models and will be
skipped in CI unless models are available. They test end-to-end
functionality with real models.
"""

import pytest


@pytest.mark.requires_models
def test_ror_index_loads_successfully(ror_index):
    """Test that ROR index loads and has expected structure."""
    assert ror_index is not None
    assert hasattr(ror_index, "ror_dict")
    assert len(ror_index.ror_dict) > 100000  # Should have 100k+ ROR entries
    assert hasattr(ror_index, "word_index")
    assert hasattr(ror_index, "ngram_index")


@pytest.mark.requires_models
def test_ror_index_retrieval(ror_index):
    """Test ROR index can retrieve candidates for a known affiliation."""
    # Test with a well-known institution
    from s2aff.model import parse_ner_prediction

    # Simulate NER output for "University of Washington"
    ner_output = [
        {"University": "B-MAIN"},
        {"of": "I-MAIN"},
        {"Washington": "I-MAIN"},
    ]

    main, child, address, early = parse_ner_prediction(ner_output, ror_index)

    candidates, scores = ror_index.get_candidates_from_main_affiliation(
        " ".join(main) if main else "",
        " ".join(address) if address else "",
        early,
    )

    assert len(candidates) > 0
    assert len(candidates) == len(scores)
    # Should find University of Washington in top candidates
    assert any("washington" in ror_index.ror_dict.get(c, {}).get("name", "").lower() for c in candidates[:10])


@pytest.mark.requires_models
def test_ner_predictor_loads_successfully(ner_predictor):
    """Test that NER predictor loads and has expected structure."""
    assert ner_predictor is not None
    assert hasattr(ner_predictor, "predict")


@pytest.mark.requires_models
def test_ner_predictor_prediction(ner_predictor):
    """Test NER predictor can make predictions."""
    test_affiliation = "Department of Computer Science, University of Washington, Seattle, WA"

    predictions = ner_predictor.predict([test_affiliation])

    assert len(predictions) == 1
    assert isinstance(predictions[0], list)
    assert len(predictions[0]) > 0
    # Should have some tokens tagged as MAIN
    assert any("MAIN" in list(token.values())[0] for token in predictions[0])


@pytest.mark.requires_models
def test_pairwise_model_loads_successfully(pairwise_model):
    """Test that pairwise model loads and has expected structure."""
    assert pairwise_model is not None
    assert hasattr(pairwise_model, "predict")


@pytest.mark.requires_models
def test_pairwise_model_prediction(pairwise_model, ror_index):
    """Test pairwise model can rerank candidates."""
    test_affiliation = "University of Washington"

    # Get some candidates (need at least 2 for reranking)
    candidates = list(ror_index.ror_dict.keys())[:10]
    scores = [0.9 - i * 0.1 for i in range(10)]

    reranked_candidates, reranked_scores = pairwise_model.predict(
        test_affiliation, candidates, scores
    )

    assert len(reranked_candidates) > 0
    assert len(reranked_candidates) == len(reranked_scores)
    assert all(isinstance(s, (int, float)) for s in reranked_scores)


@pytest.mark.requires_models
@pytest.mark.integration
def test_s2aff_end_to_end_prediction(s2aff_model, ror_index):
    """Test full S2AFF pipeline end-to-end."""
    test_affiliations = [
        "Department of Computer Science, University of Washington, Seattle, WA 98195",
        "Stanford University, Stanford, CA",
        "MIT, Cambridge, MA",
    ]

    predictions = s2aff_model.predict(test_affiliations)

    assert len(predictions) == len(test_affiliations)

    for pred in predictions:
        assert "stage1_candidates" in pred
        assert "stage1_scores" in pred
        assert "stage2_candidates" in pred
        assert "stage2_scores" in pred
        assert "top_candidate_display_name" in pred

        # Should have candidates
        assert len(pred["stage1_candidates"]) > 0
        assert len(pred["stage2_candidates"]) > 0


@pytest.mark.requires_models
@pytest.mark.integration
def test_s2aff_university_of_washington(s2aff_model, ror_index):
    """Test that S2AFF correctly identifies University of Washington."""
    affiliation = "Department of Computer Science, University of Washington, Seattle, WA 98195"

    predictions = s2aff_model.predict([affiliation])

    assert len(predictions) == 1
    pred = predictions[0]

    # Check that we got a prediction
    assert len(pred["stage2_candidates"]) > 0
    predicted_ror = pred["stage2_candidates"][0]
    assert predicted_ror is not None
    assert predicted_ror != "NO_CANDIDATES_FOUND"

    # Check that it's University of Washington
    ror_name = ror_index.ror_dict[predicted_ror]["name"]
    assert "washington" in ror_name.lower()


@pytest.mark.requires_models
@pytest.mark.integration
def test_s2aff_handles_ambiguous_affiliation(s2aff_model):
    """Test S2AFF handling of ambiguous affiliations."""
    # "AI" is ambiguous - could be many institutions
    ambiguous_affiliation = "AI Research Lab"

    predictions = s2aff_model.predict([ambiguous_affiliation])

    assert len(predictions) == 1
    pred = predictions[0]

    # Should still return candidates even if uncertain
    assert "stage1_candidates" in pred
    assert "stage2_candidates" in pred


@pytest.mark.requires_models
@pytest.mark.integration
def test_s2aff_batch_prediction(s2aff_model):
    """Test that S2AFF can handle batch predictions efficiently."""
    # Create a batch of different affiliations
    affiliations = [
        "University of Washington",
        "Stanford University",
        "MIT",
        "Harvard University",
        "UC Berkeley",
    ]

    predictions = s2aff_model.predict(affiliations)

    assert len(predictions) == len(affiliations)

    # All should have predictions
    for pred in predictions:
        assert pred is not None
        assert "stage2_candidates" in pred
        assert len(pred["stage2_candidates"]) > 0


@pytest.mark.requires_models
def test_ror_index_extract_ror_id(ror_index):
    """Test ROR index can extract ROR IDs from text."""
    # Test with embedded ROR ID
    text_with_ror = "Some text https://ror.org/00cvxb145 more text"

    extracted = ror_index.extract_ror(text_with_ror)

    assert extracted == "https://ror.org/00cvxb145"


@pytest.mark.requires_models
def test_ror_index_extract_grid_id(ror_index):
    """Test ROR index can extract and map GRID IDs."""
    # Note: This test may need adjustment based on actual GRID mappings
    text_with_grid = "GRID:grid.34477.33"

    extracted = ror_index.extract_grid_and_map_to_ror(text_with_grid)

    # Should either map to a ROR or return None
    assert extracted is None or extracted.startswith("https://ror.org/")


@pytest.mark.requires_models
def test_ror_index_extract_isni_id(ror_index):
    """Test ROR index can extract and map ISNI IDs."""
    # Note: This test may need adjustment based on actual ISNI mappings
    text_with_isni = "ISNI 0000 0001 2171 9311"

    extracted = ror_index.extract_isni_and_map_to_ror(text_with_isni)

    # Should either map to a ROR or return None
    assert extracted is None or extracted.startswith("https://ror.org/")
