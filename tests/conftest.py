"""
Pytest configuration and fixtures for S2AFF tests.

This module provides session-scoped fixtures that download and cache
large model artifacts once per test session, avoiding repeated downloads
during integration tests.
"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-requires-models",
        action="store_true",
        default=False,
        help="Enable tests that require downloaded model artifacts.",
    )


def pytest_collection_modifyitems(config, items):
    if _requires_models_enabled(config):
        return

    skip_reason = pytest.mark.skip(
        reason=(
            "requires_models tests skipped by default. "
            "Pass --run-requires-models or set S2AFF_RUN_REQUIRES_MODELS=1 to enable."
        )
    )
    for item in items:
        if "requires_models" in item.keywords:
            item.add_marker(skip_reason)


def _requires_models_enabled(pytestconfig):
    return pytestconfig.getoption("--run-requires-models") or os.environ.get("S2AFF_RUN_REQUIRES_MODELS")


@pytest.fixture(scope="session")
def ror_index(pytestconfig):
    """
    Session-scoped fixture that loads the ROR index once.

    This fixture downloads the ROR data if needed and creates the index.
    It's cached for the entire test session to avoid repeated loading.
    """
    if not _requires_models_enabled(pytestconfig):
        pytest.skip("ROR index fixture requires models; enable with --run-requires-models.")

    from s2aff.ror import RORIndex

    index = RORIndex()
    return index


@pytest.fixture(scope="session")
def ner_predictor(pytestconfig):
    """
    Session-scoped fixture that loads the NER model once.
    """
    if not _requires_models_enabled(pytestconfig):
        pytest.skip("NER predictor fixture requires models; enable with --run-requires-models.")

    from s2aff.model import NERPredictor

    predictor = NERPredictor(use_cuda=False)
    return predictor


@pytest.fixture(scope="session")
def pairwise_model(pytestconfig, ror_index):
    """
    Session-scoped fixture that loads the pairwise reranker once.
    """
    if not _requires_models_enabled(pytestconfig):
        pytest.skip("Pairwise reranker fixture requires models; enable with --run-requires-models.")

    from s2aff.model import PairwiseRORLightGBMReranker

    model = PairwiseRORLightGBMReranker(ror_index)
    return model


@pytest.fixture(scope="session")
def s2aff_model(pytestconfig, ner_predictor, ror_index, pairwise_model):
    """
    Session-scoped fixture that creates a full S2AFF model.
    """
    if not _requires_models_enabled(pytestconfig):
        pytest.skip("Full S2AFF fixture requires models; enable with --run-requires-models.")

    from s2aff import S2AFF

    model = S2AFF(ner_predictor, ror_index, pairwise_model)
    return model
