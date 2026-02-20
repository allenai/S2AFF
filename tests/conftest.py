"""
Pytest configuration and fixtures for S2AFF tests.

This module provides session-scoped fixtures that download and cache
large model artifacts once per test session, avoiding repeated downloads
during integration tests.
"""

import os
from pathlib import Path

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
            "Pass --run-requires-models or set S2AFF_RUN_REQUIRES_MODELS=1 to enable. "
            "Tests auto-enable when local model files are present unless "
            "S2AFF_SKIP_REQUIRES_MODELS=1 is set."
        )
    )
    for item in items:
        if "requires_models" in item.keywords:
            item.add_marker(skip_reason)


def _env_flag(name):
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _project_root():
    return Path(__file__).resolve().parents[1]


def _local_models_available():
    data_dir = _project_root() / "data"
    if not data_dir.is_dir():
        return False

    ror_data = list(data_dir.glob("*-ror-data.json"))
    if not ror_data:
        return False

    required_files = [
        data_dir / "openalex_works_counts.csv",
        data_dir / "country_info.txt",
        data_dir / "ror_edits.jsonl",
        data_dir / "raw_affiliations_lowercased.binary",
        data_dir / "lightgbm_model.booster",
    ]
    if any(not path.is_file() for path in required_files):
        return False

    ner_dir = data_dir / "ner_model"
    if not ner_dir.is_dir():
        return False

    ner_files = [
        ner_dir / "pytorch_model.bin",
        ner_dir / "config.json",
        ner_dir / "tokenizer_config.json",
    ]
    if any(not path.is_file() for path in ner_files):
        return False

    return True


def _requires_models_enabled(pytestconfig):
    if pytestconfig.getoption("--run-requires-models"):
        return True
    run_flag = _env_flag("S2AFF_RUN_REQUIRES_MODELS")
    if run_flag is not None:
        return run_flag
    if _env_flag("S2AFF_SKIP_REQUIRES_MODELS"):
        return False
    return _local_models_available()


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
