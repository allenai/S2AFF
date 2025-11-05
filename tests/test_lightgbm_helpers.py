import numpy as np
import pytest

from s2aff.lightgbm_helpers import format_group, lightgbmExperiment


def test_format_group_computes_run_lengths():
    groups = [0, 0, 0, 1, 2, 2]
    assert np.array_equal(format_group(groups), np.array([3, 1, 2]))


def test_lightgbm_experiment_fit_and_convert_data(monkeypatch):
    experiment = lightgbmExperiment(
        learning_task="ranking",
        metric="lambdarank",
        eval_metric="ndcg",
        label_gain_max=2,
        eval_at="1",
        n_estimators=3,
        hyperopt_evals=1,
        device="cpu",
        tree_learner="serial",
        random_seed=0,
        threads=1,
    )

    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    group = [0, 0, 1, 1]

    dataset = experiment.convert_data(X, y, group)
    assert dataset.get_group().tolist() == [2, 2]

    recorded = {}

    class DummyBooster:
        pass

    def fake_train(params, train_data, valid_sets, valid_names, evals_result, num_boost_round):
        recorded["params"] = params
        recorded["train_data"] = train_data
        recorded["valid_sets"] = valid_sets
        evals_result["test"] = {"ndcg@1": [0.2] * num_boost_round}
        return DummyBooster()

    monkeypatch.setattr("s2aff.lightgbm_helpers.lgb.train", fake_train)

    params = {
        "learning_rate": 0.1,
        "num_leaves": 4,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
    }
    booster, eval_result = experiment.fit(params, dataset, dataset, n_estimators=3, seed=0)

    assert isinstance(booster, DummyBooster)
    assert len(eval_result) == 3
    assert recorded["params"]["feature_fraction_seed"] == 2
    assert recorded["params"]["bagging_seed"] == 3
