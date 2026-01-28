# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:35:14 2018
@author: serge
"""

import time
from abc import ABCMeta, abstractmethod
from itertools import groupby
import numpy as np
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL


def format_group(g):
    return np.array([len(list(group)) for _, group in groupby(g)])


class Experiment(object, metaclass=ABCMeta):

    # This method must be overridden but can still be called via super by subclasses have shared construction logic
    @abstractmethod
    def __init__(self):
        pass

    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        raise NotImplementedError("Method fit is not implemented.")

    def convert_data(self, X, y, group):
        raise NotImplementedError("Method fit is not implemented.")

    def run_optimization(self, train_dmatrix, val_dmatrix, params, n_estimators=None, verbose=False):
        n_estimators = n_estimators or self.n_estimators
        best_n_estimators = n_estimators
        evals_results, start_time = [], time.time()
        _, evals_result = self.fit(params, train_dmatrix, val_dmatrix, n_estimators, self.random_seed)
        try:
            evals_result_final = evals_result[best_n_estimators - 1]
        except:
            evals_result_final = evals_result

        eval_time = time.time() - start_time

        cv_result = {
            "loss": self.eval_multiplier * evals_result_final,
            "best_n_estimators": best_n_estimators,
            "eval_time": eval_time,
            "status": STATUS_FAIL if np.isnan(evals_result_final) else STATUS_OK,
            "params": params.copy(),
        }
        self.best_loss = min(self.best_loss, cv_result["loss"])
        self.hyperopt_eval_num += 1
        cv_result.update({"hyperopt_eval_num": self.hyperopt_eval_num, "best_loss": self.best_loss})

        if verbose:
            print(
                "[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}".format(
                    self.hyperopt_eval_num,
                    self.hyperopt_evals,
                    eval_time,
                    self.eval_metric,
                    cv_result["loss"],
                    self.best_loss,
                )
            )
        return cv_result

    def run_test(self, dtrain, dtest, params=None, n_estimators=None, custom_metric=None, seed=0):
        params = params or self.best_params
        n_estimators = n_estimators or self.best_n_estimators or self.n_estimators
        start_time = time.time()
        self.bst, evals_result = self.fit(params, dtrain, dtest, n_estimators, seed=seed)
        try:
            evals_result_final = evals_result[-1]
        except:
            evals_result_final = evals_result
        eval_time = time.time() - start_time
        result = {
            "loss": evals_result_final,
            "n_estimators": n_estimators,
            "eval_time": eval_time,
            "status": STATUS_OK,
            "params": params.copy(),
        }
        return result

    def run(
        self,
        X_train,
        y_train,
        group_train,
        weights_train,
        X_val,
        y_val,
        group_val,
        weights_val,
        X_test,
        y_test,
        group_test,
        weights_test,
    ):
        print("Loading and preprocessing dataset...")
        train_dmatrix = self.convert_data(X_train, y_train, group_train, weights_train)
        val_dmatrix = self.convert_data(X_val, y_val, group_val, weights_val)
        test_dmatrix = self.convert_data(X_test, y_test, group_test, weights_test)

        print("Optimizing params...")
        cv_result = self.optimize_params(train_dmatrix, val_dmatrix)
        self.print_result(cv_result, "\nBest result on cv")

        print("\nTraining algorithm with the tuned parameters for different seeds...")
        test_losses = []
        for seed in [42, 1999, self.random_seed]:
            test_result = self.run_test(train_dmatrix, test_dmatrix, seed=seed)
            test_losses.append(test_result["loss"])
            print("For seed=%d Test's %s : %.5f" % (seed, self.eval_metric, test_losses[-1]))

        print(
            "\nTest's %s mean: %.5f, Test's %s std: %.5f"
            % (self.eval_metric, np.mean(test_losses), self.eval_metric, np.std(test_losses))
        )

        return test_losses

    def optimize_params(self, train_dmatrix, val_dmatrix, max_evals=None, verbose=True):
        max_evals = max_evals or self.hyperopt_evals
        self.trials = Trials()
        self.hyperopt_eval_num, self.best_loss = 0, np.inf

        fmin(
            fn=lambda params: self.run_optimization(train_dmatrix, val_dmatrix, params, verbose=verbose),
            space=self.space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
        )

        self.best_params = self.trials.best_trial["result"]["params"]
        if "best_n_estimators" in self.trials.best_trial["result"].keys():
            self.best_n_estimators = self.trials.best_trial["result"]["best_n_estimators"]
        return self.trials.best_trial["result"]

    def print_result(self, result, name="", extra_keys=None):
        print("%s:\n" % name)
        print("%s = %s" % (self.eval_metric, result["loss"]))
        if "best_n_estimators" in result.keys():
            print("best_n_estimators = %s" % result["best_n_estimators"])
        elif "n_estimators" in result.keys():
            print("n_estimators = %s" % result["n_estimators"])
        print("params = %s" % result["params"])
        if extra_keys is not None:
            for k in extra_keys:
                if k in result:
                    print("%s = %f" % (k, result[k]))


class lightgbmExperiment(Experiment):
    def __init__(
        self,
        learning_task,
        metric,
        eval_metric=None,
        label_gain_max=1000,
        eval_at="1,2,3,4,5,100",
        n_estimators=5000,
        hyperopt_evals=25,
        device="cpu",
        tree_learner="data",
        random_seed=0,
        threads=10,
        feval=None,
    ):

        super().__init__()

        if learning_task == "classification":
            assert metric in {"binary", "multiclass", "multiclassova"}
        elif learning_task == "regression":
            assert metric in {"rmse", "mae", "huber", "fair", "poisson", "quantile", "mape", "gamma", "tweedie"}
        elif learning_task == "ranking":
            assert metric in {"lambdarank", "rank_xendcg"}

        self.learning_task = learning_task
        self.metric = metric
        self.n_estimators = n_estimators
        self.label_gain_max = int(label_gain_max) + 1
        self.hyperopt_evals = hyperopt_evals
        self.device = device
        self.random_seed = random_seed
        self.threads = threads
        self.feval = feval

        self.best_loss = np.inf
        self.best_n_estimators = None
        self.hyperopt_eval_num = 0

        # eval metric
        if eval_metric in {
            "auc",
            "binary_logloss",
            "binary_error",
            "multi_logloss",
            "multi_error",
            "l1",
            "l2",
            "rmse",
            "quantile",
            "mape",
            "huber",
            "fair",
            "poisson",
            "gamma",
            "gamma_deviance",
            "tweedie",
            "ndcg",
            "map",
        }:
            self.eval_metric = eval_metric
        else:
            self.eval_metric = metric  # same as objective

        self.space = {
            "learning_rate": hp.loguniform("learning_rate", -7, 0),
            "num_leaves": hp.qloguniform("num_leaves", 2, 7, 1),
            "feature_fraction": hp.uniform("feature_fraction", 0.5, 1),
            "bagging_fraction": hp.uniform("bagging_fraction", 0.5, 1),
            "min_data_in_leaf": hp.qloguniform("min_data_in_leaf", 2, 7, 1),
            "min_sum_hessian_in_leaf": hp.loguniform("min_sum_hessian_in_leaf", -16, 5),
            "lambda_l1": hp.loguniform("lambda_l1", -16, 2),
            "lambda_l2": hp.loguniform("lambda_l2", -16, 2),
        }

        self.constant_params = {
            "max_bin": 63,
            "device": self.device,
            "gpu_use_dp": False,
            "min_gain_to_split": 0.1,  # try to control overfitting
            "num_threads": self.threads,
            "tree_learner": tree_learner,
            "objective": self.metric,
            "metric": self.eval_metric,
            "label_gain": ",".join([str(int(np.log2(i + 1))) for i in range(self.label_gain_max)]),
            "bagging_freq": 1,
            "verbose": -1,
            "eval_at": eval_at,
        }

        # hyperopt minimizes metrics, so we have negate ones where larger is better
        if self.eval_metric in {"auc", "ndcg", "map"}:
            self.eval_multiplier = -1
        else:
            self.eval_multiplier = 1

    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        params["num_leaves"] = max(int(params["num_leaves"]), 2)
        params["min_data_in_leaf"] = int(params["min_data_in_leaf"])
        params.update(self.constant_params)
        params.update(
            {
                "feature_fraction_seed": 2 + seed,
                "bagging_seed": 3 + seed,
                "drop_seed": 4 + seed,
            }
        )
        if self.metric in {"multiclass", "multiclassova"} and "num_class" not in params:
            params["num_class"] = len(np.unique(dtrain.get_label()))

        evals_result = {}
        # LightGBM versions differ on whether `evals_result` is a supported kwarg; fall back to callbacks if needed.
        try:
            bst = lgb.train(
                params,
                dtrain,
                valid_sets=dtest,
                valid_names=["test"],
                evals_result=evals_result,
                num_boost_round=n_estimators,
            )
        except TypeError:
            evals_result = {}
            bst = lgb.train(
                params,
                dtrain,
                valid_sets=dtest,
                valid_names=["test"],
                num_boost_round=n_estimators,
                callbacks=[lgb.record_evaluation(evals_result)],
            )

        if self.eval_metric in {"ndcg", "map"}:
            key = f"{self.eval_metric}@1"
        else:
            key = self.eval_metric
        results = evals_result["test"][key]

        return bst, results

    def convert_data(self, X, y, group=None, weight=None, cat_cols=None):
        if cat_cols is None:
            categorical_feature = None
        else:
            categorical_feature = np.nonzero(cat_cols)[0].tolist()
        if len(group) == len(y):
            group = format_group(group)
        dataset = lgb.Dataset(
            X, y, group=group, weight=weight, categorical_feature=categorical_feature, free_raw_data=False
        )
        return dataset
