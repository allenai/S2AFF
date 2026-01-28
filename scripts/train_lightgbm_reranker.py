"""
Train and evaluate the LightGBM reranker.

Measured Jan 26, 2026 (train/val/test):
Train - MAP: 0.946, MRR: 0.958, Prec@1: 0.931
Validation - MAP: 0.973, MRR: 0.981, Prec@1: 0.966
Test - MAP: 0.978, MRR: 0.981, Prec@1: 0.965
"""

import os
import pickle
import numpy as np
import sklearn
import lightgbm as lgb
import kenlm
from s2aff.ror import RORIndex
from s2aff.text import fix_text
from s2aff.consts import PATHS
from s2aff.lightgbm_helpers import lightgbmExperiment
from s2aff.features import (
    FEATURE_CONSTRAINTS,
    FEATURE_NAMES,
    build_query_context,
    make_lightgbm_features_with_query_context,
    parse_ror_entry_into_single_string_lightgbm,
)
from s2aff.data import load_gold_affiliation_annotations
from s2aff.model import NERPredictor, parse_ner_prediction
from s2aff.rust_backend import get_rust_backend, rust_available
from s2aff.flags import get_stage1_pipeline

FEATURE_NAMES = list(FEATURE_NAMES)

inds_to_check = [
    FEATURE_NAMES.index("names_frac_of_query_matched_in_text"),
    FEATURE_NAMES.index("acronyms_frac_of_query_matched_in_text"),
]

city_ind = FEATURE_NAMES.index("city_frac_of_query_matched_in_text")
stage_1_ind = FEATURE_NAMES.index("stage_1_score")
USE_RUST = rust_available()
REBUILD_FEATURES = os.getenv("S2AFF_REBUILD_LGBM_FEATURES", "").lower() in {"1", "true", "yes", "on"}
STAGE1_PIPELINE = get_stage1_pipeline()
USE_CUDA = os.getenv("S2AFF_USE_CUDA", "0").lower() in {"1", "true", "yes", "y"}
SKIP_SHAP = os.getenv("S2AFF_SKIP_SHAP", "1").lower() in {"1", "true", "yes", "on"}


def _build_pairwise_data_rust(
    df_gold,
    lm,
    ror_index,
    split_str="train",
    n_negatives=100,
    pos_weight=10,
    use_alt_pos=True,
):
    if split_str is not None:
        df_sub = df_gold[df_gold.split == split_str]
    else:
        df_sub = df_gold

    parsed = [parse_ner_prediction(pred, ror_index) for pred in df_sub.parsed_affiliations]
    mains = [main for main, _, _, _ in parsed]
    addresses = [address for _, _, address, _ in parsed]
    early_candidates_list = [early for _, _, _, early in parsed]

    if STAGE1_PIPELINE == "rust" and hasattr(
        ror_index, "get_candidates_from_main_affiliation_rust_batch"
    ):
        candidates_list, scores_list = ror_index.get_candidates_from_main_affiliation_rust_batch(
            mains, addresses, early_candidates_list
        )
    else:
        candidates_list = []
        scores_list = []
        for main, address, early_candidates in zip(mains, addresses, early_candidates_list):
            if STAGE1_PIPELINE == "rust" and hasattr(
                ror_index, "get_candidates_from_main_affiliation_rust"
            ):
                candidates, scores = ror_index.get_candidates_from_main_affiliation_rust(
                    main, address, early_candidates
                )
            else:
                candidates, scores = ror_index.get_candidates_from_main_affiliation(
                    main, address, early_candidates
                )
            candidates_list.append(candidates)
            scores_list.append(scores)

    ror_entry_cache = {}

    def get_ror_entry(ror_id):
        cached = ror_entry_cache.get(ror_id)
        if cached is not None:
            return cached
        value = parse_ror_entry_into_single_string_lightgbm(ror_id, ror_index)
        ror_entry_cache[ror_id] = value
        return value

    queries = []
    candidates_by_query = []
    scores_by_query = []
    labels_by_query = []
    weights_by_query = []
    rors_by_query = []
    qs_by_query = []
    group_ids = []

    outer_ind = 0
    for (row, parsed_tuple, candidates, scores) in zip(
        df_sub.itertuples(), parsed, candidates_list, scores_list
    ):
        main, child, address, early_candidates = parsed_tuple
        gold = set(row.labels)
        candidates_scores_map = {c: s for c, s in zip(candidates, scores)}
        candidates_positives = [(i, candidates_scores_map.get(i, 0.001)) for i in gold]
        candidates_negatives = [
            (i, s)
            for i, s in zip(candidates[:n_negatives], scores[:n_negatives])
            if i not in gold
        ]
        if candidates_negatives:
            candidates_negatives_ids, candidates_negatives_scores = zip(*candidates_negatives)
        else:
            candidates_negatives_ids, candidates_negatives_scores = [], []

        original_affiliation_fixed = fix_text(row.original_affiliation).lower().replace(",", "")
        main_fixed = fix_text(" ".join(main)).lower().replace(",", "")
        address_fixed = fix_text(" ".join(address)).lower().replace(",", "")

        def add_group(query_text, suffix=""):
            nonlocal outer_ind
            pos_ids = [i for i, _ in candidates_positives]
            pos_scores = [s for _, s in candidates_positives]
            neg_ids = list(candidates_negatives_ids)
            neg_scores = list(candidates_negatives_scores)

            candidates_group = pos_ids + neg_ids
            scores_group = pos_scores + neg_scores
            labels_group = [1] * len(pos_ids) + [0] * len(neg_ids)
            weights_group = [pos_weight] * len(pos_ids) + [1] * len(neg_ids)

            queries.append(query_text)
            candidates_by_query.append(candidates_group)
            scores_by_query.append(scores_group)
            labels_by_query.append(labels_group)
            weights_by_query.append(weights_group)
            rors_by_query.append(candidates_group)
            qs_by_query.append(query_text + suffix)
            group_ids.append(outer_ind)
            outer_ind += 1

        add_group(original_affiliation_fixed)

        if use_alt_pos:
            if main_fixed and main_fixed != original_affiliation_fixed:
                add_group(main_fixed, suffix=" [MAIN ONLY]")

            if address_fixed and (main_fixed + " " + address_fixed) != original_affiliation_fixed:
                add_group(main_fixed + " " + address_fixed, suffix=" [MAIN+ADDRESS]")

    query_contexts = [build_query_context(q) for q in queries]
    rust_backend = get_rust_backend(ror_index) if USE_RUST else None
    if rust_backend is not None:
        X_all, split_indices = rust_backend.build_lightgbm_features_with_query_context_batch(
            lm, query_contexts, candidates_by_query, get_ror_entry
        )
        if len(X_all) == 0:
            X_all = np.array([])
    else:
        X_all_list = []
        for query_context, candidates in zip(query_contexts, candidates_by_query):
            for candidate_id in candidates:
                ror_entry = get_ror_entry(candidate_id)
                x = make_lightgbm_features_with_query_context(query_context, ror_entry, lm)
                X_all_list.append(x)
        X_all = np.array(X_all_list)

    if len(X_all):
        flat_scores = [s for scores in scores_by_query for s in scores]
        if len(flat_scores) == len(X_all):
            X_all[:, -3] = flat_scores
            X_all[:, -2] = [int(s == -0.15) for s in flat_scores]
            X_all[:, -1] = [int(s == -0.1) for s in flat_scores]

    y = np.array([label for labels in labels_by_query for label in labels])
    w = np.array([weight for weights in weights_by_query for weight in weights], dtype=np.float32)
    group = np.array(
        [group_ids[i] for i in range(len(group_ids)) for _ in candidates_by_query[i]]
    )
    rors = np.array([ror for rors in rors_by_query for ror in rors])
    qs = np.array([qs for qs, rors in zip(qs_by_query, rors_by_query) for _ in rors])

    return X_all, y, w, group, rors, qs


def eval_lgb(y, X, scores, group, weight=0.05):
    has_no_match = X[:, inds_to_check].sum(1) == 0
    scores_adjusted = scores - weight * has_no_match
    scores_adjusted = scores_adjusted + weight * X[:, city_ind]
    APs = []
    RR = []
    Pat1 = []
    sorted_row_inds = np.sort(list(set(group)))
    for i in sorted_row_inds:
        flag = group == i
        p = scores_adjusted[flag]
        y_true = y[flag]
        APs.append(sklearn.metrics.average_precision_score(y_true, p))
        argsort = np.argsort(p)[::-1]  # largest to smallest
        y_true_argsort = y_true[argsort]  # labels are sorted according to probs
        first_index_of_label_1 = list(y_true_argsort).index(1) + 1  # where is the first appearance of label 1?
        RR.append(1.0 / first_index_of_label_1)  # reciprocal of that
        Pat1.append(first_index_of_label_1 == 1)  # is it the first position?

    results = {
        "MAP": np.mean(APs),
        "MRR": np.mean(RR),
        "Prec@1": np.mean(Pat1),
        "MAP_vec": APs,
        "MRR_vec": RR,
        "Prec@1_vec": Pat1,
        "group_vec": sorted_row_inds,
    }
    print({"MAP": np.mean(APs), "MRR": np.mean(RR), "Prec@1": np.mean(Pat1)})
    return results


"""
Step 1: get all available raw affiliations and their grid ids, map to ROR, and save to a file.
We will also save a raw file with all the lower-cased affiliations only. 

Step 2: Run kenlm in the command line like this:
~/kenlm/build/bin/lmplz -o 6 -S 100G --skip_symbols --prune 0 3 -T /tmp <data/raw_affiliations_lowercased.txt >data/raw_affiliations_lowercased.arma
~/kenlm/build/bin/build_binary -T /tmp data/raw_affiliations_lowercased.arma data/raw_affiliations_lowercased.binary

These two commands were run offline, and the steps are not reproducible in this repo
"""


"""
Step 3: LightGBM ranker requires featurization for pairs.
We can mostly use the featurization from the search project,
but there are some differences:
- don't need year, citations, oldness, special treatment for authors, special treatment for quotes
- some fields have multiple entries, so need to combine those somehow
"""

if REBUILD_FEATURES:
    if not USE_RUST:
        raise RuntimeError("S2AFF_REBUILD_LGBM_FEATURES requires S2AFF_PIPELINE=rust")
    print("Rebuilding lightgbm_gold_features with rust-enabled pipeline...")
    df_gold = load_gold_affiliation_annotations()
    ner_predictor = NERPredictor(use_cuda=USE_CUDA)
    parsed_affiliations = ner_predictor.predict(df_gold.loc[:, "original_affiliation"].values)
    df_gold.loc[:, "parsed_affiliations"] = parsed_affiliations
    ner_predictor.delete_model()

    lm = kenlm.LanguageModel(PATHS["kenlm_model"])
    ror_index = RORIndex()
    if get_rust_backend(ror_index) is None:
        raise RuntimeError("Rust backend not available; cannot rebuild features")

    X_train, y_train, w_train, group_train, rors_train, qs_train = _build_pairwise_data_rust(
        df_gold,
        lm,
        ror_index,
        split_str="train",
        n_negatives=100,
        use_alt_pos=True,
        pos_weight=10,
    )
    X_val, y_val, w_val, group_val, rors_val, qs_val = _build_pairwise_data_rust(
        df_gold,
        lm,
        ror_index,
        split_str="val",
        n_negatives=100,
        use_alt_pos=False,
        pos_weight=10,
    )
    X_test, y_test, w_test, group_test, rors_test, qs_test = _build_pairwise_data_rust(
        df_gold,
        lm,
        ror_index,
        split_str="test",
        n_negatives=100,
        use_alt_pos=False,
        pos_weight=10,
    )

    with open(PATHS["lightgbm_gold_features"], "wb") as f:
        pickle.dump(
            [
                (X_train, y_train, w_train, group_train, rors_train, qs_train),
                (X_val, y_val, w_val, group_val, rors_val, qs_val),
                (X_test, y_test, w_test, group_test, rors_test, qs_test),
            ],
            f,
        )

with open(PATHS["lightgbm_gold_features"], "rb") as f:
    [
        (X_train, y_train, w_train, group_train, rors_train, qs_train),
        (X_val, y_val, w_val, group_val, rors_val, qs_val),
        (X_test, y_test, w_test, group_test, rors_test, qs_test),
    ] = pickle.load(f)

train_main_flag = np.array([i.endswith("[MAIN ONLY]") for i in qs_train])
train_main_address_flag = np.array([i.endswith("[MAIN+ADDRESS]") for i in qs_train])
train_augmented_flag = train_main_flag | train_main_address_flag

w_train = w_train.astype(np.float32)
w_train[train_main_flag] = w_train[train_main_flag] / 10
w_train[train_main_address_flag] = w_train[train_main_address_flag] / 2

lgb_exp = lightgbmExperiment(
    learning_task="ranking",
    metric="rank_xendcg",
    eval_metric="map",
    n_estimators=500,
    hyperopt_evals=50,
    device="cpu",
    eval_at="1,5,25,100",
    tree_learner="data",
    random_seed=7,
    threads=64,
)

# lgb_exp.constant_params["monotone_constraints"] = FEATURE_CONSTRAINTS  # generally makes things worse
lgb_exp.constant_params["feature_pre_filter"] = False  # prevents some obscure errors

test_ndcgs = lgb_exp.run(
    X_train,
    y_train,
    group_train,
    w_train,
    X_val,
    y_val,
    group_val,
    w_val,
    X_test,
    y_test,
    group_test,
    w_test,
)

# load the best model seen so far
# lgb_exp.bst = lgb.Booster(model_file=PATHS["lightgbm_model"])

# train_dmatrix = lgb_exp.convert_data(X_train, y_train, group_train, w_train)
# val_dmatrix = lgb_exp.convert_data(X_val, y_val, group_val, w_val)
# lgb_exp.bst, _ = lgb_exp.fit(lgb_exp.best_params, train_dmatrix, val_dmatrix, 500, 8)

train_results = eval_lgb(y_train, X_train, lgb_exp.bst.predict(X_train), group_train)
val_results = eval_lgb(y_val, X_val, lgb_exp.bst.predict(X_val), group_val)
test_results = eval_lgb(y_test, X_test, lgb_exp.bst.predict(X_test), group_test)

# save model
lgb_exp.bst.save_model(PATHS["lightgbm_model"])

# optional SHAP analysis (skipped by default)
if SKIP_SHAP:
    print("Skipping SHAP analysis (set S2AFF_SKIP_SHAP=0 to enable).")
    raise SystemExit(0)

# look at results manually
import shap


ror_index = RORIndex()


with open(PATHS["lightgbm_gold_features"], "rb") as f:
    [
        (X_train, y_train, w_train, group_train, rors_train, qs_train),
        (X_val, y_val, w_val, group_val, rors_val, qs_val),
        (X_test, y_test, w_test, group_test, rors_test, qs_test),
    ] = pickle.load(f)


train_main_flag = np.array([i.endswith("[MAIN ONLY]") for i in qs_train])
train_main_address_flag = np.array([i.endswith("[MAIN+ADDRESS]") for i in qs_train])
train_augmented_flag = train_main_flag | train_main_address_flag

X_train = X_train[~train_augmented_flag]
y_train = y_train[~train_augmented_flag]
w_train = w_train[~train_augmented_flag]
group_train = group_train[~train_augmented_flag]
rors_train = rors_train[~train_augmented_flag]
qs_train = qs_train[~train_augmented_flag]

scores_train = lgb_exp.bst.predict(X_train)
scores_test = lgb_exp.bst.predict(X_test)
scores_val = lgb_exp.bst.predict(X_val)

lgb_exp.bst.params["objective"] = "rank_xendcg"
explainer = shap.Explainer(lgb_exp.bst)
shap_values_train = explainer.shap_values(X_train)
shap_values_val = explainer.shap_values(X_val)


def print_val(i, split="train"):
    if split == "train":
        flag = group_train == i
        q = qs_train[flag][0]
        rors = rors_train[flag]
        p = scores_train[flag]
        y_true = y_train[flag]
        x = X_train[flag]
    elif split == "val":
        flag = group_val == i
        q = qs_val[flag][0]
        rors = rors_val[flag]
        p = scores_val[flag]
        y_true = y_val[flag]
        x = X_val[flag]
    elif split == "test":
        flag = group_test == i
        q = qs_test[flag][0]
        rors = rors_test[flag]
        p = scores_test[flag]
        y_true = y_test[flag]
        x = X_test[flag]

    argsort = np.argsort(p)[::-1]  # largest to smallest
    rors = rors[argsort]
    p = p[argsort]
    y_true = y_true[argsort]
    x = x[argsort]

    print("Affiliation string:", q)
    print("----------")
    print("Correct ones:")
    for label, ror, prob in zip(y_true, rors, p):
        if label == 1:
            print(np.round(prob, 3), ror, parse_ror_entry_into_single_string_lightgbm(ror, ror_index))

    print("----------")
    print("Incorrect ones:")
    counter = 0
    for label, ror, prob in zip(y_true, rors, p):
        if label == 0:
            print(np.round(prob, 3), ror, parse_ror_entry_into_single_string_lightgbm(ror, ror_index))
            counter += 1
        if counter == 5:
            break


train_results = eval_lgb(y_train, X_train, lgb_exp.bst.predict(X_train), group_train)
val_results = eval_lgb(y_val, X_val, lgb_exp.bst.predict(X_val), group_val)
test_results = eval_lgb(y_test, X_test, lgb_exp.bst.predict(X_test), group_test)

wrong_ones_train = train_results["group_vec"][np.array(train_results["Prec@1_vec"]) == False]
wrong_ones_val = val_results["group_vec"][np.array(val_results["Prec@1_vec"]) == False]
wrong_ones_test = test_results["group_vec"][np.array(test_results["Prec@1_vec"]) == False]

for i in wrong_ones_train:
    print_val(i, "train")
    print("\n=====================================================\n")


# i = wrong_ones_val[-2]
# flag = group_val == i
# q = qs_val[flag][0]
# rors = rors_val[flag]
# p = scores_val[flag]
# y_true = y_val[flag]
# x = X_val[flag]
# shap_v = shap_values_val[flag]

i = wrong_ones_train[-5]
flag = group_train == i
q = qs_train[flag][0]
rors = rors_train[flag]
p = scores_train[flag]
y_true = y_train[flag]
x = X_train[flag]
shap_v = shap_values_train[flag]


argsort = np.argsort(p)[::-1]  # largest to smallest
rors = rors[argsort]
p = p[argsort]
y_true = y_true[argsort]
x = np.round(x[argsort], 3)
shap_v = np.round(shap_v[argsort], 3)

first_zero = np.where(y_true == 0)[0][0]
first_one = np.where(y_true == 1)[0][0]

print(q)
parse_ror_entry_into_single_string_lightgbm(rors[first_zero], ror_index)
list(zip(FEATURE_NAMES, x[first_zero], shap_v[first_zero]))
print()
parse_ror_entry_into_single_string_lightgbm(rors[first_one], ror_index)
list(zip(FEATURE_NAMES, x[first_one], shap_v[first_one]))
