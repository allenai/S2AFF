import pickle
import numpy as np
import sklearn
import lightgbm as lgb
from s2aff.ror import RORIndex
from s2aff.text import fix_text
from s2aff.consts import PATHS
from s2aff.lightgbm_helpers import lightgbmExperiment
from s2aff.features import FEATURE_CONSTRAINTS, FEATURE_NAMES

FEATURE_NAMES = list(FEATURE_NAMES)

inds_to_check = [
    FEATURE_NAMES.index("names_frac_of_query_matched_in_text"),
    FEATURE_NAMES.index("acronyms_frac_of_query_matched_in_text"),
]

city_ind = FEATURE_NAMES.index("city_frac_of_query_matched_in_text")
stage_1_ind = FEATURE_NAMES.index("stage_1_score")


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

# look at results manually
import shap
from s2aff.features import parse_ror_entry_into_single_string_lightgbm

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
