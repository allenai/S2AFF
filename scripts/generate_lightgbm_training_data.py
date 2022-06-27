import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change to your reserved GPU
import pickle
import kenlm
from random import sample
import numpy as np
from s2aff.text import fix_text
from s2aff.consts import PATHS
from s2aff.ror import RORIndex
from s2aff.features import make_lightgbm_features, parse_ror_entry_into_single_string_lightgbm
from s2aff.model import NERPredictor, parse_ner_prediction
from s2aff.data import load_gold_affiliation_annotations


lm = kenlm.LanguageModel(PATHS["kenlm_model"])
ror_index = RORIndex()


def make_pairwise_data(df_gold, lm, ror_index, split_str="train", n_negatives=100, pos_weight=10, use_alt_pos=True):
    X = []
    y = []
    w = []
    rors = []
    qs = []
    group = []

    def make_single_pair(text, ror_id, label, weight, ind, score, suffix=""):
        x = make_lightgbm_features(text, parse_ror_entry_into_single_string_lightgbm(ror_id, ror_index), lm)
        x[-3:] = [score, int(score == -0.15), int(score == -0.1)]
        X.append(x)
        y.append(label)
        w.append(weight)
        group.append(ind)
        rors.append(ror_id)
        qs.append(text + suffix)

    outer_ind = 0
    if split_str is not None:
        df_sub = df_gold[df_gold.split == split_str]
    else:
        df_sub = df_gold
    for _, row in df_sub.iterrows():
        main, child, address, early_candidates = parse_ner_prediction(row.parsed_affiliations, ror_index)
        candidates, scores = ror_index.get_candidates_from_main_affiliation(main, address, early_candidates)
        gold = set(row.labels)
        candidates_positives_dict = {c: s for c, s in zip(candidates, scores)}
        candidates_positives = [(i, candidates_positives_dict.get(i, 0.001)) for i in gold]
        # top 100 candidates are what we will rerank in prod, so sampling those here
        candidates_negatives = [(i, s) for i, s in zip(candidates[:n_negatives], scores[:n_negatives]) if i not in gold]
        if len(candidates_negatives) > 0:
            candidates_negatives, scores_negatives = zip(*candidates_negatives)
        else:
            candidates_negatives, scores_negatives = [], []
        original_affiliation_fixed = fix_text(row.original_affiliation).lower().replace(",", "")
        main = fix_text(" ".join(main)).lower().replace(",", "")
        address = fix_text(" ".join(address)).lower().replace(",", "")

        for gold_ror_id, positive_score in candidates_positives:
            make_single_pair(original_affiliation_fixed, gold_ror_id, 1, pos_weight, outer_ind, positive_score)

        for neg_ror_id, negative_score in zip(candidates_negatives, scores_negatives):
            make_single_pair(original_affiliation_fixed, neg_ror_id, 0, 1, outer_ind, negative_score)

        outer_ind += 1

        if use_alt_pos:

            if main != original_affiliation_fixed:

                for gold_ror_id, positive_score in candidates_positives:
                    make_single_pair(main, gold_ror_id, 1, pos_weight, outer_ind, positive_score, suffix=" [MAIN ONLY]")

                for neg_ror_id, negative_score in zip(candidates_negatives, scores_negatives):
                    make_single_pair(main, neg_ror_id, 0, 1, outer_ind, negative_score, suffix=" [MAIN ONLY]")

                outer_ind += 1

            if address != "" and (main + " " + address) != original_affiliation_fixed:
                for gold_ror_id, positive_score in candidates_positives:
                    make_single_pair(
                        main + " " + address,
                        gold_ror_id,
                        1,
                        pos_weight,
                        outer_ind,
                        positive_score,
                        suffix=" [MAIN+ADDRESS]",
                    )

                for neg_ror_id, negative_score in zip(candidates_negatives, scores_negatives):
                    make_single_pair(
                        main + " " + address, neg_ror_id, 0, 1, outer_ind, negative_score, suffix=" [MAIN+ADDRESS]"
                    )

                outer_ind += 1

    return np.array(X), np.array(y), np.array(w), np.array(group), np.array(rors), np.array(qs)


"""
Training set for gold data
"""
df_gold = load_gold_affiliation_annotations()

# should do this batched
ner_predictor = NERPredictor(use_cuda=False)
parsed_affiliations = ner_predictor.predict(df_gold.loc[:, "original_affiliation"].values)
df_gold.loc[:, "parsed_affiliations"] = parsed_affiliations
ner_predictor.delete_model()  # clean up to make space

X_train, y_train, w_train, group_train, rors_train, qs_train = make_pairwise_data(
    df_gold,
    lm,
    ror_index,
    split_str="train",
    n_negatives=100,
    use_alt_pos=True,
    pos_weight=10,
)
X_val, y_val, w_val, group_val, rors_val, qs_val = make_pairwise_data(
    df_gold,
    lm,
    ror_index,
    split_str="val",
    n_negatives=100,
    use_alt_pos=False,
    pos_weight=10,
)
X_test, y_test, w_test, group_test, rors_test, qs_test = make_pairwise_data(
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

# same but do it on the stuff with no ROR at all
df_gold = load_gold_affiliation_annotations(keep_non_ror_gold=True)
keep_flag = df_gold.labels.apply(lambda x: any(["ror.org" not in i for i in x]))
df_gold = df_gold[keep_flag]

# should do this batched
ner_predictor = NERPredictor(use_cuda=False)
parsed_affiliations = ner_predictor.predict(df_gold.loc[:, "original_affiliation"].values)
df_gold.loc[:, "parsed_affiliations"] = parsed_affiliations
ner_predictor.delete_model()  # clean up to make space

X_none, y_none, w_none, group_none, rors_none, qs_none = make_pairwise_data(
    df_gold,
    lm,
    ror_index,
    split_str=None,
    n_negatives=100,
    use_alt_pos=False,
    pos_weight=10,
)

with open(PATHS["lightgbm_gold_no_ror_features"], "wb") as f:
    pickle.dump(
        (X_none, y_none, w_none, group_none, rors_none, qs_none),
        f,
    )
