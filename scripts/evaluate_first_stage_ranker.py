"""
How good is the first-stage ranker? Find out by running this script!

Measured Jan 26, 2026 (train + val + test):
MRR: 0.897
Recall@25: 0.971
Recall@100: 0.984 <- reranking at this point (as of now)
Recall@250: 0.986
Recall@500: 0.988
Recall@1000: 0.990
Recall@10000: 0.992
Recall@100000: 0.993
"""

import os
import numpy as np
from s2aff.data import load_gold_affiliation_annotations
from s2aff.flags import get_stage1_variant
from s2aff.ror import RORIndex, index_min
from s2aff.model import NERPredictor, parse_ner_prediction


USE_CUDA = True
STAGE1_VARIANT = get_stage1_variant()
VERBOSE = os.getenv("S2AFF_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}

# load the gold, and subset to training data
df = load_gold_affiliation_annotations()
# df = df[df.split == "train"]

# load the ROR index and NERPredictor
ner_predictor = NERPredictor(use_cuda=USE_CUDA)  # defaults to models in the data directory

# get NER predictions for a bunch
ner_predictions = ner_predictor.predict(df.original_affiliation.values)


def run_stage_1_ranker(
    word_multiplier=0.3,
    ns={3},
    max_intersection_denominator=True,
    use_prob_weights=True,
    insert_early_candidates_ind=1,
    verbose=False,
):
    # load ror index
    ror_index = RORIndex(
        word_multiplier=word_multiplier,
        ns=ns,
        max_intersection_denominator=max_intersection_denominator,
        use_prob_weights=use_prob_weights,
        insert_early_candidates_ind=insert_early_candidates_ind,
    )

    parsed = [parse_ner_prediction(pred, ror_index) for pred in ner_predictions]
    if STAGE1_VARIANT == "v7" and hasattr(ror_index, "get_candidates_from_main_affiliation_v7_batch"):
        mains = [main for main, _, _, _ in parsed]
        addresses = [address for _, _, address, _ in parsed]
        early_candidates_list = [early_candidates for _, _, _, early_candidates in parsed]
        candidates_list, scores_list = ror_index.get_candidates_from_main_affiliation_v7_batch(
            mains, addresses, early_candidates_list
        )
    else:
        candidates_list = []
        scores_list = []
        for main, child, address, early_candidates in parsed:
            if STAGE1_VARIANT == "v7" and hasattr(ror_index, "get_candidates_from_main_affiliation_v7"):
                candidates, scores = ror_index.get_candidates_from_main_affiliation_v7(main, address, early_candidates)
            else:
                candidates, scores = ror_index.get_candidates_from_main_affiliation(main, address, early_candidates)
            candidates_list.append(candidates)
            scores_list.append(scores)

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        main, child, address, early_candidates = parsed[i]
        candidates = candidates_list[i]
        scores = scores_list[i]
        gold_names = [ror_index.ror_dict[i]["name"] for i in row.labels]
        candidates_names = [ror_index.ror_dict[i]["name"] for i in candidates]
        rank_of_gold = index_min(candidates, row.labels)  # location of the gold in candidates
        result = {
            "original_affiliation": row.original_affiliation,
            "main": main,
            "child": child,
            "address": address,
            "early_candidates": early_candidates,
            "candidates": candidates,
            "candidate_names": candidates_names,
            "gold": row.labels,
            "gold_names": gold_names,
            "candidate_scores": scores,
            "gold_rank": rank_of_gold + 1,
            "reciprocal_rank": 1 / (rank_of_gold + 1),
        }

        results.append(result)
        if verbose:
            mrr = np.mean([result["reciprocal_rank"] for result in results])
            print(f"{i} | MRR: {mrr}")

    # aggregated scores
    mrr = np.mean([result["reciprocal_rank"] for result in results])

    metrics = {"mrr": mrr}
    if verbose:
        print(f"MRR: {mrr}")

    # recalls at various values
    for r_k in [25, 100, 250, 500, 1000, 10000, 100000]:
        r_k_recall = np.mean([result["gold_rank"] <= r_k for result in results])
        metrics[f"recall@{r_k}"] = r_k_recall
        if verbose:
            print(f"Recall@{r_k}: {r_k_recall}")

    return metrics, results, ror_index


metrics, results, ror_index = run_stage_1_ranker(verbose=VERBOSE)

for k, v in metrics.items():
    print(f"{k}: {v}")

# find all the ones that don't have gold_rank == 1
results_bad = []
results_failed_recall = []
for result in results:
    if result["gold_rank"] != 1:
        results_bad.append(result)
    if result["gold_rank"] > 100:
        results_failed_recall.append(result)

for result in results_failed_recall:
    print("Original_affiliation:", result["original_affiliation"])
    print("Main/Child/Address", f"{result['main']} / {result['child']} / {result['address']}")
    print("Gold:", result["gold_names"])
    print("-------------------------------------------------")


# all_metrics = {}
# for word_multiplier in [0.3]:
#     for insert_early_candidates_ind in [0, 1, 2, 3, 4, 5, None]:
#         key = (word_multiplier, insert_early_candidates_ind)
#         if key not in all_metrics:
#             all_metrics[key] = run_stage_1_ranker(
#                 word_multiplier=word_multiplier, insert_early_candidates_ind=insert_early_candidates_ind, verbose=False
#             )
#         print(f"{key} | MRR: {all_metrics[key][0]['mrr']} | Recall@100: {all_metrics[key][0]['recall@100']}")

# for key in all_metrics.keys():
#     print(f"{key} | MRR: {all_metrics[key][0]['mrr']} | Recall@100: {all_metrics[key][0]['recall@100']}")
