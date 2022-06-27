"""
How good is the first-stage ranker? Find out by running this script!

train + val + test:
MRR: 0.9081398880736107
Recall@25: 0.9751609935602575
Recall@100: 0.9829806807727691 <- reranking at this point (as of now)
Recall@250: 0.9875804967801288
Recall@500: 0.9894204231830727
Recall@1000: 0.9917203311867525
Recall@10000: 0.9931002759889604
Recall@100000: 0.9940202391904324
"""

import numpy as np
from s2aff.data import load_gold_affiliation_annotations
from s2aff.ror import RORIndex, index_min
from s2aff.model import NERPredictor, parse_ner_prediction


USE_CUDA = False

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

    # eval
    i = 0
    results = []
    for _, row in df.iloc[i:].iterrows():
        main, child, address, early_candidates = parse_ner_prediction(ner_predictions[i], ror_index)
        candidates, scores = ror_index.get_candidates_from_main_affiliation(main, address, early_candidates)
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
        mrr = np.mean([result["reciprocal_rank"] for result in results])
        if verbose:
            print(f"{i} | MRR: {mrr}")
        i += 1

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


metrics, results, ror_index = run_stage_1_ranker(verbose=True)

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
