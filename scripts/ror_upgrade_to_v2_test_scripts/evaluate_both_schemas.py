"""
Evaluate first-stage ranker for both ROR v1 and v2 schemas.
Runs NER prediction only once (expensive), then tests both schema versions.
Caches NER predictions to avoid re-running.
"""

import numpy as np
import pickle
import os
from s2aff.data import load_gold_affiliation_annotations
from s2aff.ror import RORIndex, index_min
from s2aff.model import NERPredictor, parse_ner_prediction
from s2aff.consts import PROJECT_ROOT_PATH


USE_CUDA = False
NER_CACHE_FILE = os.path.join(PROJECT_ROOT_PATH, "data", "ner_predictions_cache.pkl")

print("Loading gold affiliation annotations...")
df = load_gold_affiliation_annotations()

print(f"Loaded {len(df)} annotations")
print(f"  Train: {sum(df.split == 'train')}")
print(f"  Val: {sum(df.split == 'val')}")
print(f"  Test: {sum(df.split == 'test')}")

# Try to load cached NER predictions
if os.path.exists(NER_CACHE_FILE):
    print(f"\nLoading cached NER predictions from {NER_CACHE_FILE}...")
    with open(NER_CACHE_FILE, 'rb') as f:
        ner_predictions = pickle.load(f)
    print(f"Loaded {len(ner_predictions)} cached predictions")
else:
    # Run NER predictions ONCE (expensive)
    print("\nRunning NER predictions (this takes time)...")
    ner_predictor = NERPredictor(use_cuda=USE_CUDA)
    ner_predictions = ner_predictor.predict(df.original_affiliation.values)
    print(f"NER predictions complete: {len(ner_predictions)} predictions")

    # Cache the predictions
    print(f"Caching predictions to {NER_CACHE_FILE}...")
    with open(NER_CACHE_FILE, 'wb') as f:
        pickle.dump(ner_predictions, f)
    print("Cache saved!")


def evaluate_with_ror_file(ror_file_path, label):
    """Evaluate first-stage ranker with a specific ROR data file."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {label}")
    print(f"ROR file: {ror_file_path}")
    print(f"{'='*70}")

    # Load ROR index with specified file
    print("Loading ROR index...")
    ror_index = RORIndex(
        ror_data_path=ror_file_path,
        word_multiplier=0.3,
        ns={3},
        max_intersection_denominator=True,
        use_prob_weights=True,
        insert_early_candidates_ind=1,
    )

    print(f"ROR index loaded: {len(ror_index.ror_dict)} records")

    # Run evaluation
    print("\nEvaluating first-stage ranker...")

    topks = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 10000, 100000]
    recalls = {k: [] for k in topks}
    reciprocal_ranks = []

    for i, (row, ner_pred) in enumerate(zip(df.itertuples(), ner_predictions)):
        if i > 0 and i % 500 == 0:
            print(f"  Processed {i}/{len(df)} ({100*i/len(df):.1f}%)")

        main, child, address, early_candidates = parse_ner_prediction(ner_pred, ror_index)
        candidates, scores = ror_index.get_candidates_from_main_affiliation(main, address, early_candidates)

        # Find gold ROR position (row.labels is a list of gold ROR IDs)
        rank_of_gold = index_min(candidates, row.labels)

        if rank_of_gold < 100000:  # Found (index_min returns 100000 if not found)
            rank = rank_of_gold + 1  # Convert to 1-indexed
            reciprocal_ranks.append(1.0 / rank)

            for k in topks:
                recalls[k].append(1 if rank <= k else 0)
        else:
            # Not found at all
            reciprocal_ranks.append(0.0)
            for k in topks:
                recalls[k].append(0)

    # Calculate metrics
    mrr = np.mean(reciprocal_ranks)

    print(f"\n{label} RESULTS:")
    print(f"  MRR: {mrr:.6f}")
    for k in topks:
        recall = np.mean(recalls[k])
        print(f"  Recall@{k}: {recall:.6f}")

    return {
        'mrr': mrr,
        'recalls': {k: np.mean(recalls[k]) for k in topks},
        'label': label,
        'ror_file': ror_file_path
    }


# Evaluate both versions
v1_file = os.path.join(PROJECT_ROOT_PATH, "data", "v1.73-2025-10-28-ror-data.json")
v2_file = os.path.join(PROJECT_ROOT_PATH, "data", "v1.73-2025-10-28-ror-data_schema_v2.json")

results_v1 = evaluate_with_ror_file(v1_file, "ROR v1 Schema")
results_v2 = evaluate_with_ror_file(v2_file, "ROR v2 Schema (after coercion)")


# Compare results
print(f"\n{'='*70}")
print("COMPARISON SUMMARY")
print(f"{'='*70}")

print(f"\nMRR:")
print(f"  v1: {results_v1['mrr']:.6f}")
print(f"  v2: {results_v2['mrr']:.6f}")
print(f"  Δ:  {results_v2['mrr'] - results_v1['mrr']:.6f} ({100*(results_v2['mrr'] - results_v1['mrr'])/results_v1['mrr']:.2f}%)")

print(f"\nRecall metrics:")
for k in [1, 5, 10, 25, 50, 100, 250, 500, 1000]:
    r1 = results_v1['recalls'][k]
    r2 = results_v2['recalls'][k]
    diff = r2 - r1
    pct_change = 100 * diff / r1 if r1 > 0 else 0
    print(f"  Recall@{k:>6}: v1={r1:.6f}, v2={r2:.6f}, Δ={diff:+.6f} ({pct_change:+.2f}%)")

print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
print("If v2 results are similar or better, the schema migration is successful.")
print("Small differences (<1%) are expected due to richer location data in v2.")
