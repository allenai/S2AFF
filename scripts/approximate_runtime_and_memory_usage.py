"""
How long does this all take?

Times below are all run on s2-server2 on the CPU. 
"""
from time import time
import os, psutil
from s2aff.consts import PATHS
from s2aff.flags import get_stage1_variant, get_stage2_variant
from s2aff.ror import RORIndex
from s2aff.model import NERPredictor, PairwiseRORLightGBMReranker, parse_ner_prediction
from s2aff.data import load_gold_affiliation_annotations
from s2aff.text import fix_text
from blingfire import text_to_words

USE_CUDA = os.getenv("S2AFF_USE_CUDA", "0").lower() in {"1", "true", "yes", "y"}
STAGE1_VARIANT = get_stage1_variant()
STAGE2_VARIANT = get_stage2_variant()

print(
    "Total memory footprint before loading various artifacts (in MB): ",
    psutil.Process(os.getpid()).memory_info().rss / 1024**2,
)

# fixed costs: load everything
ner_predictor = NERPredictor(use_cuda=USE_CUDA)
print(
    "Total memory footprint after loading NER (in MB): ",
    psutil.Process(os.getpid()).memory_info().rss / 1024**2,
)
ror_index = RORIndex()
print(
    "Total memory footprint after loading inverted index (in MB): ",
    psutil.Process(os.getpid()).memory_info().rss / 1024**2,
)
print(
    "Total memory footprint after loading lightgbm (in MB): ",
    psutil.Process(os.getpid()).memory_info().rss / 1024**2,
)
pairwise_model = PairwiseRORLightGBMReranker(ror_index, num_threads=0)  # 0 is default open threads

# what is the total memory footprint at this point?
print(
    "Total memory footprint after loading everything (in MB): ",
    psutil.Process(os.getpid()).memory_info().rss / 1024**2,
)


# some data to test on - 100 samples
df = load_gold_affiliation_annotations()
texts = df.original_affiliation.values[-100:].tolist()

# prepare for NER
# texts = [text_to_words(fix_text(text)) for text in texts]

print("Longest test affiliation (# characters):", max(map(len, texts)))
print("Shortest test affiliation (# characters):", min(map(len, texts)))

# NER time: 9.02 seconds on CPU
# there are no batch size efficiency gains above 8

start = time()
preds = ner_predictor.predict(texts)
end = time()
print(f"NER on CPU: {end - start}")

# parsing is negligible: 0.08 seconds
parsed_tuples = []
start = time()
for i in preds:
    main, child, address, early_candidates = parse_ner_prediction(i, ror_index)
    parsed_tuples.append((main, child, address, early_candidates))
end = time()
print(f"Parsing NER Preds: {end - start}")

# stage 1 ranker time: 27s
start = time()
candidates_and_scores = []
if STAGE1_VARIANT == "v7" and hasattr(ror_index, "get_candidates_from_main_affiliation_v7_batch"):
    mains = [main for main, _, _, _ in parsed_tuples]
    addresses = [address for _, _, address, _ in parsed_tuples]
    early_candidates_list = [early_candidates for _, _, _, early_candidates in parsed_tuples]
    candidates_list, scores_list = ror_index.get_candidates_from_main_affiliation_v7_batch(
        mains, addresses, early_candidates_list
    )
    candidates_and_scores = list(zip(candidates_list, scores_list))
else:
    for main, child, address, early_candidates in parsed_tuples:
        if STAGE1_VARIANT == "v7" and hasattr(ror_index, "get_candidates_from_main_affiliation_v7"):
            candidates, scores = ror_index.get_candidates_from_main_affiliation_v7(
                main, address, early_candidates
            )
        else:
            candidates, scores = ror_index.get_candidates_from_main_affiliation_v1(
                main, address, early_candidates
            )
        candidates_and_scores.append((candidates, scores))
end = time()
print(f"Stage 1 Candidates: {end - start}")

# lightgbm: 12.6s for top 100
raw_affiliations = df.original_affiliation.values[-100:].tolist()
start = time()
if STAGE2_VARIANT == "v3" and hasattr(pairwise_model, "batch_predict_v3"):
    batch_candidates = [c[:100] for c, _ in candidates_and_scores]
    batch_scores = [s[:100] for _, s in candidates_and_scores]
    pairwise_model.batch_predict_v3(raw_affiliations, batch_candidates, batch_scores)
else:
    for raw_affiliation, (candidates, scores) in zip(raw_affiliations, candidates_and_scores):
        if STAGE2_VARIANT == "v3" and hasattr(pairwise_model, "predict_v3"):
            reranked_candidates, reranked_scores = pairwise_model.predict_v3(
                raw_affiliation, candidates[:100], scores[:100]
            )
        else:
            reranked_candidates, reranked_scores = pairwise_model.predict(
                raw_affiliation, candidates[:100], scores[:100]
            )
end = time()
print(f"LightGBM: {end - start}")

print(
    "Total memory footprint after running everything (in MB): ",
    psutil.Process(os.getpid()).memory_info().rss / 1024**2,
)
