"""
How long does this all take?

Times below are all run on s2-server2 on the CPU. 
"""
from time import time
import os, psutil
from s2aff.consts import PATHS
from s2aff.ror import RORIndex
from s2aff.model import NERPredictor, PairwiseRORLightGBMReranker, parse_ner_prediction
from s2aff.data import load_gold_affiliation_annotations
from s2aff.text import fix_text
from blingfire import text_to_words

USE_CUDA = False

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
candidates_and_scores = []
start = time()
for main, child, address, early_candidates in parsed_tuples:
    candidates, scores = ror_index.get_candidates_from_main_affiliation(main, address, early_candidates)
    candidates_and_scores.append((candidates, scores))
end = time()
print(f"Stage 1 Candidates: {end - start}")

# lightgbm: 12.6s for top 100
raw_affiliations = df.original_affiliation.values[-100:].tolist()
start = time()
for raw_affiliation, (candidates, scores) in zip(raw_affiliations, candidates_and_scores):
    reranked_candidates, reranked_scores = pairwise_model.predict(raw_affiliation, candidates[:100], scores[:100])
end = time()
print(f"LightGBM: {end - start}")

print(
    "Total memory footprint after running everything (in MB): ",
    psutil.Process(os.getpid()).memory_info().rss / 1024**2,
)
