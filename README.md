# S2AFF - Semantic Scholar Affiliations Linker
This repository contains code that links raw affiliation strings to ROR ids.

## Installation
To install this package, run the following:

```bash
git clone git@github.com:allenai/S2AFF.git
cd S2AFF
conda create -y --name s2aff python==3.9.11
conda activate s2aff
pip install -r requirements.in
pip install -e .
```

Note that this will install a `torch` wihout GPU support. To install a GPU version, follow instructions here: https://pytorch.org/get-started/locally/

## Data

To obtain the training data, run this command after the package is installed (from inside the `S2AFF` directory):  
```[Expected download size is about 3.4 GiB]```

`aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2aff-release data/`

If you run into cryptic errors about GCC on macOS while installing the requirments, try this instead:
```bash
CFLAGS='-stdlib=libc++' pip install -r requirements.in
```

## Getting Started

Once you've downloaded all of the model files, you can get up and running as follows:
```python
from s2aff import S2AFF
from s2aff.ror import RORIndex
from s2aff.model import NERPredictor, PairwiseRORLightGBMReranker

# an example affiliation
raw_affiliation = "Department of Horse Racing, University of Washington, Seattle, WA 98115 USA"

# instantiate the ner model and the ror index. loading first time will take ~10-30s
ner_predictor = NERPredictor(use_cuda=False)  # defaults to models in the data directory
ror_index = RORIndex()  # ditto
pairwise_model = PairwiseRORLightGBMReranker(ror_index)

# the easiest way to use this package is with the `S2AFF` class.
s2aff = S2AFF(ner_predictor, ror_index, pairwise_model)
predictions = s2aff.predict([raw_affiliation])  # takes lists of strings
print(predictions)

# alternatively, you can also do it more manually
# first we get some fast-sh first-stage candidatges using the NERModel and RORIndex
candidates, scores = ror_index.get_candidates_from_raw_affiliation(raw_affiliation, ner_predictor)

# print out the top 5 candidates and scores from the first stage
# University of Washington is already first
for i, j in zip(candidates[:5], scores[:5]):
    print(ror_index.ror_dict[i]["name"], j)

# NOTE: ner_predictor should be batched if possible, as that will massively increase retrieval speed
# this is taken care of for you in the S2AFF class

# next, we rerank the top 100 candidates using the pairwise model
reranked_candidates, reranked_scores = pairwise_model.predict(raw_affiliation, candidates[:100], scores[:100])

# print out the top 5 candidates and scores from the second stage
# University of Washington is first with a high certainty
# but so is Seattle University
for i, j in zip(reranked_candidates[:5], reranked_scores[:5]):
    print(ror_index.ror_dict[i]["name"], j)
```
