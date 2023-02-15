# S2AFF - Semantic Scholar Affiliations Linker
This repository contains code that links raw affiliation strings to ROR ids.

This is a 3-stage model:
1. Raw affiliation strings are run through a NER transformer that parses them into main institute, child institute, and address components.
2. A retrieval index fetches top 100 candidates. This component has a MRR of about 0.91 on our gold data, with a recall@100 of 0.98.
3. A pairwise, feature-based LightGBM model does reranking of the top 100 candidates. This component has a precision@1 of about 0.96 and a MRR of 0.97.

The model has some heuristic rules for predicting "no ROR found". These are: (1) if the score of the top candidate is below 0.3 and (2) if the difference in the top score and the second top score is less than 0.2. These were found by empirical exploration. See `scripts/analyze_thresholds.ipynb` for more details.

## Installation
To install this package, run the following:

```bash
git clone git@github.com:allenai/S2AFF.git
cd S2AFF
conda create -y --name s2aff python==3.9.11
conda activate s2aff
pip install -e .
```

This installs the default CPU version of `pytorch`. If you want to install the GPU version, you'll have to
comment out `torch` in `requirements.in` and then install as per instructions here: https://pytorch.org/get-started/locally/

If you run into cryptic errors about GCC on macOS while installing the requirements, try this instead:
```bash
CFLAGS='-stdlib=libc++' pip install -e .
```

## Models Download

To get the models, run this command after the package is installed (from inside the `S2AFF` directory):  
```[Expected download size is about 3.4 GiB]```

`aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2aff-release data/`

## Updating the ROR Database
The ROR json database is stored in `data`. To update it you have to do the following:

1. Download the latest ROR database zip from https://zenodo.org/record/7574659
2. Unzip the json into the `data` directory.
3. In `s2aff/constants.py` there is a variable called `PATHS` with an entry for the `ror_data`. Update it to the new ROR json path
4. Run `scripts/update_openalex_works_counts.py` to get the latest works counts for each ROR id from OpenAlex.
5. [If you work at AI2] Upload the new ROR json and the `openalex_works_counts.csv` to `s3://ai2-s2-research-public/s2aff-release/` and delete the old ROR json from there.

## Performance on Gold Data
Included in this repository in the `data` directory is a dataset of some challenging raw affiliation strings that have been manually assigned
one or more ROR ids, and some of which have no correct RORs at all. Please see `data/gold_affiliations.csv`. The training split of this data has been used to tweak the first stage ROR indexer as well as train the second-stage LightGBM pairwise model.

Data sizes:

| split | n samples |
|:------|:----------|
| train |    1132|
|test  |   644 |
|val   |   588 |

### First-Stage Model Performance
The following values are obtained when combining training, validation and test sets:

- MRR: 0.908
- Recall@25: 0.975
- Recall@100: 0.983 <- reranking at this point
- Recall@250: 0.988
- Recall@500: 0.989
- Recall@1000: 0.992
- Recall@10000: 0.993
- Recall@100000: 0.994

See `scripts/evaluate_first_stage_ranker.py` for methodology.

### Second-Stage Model Performance
When reranking the top 100 candidates from the first stage model, the second stage model achieves the following:

- Train - MAP: 0.984, MRR: 0.989, Prec@1: 0.981
- Validation - MAP: 0.979, MRR: 0.984, Prec@1: 0.976
- Test - MAP: 0.968, MRR: 0.972, Prec@1: 0.957

See `scripts/train_lightgbm_ranker.py` for details. The above values are computed in lines 217 to 219.

### Speed and Memory
The model can do about 2 queries per second and will take up between 4 and 5 Gb of ram when fully loaded. If you use a GPU, the NER part will be a lot faster.

For 100 affiliation strings, speed breakdown is as follows:

- NER (CPU, batch size = 1): 17s
- NER (CPU, batch size = 8): 7s
- Stage 1 Candidates: 35s
- LightGBM Rerank top 100 (3 threads): 12s

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

## Why Not Rerank with BERT?
Because it's worse! We couldn't make `roberta-large` any better than what you see below (validation scores).
![image](https://user-images.githubusercontent.com/1874668/178345697-6af8311d-416e-4540-8489-6eb2eca4186c.png)


## TODO
S2AFF has not been trained to deal with non-English affiliations. It may work for some latin-character languages, but it may not.
If this model is deployed, a language detector should first be applied to the affiliation strings. Then, an analysis will need to be done
to determine which languages will be too error-prone.
