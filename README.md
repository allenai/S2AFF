# S2AFF - Semantic Scholar Affiliations Linker
This repository contains code that links raw affiliation strings to ROR ids.

This is a 3-stage model:
1. Raw affiliation strings are run through a NER transformer that parses them into main institute, child institute, and address components.
2. A retrieval index fetches top 100 candidates. This component has a MRR of about 0.91 on our gold data, with a recall@100 of 0.98.
3. A pairwise, feature-based LightGBM model does reranking of the top 100 candidates. This component has a precision@1 of about 0.96 and a MRR of 0.97.

The model has some heuristic rules for predicting "no ROR found". These are: (1) if the score of the top candidate is below 0.3 and (2) if the difference in the top score and the second top score is less than 0.2. These were found by empirical exploration. See `scripts/analyze_thresholds.ipynb` for more details.

## Installation
To install this package, run the following (uv):
```
git clone git@github.com:allenai/S2AFF.git
cd S2AFF
uv python install 3.11.13
uv venv --python 3.11.13

# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1
# Windows (cmd):
# .\.venv\Scripts\activate.bat

uv pip install -e .
```


PyTorch installs via the CUDA wheel index configured in `pyproject.toml`, so `uv pip install -e .` will pull
GPU-enabled wheels when available and fall back to CPU-only wheels otherwise. If you need a specific CUDA build,
install `torch` explicitly before running the editable install.


## Optional but Recommended: Rust Acceleration
Rust is optional. If installed, it becomes the default path and S2AFF falls back to the Python pipeline when Rust is unavailable.

**Build requirements**
- Rust toolchain (rustup)
- `maturin`

**macOS/Linux**
```bash
# install rust (if needed)
curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env

# from repo root with your venv active
python -m pip install maturin
cd s2aff_rust
python -m maturin develop --release
```

**Windows (PowerShell)**
```powershell
# install rust (if needed) via rustup-init.exe or winget
# https://rustup.rs

$env:PATH="$env:USERPROFILE\.cargo\bin;$env:PATH"
.\.venv\Scripts\python.exe -m pip install maturin
cd s2aff_rust
.\.venv\Scripts\python.exe -m maturin develop --release --pip-path .\.venv\Scripts\pip.exe
```
If you see `rustc is not installed or not in PATH`, make sure `C:\Users\<you>\.cargo\bin` is on PATH (or open a new shell after installing rustup).

**Runtime toggles**
```bash
# macOS/Linux
export S2AFF_PIPELINE=python  # force Python pipeline
export S2AFF_PIPELINE=rust    # force Rust pipeline (if installed)

# Windows (PowerShell)
$env:S2AFF_PIPELINE="python"
$env:S2AFF_PIPELINE="rust"
```
Set `S2AFF_PIPELINE=rust` to force Rust usage when it is installed.

**Notes**
- First run builds a cache file under `~/.s2aff/indices/rust/ror_index_<hash>.bin`.
- The cache key includes the ROR file stats + settings, so a new ROR version builds a new cache.
- Use `S2AFF_RUST_LOG=1` for verbose Rust timing logs.
- No environment variable is required when Rust is installed; set `S2AFF_PIPELINE=python` to force the Python pipeline.
- `S2AFF_STAGE1_VARIANT` / `S2AFF_STAGE2_VARIANT` are deprecated.

## Models Download

To get the models, run this command after the package is installed (from inside the `S2AFF` directory):  
```[Expected download size is about 3.4 GiB]```

`aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2aff-release data/`

## Updating the ROR Database
The ROR json database is stored in `data`. To update it locally you have to do the following:

1. `cd data`
2. `python download_latest_ror.py`
3. In `s2aff/constants.py` there is a variable called `ROR_VERSION`. Update it to the new ROR version as per the filename.
4. Run `python scripts/update_openalex_works_counts.py` to get the latest works counts for each ROR id from OpenAlex. (Don't need to do this after every ROR version.)



[If you work at AI2]
To update the ROR database used by default
1. Run this http://s2build.inf.ai2/buildConfiguration/SemanticScholar_SparkCluster_Timo_S2aff_RorUpdate
To update openalex:
2. Run `python scripts/update_openalex_works_counts.py` to get the latest works counts for each ROR id from OpenAlex. (Don't need to do this after every ROR version.)
3. Upload the `openalex_works_counts.csv` to `s3://ai2-s2-research-public/s2aff-release/`



## Modifying the ROR Database to Reduce Systematic Errors
Sometimes the ROR database has issues that we'd like to patch. For example, the ROR entry of https://ror.org/02b34td92 has the alias `ai` which is very common and ends up being a source of many errors. There is an entry in the `data/ror_edits.jsonl` file that patches this. To add more patches, just add a new line to this file with the following format:

```json
{"ror_id": "https://ror.org/02b34td92", "action": "remove", "key": "aliases", "value": "ai"}
```
The actions can be `remove` or `add`. Key is usually `aliases` or `acronyms`. Make sure to make a commit to update the main repo with your changes.

## Performance on Gold Data
Included in this repository in the `data` directory is a dataset of some challenging raw affiliation strings that have been manually assigned
one or more ROR ids, and some of which have no correct RORs at all. Please see `data/gold_affiliations.csv`. The training split of this data has been used to tweak the first stage ROR indexer as well as train the second-stage LightGBM pairwise model.

Data sizes:

| split | n samples |
|:------|:----------|
| train |    1132|
|test  |   644 |
|val   |   588 |

### Latency (GPU, 100 affiliations)
Measured on this machine using:
```
$env:S2AFF_USE_CUDA="1"; $env:S2AFF_PIPELINE="python"; uv run --python .venv python scripts/approximate_runtime_and_memory_usage.py
$env:S2AFF_USE_CUDA="1"; $env:S2AFF_PIPELINE="rust"; uv run --python .venv python scripts/approximate_runtime_and_memory_usage.py
```

**Python pipeline** (Measured Jan 24, 2026):
- NER: **2.061s**
- Stage 1 candidates: **48.745s**
- LightGBM rerank (top 100): **25.492s**
- Total timed stages: **~76.30s**

**Rust pipeline** (Measured Jan 26, 2026; Rust cache hit):
- NER: **1.981s**
- Stage 1 candidates: **2.760s**
- LightGBM rerank (top 100): **7.361s**
- Total timed stages: **~12.10s**

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

## Testing

### Quick Start
To run the fast unit test suite without pulling large artifacts:

```bash
uv run --python .venv python -m pytest -m "not slow and not requires_models"
```

This runs the fast unit tests that test core functionality without requiring model downloads.

### Full Test Suite
Run the full suite including integration tests (requires downloaded models):

```bash
uv run --python .venv python -m pytest
```

When the model files are present under `data/`, tests marked `requires_models` run automatically.
To force-run even if models are not detected, use `--run-requires-models` or set `S2AFF_RUN_REQUIRES_MODELS=1`.
To skip them even when models are present, set `S2AFF_SKIP_REQUIRES_MODELS=1`.

### Parity Harness (Requires Models)
Run exact old-vs-new parity checks (stage1, stage2, end2end):

```bash
uv run --python .venv python scripts/run_parity.py --mode all
```

To enable CUDA for NER in parity checks:

```bash
uv run --python .venv python scripts/run_parity.py --mode all --use-cuda
```

### Coverage
Generate a coverage report:

```bash
uv run --python .venv python -m pytest --cov=s2aff --cov-report=term-missing
```

### Continuous Integration
The project uses GitHub Actions for CI with two test jobs:
- **Fast Tests**: Run on every push/PR without model downloads (~20 seconds)
- **Full Tests**: Run weekly or on-demand with models (~5 minutes, downloads ~3.4GB once)

## Why Not Rerank with BERT?
Because it's worse! We couldn't make `roberta-large` any better than what you see below (validation scores).
![image](https://user-images.githubusercontent.com/1874668/178345697-6af8311d-416e-4540-8489-6eb2eca4186c.png)


## TODO
S2AFF has not been trained to deal with non-English affiliations. It may work for some latin-character languages, but it may not.
If this model is deployed, a language detector should first be applied to the affiliation strings. Then, an analysis will need to be done
to determine which languages will be too error-prone.
