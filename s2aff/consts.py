import logging
import os
import ntpath
from pathlib import Path
import boto3
logger = logging.getLogger("s2aff")

try:
    PROJECT_ROOT_PATH = os.path.abspath(
        os.path.join(__file__, os.pardir, os.pardir))
except NameError:
    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))


CACHE_ROOT = Path(os.getenv("S2AFF_CACHE", str(Path.home() / ".s2aff")))

DEFAULT_ROR_VERSION = "v1.20-2023-02-28"


def get_ror_version():
    s3 = boto3.client('s3')
    suffix = "-ror-data.json"
    response = s3.list_objects_v2(
        Bucket="ai2-s2-research-public", Prefix="s2aff-release")

    if 'Contents' in response:
        files = [obj for obj in response['Contents']
                 if obj['Key'].endswith(suffix)]
        if files:
            most_recent_file = max(files, key=lambda x: x['LastModified'])
            return most_recent_file['Key'].split('/')[-1].split(suffix)[0]
    return DEFAULT_ROR_VERSION


ROR_VERSION = get_ror_version()

PATHS = {
    "ner_training_data": os.path.join(PROJECT_ROOT_PATH, "data", "ner_training_data.pickle"),
    "ror_data": os.path.join(PROJECT_ROOT_PATH, "data", f"{ROR_VERSION}-ror-data.json"),
    "country_info": os.path.join(PROJECT_ROOT_PATH, "data", "country_info.txt"),
    "openalex_works_counts": os.path.join(PROJECT_ROOT_PATH, "data", "openalex_works_counts.csv"),
    "gold_affiliation_annotations": os.path.join(PROJECT_ROOT_PATH, "data", "gold_affiliation_annotations.csv"),
    "ner_model": os.path.join(PROJECT_ROOT_PATH, "data", "ner_model"),
    "kenlm_model": os.path.join(PROJECT_ROOT_PATH, "data", "raw_affiliations_lowercased.binary"),
    "lightgbm_gold_features": os.path.join(PROJECT_ROOT_PATH, "data", "lightgbm_gold_features.pickle"),
    "lightgbm_gold_no_ror_features": os.path.join(PROJECT_ROOT_PATH, "data", "lightgbm_gold_no_ror_features.pickle"),
    "lightgbm_model": os.path.join(PROJECT_ROOT_PATH, "data", "lightgbm_model.booster"),
    "ror_edits": os.path.join(PROJECT_ROOT_PATH, "data", "ror_edits.jsonl"),
}

for key in PATHS.keys():
    if not os.path.exists(PATHS[key]):
        PATHS[key] = "https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/s2aff-release/" + ntpath.basename(
            PATHS[key]
        )

USE_PROB_WEIGHTS = True
MAX_INTERSECTION_DENOMINATOR = True
WORD_MULTIPLIER = 0.3
NS = {3}  # character trigrams only
INSERT_EARLY_CANDIDATES_IND = 1  # insert early candidates at second place
REINSERT_CUTOFF_FRAC = 1.1  # reinsert candidates with score > cutoff
# anything with a score this low is removed before any kind of sorting is done
SCORE_BASED_EARLY_CUTOFF = 0.01
