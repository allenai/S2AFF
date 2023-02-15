"""
Our ROR file needs to have the works count from OpenAlex added to it.

This does that
"""

from s2aff.consts import PATHS
import json
import boto3
import gzip
from io import BytesIO
import pandas as pd


"""
Go through every single file from s3://openalex/data/institutions/updated_date=*/part-***.gz

Stream the jsonl files inside there
"""
bucket = "openalex"
prefix = "data/institutions"

s3 = boto3.client("s3")
paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

works_count_dict = {}
for page in pages:
    for obj in page["Contents"]:
        if obj["Key"].startswith("part") and obj["Key"].endswith(".gz"):
            print("Workong on", obj["Key"])
            obj = s3.get_object(Bucket=bucket, Key=obj["Key"])
            with gzip.GzipFile(fileobj=BytesIO(obj["Body"].read())) as f:
                for line in f:
                    line = json.loads(line)
                    ror_id = line["ror"]
                    works_count = line["works_count"]
                    works_count_dict[ror_id] = works_count

# convert works_count_dict to a dataframe
df = pd.DataFrame.from_dict(works_count_dict, orient="index", columns=["works_count"]).reset_index()
df.columns = ["ror", "works_count"]
df.to_csv(PATHS["openalex_works_counts"], index=False)
