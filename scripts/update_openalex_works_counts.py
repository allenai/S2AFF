from s2aff.consts import PATHS
from s2aff.file_cache import cached_path
import json, gzip
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config

bucket = "openalex"
prefix = "data/institutions/updated_date="  # partitions

# anonymous/unsigned client (works for OpenAlex bucket)
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

works_count = {}
gz_files = recs = 0

for page in pages:
    for obj in page.get("Contents", []):
        key = obj["Key"]
        # only real data parts; skip manifests/folders
        if not key.endswith(".gz"):
            continue

        gz_files += 1
        if gz_files <= 3:
            print("example key:", key)

        resp = s3.get_object(Bucket=bucket, Key=key)
        with gzip.GzipFile(fileobj=resp["Body"]) as f:
            for bline in f:
                row = json.loads(bline)
                ror = row.get("ror") or (row.get("ids") or {}).get("ror")
                if not ror:
                    continue
                wc = row.get("works_count")
                if wc is None:  # fallback, just in case
                    wc = sum(y.get("works_count", 0) for y in row.get("counts_by_year", []))
                works_count[ror] = wc
                recs += 1

print(f"gz files: {gz_files}, institutions parsed: {recs}, unique RORs: {len(works_count)}")

df = pd.DataFrame.from_dict(works_count, orient="index", columns=["works_count"]).reset_index()
df.columns = ["ror", "works_count"]
df.to_csv(cached_path(PATHS["openalex_works_counts"]), index=False)

# now upload to S3 if you have permissions as an Ai2 employee
# supposed to end up here https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/s2aff-release/openalex_works_counts.csv
s3_client = boto3.client("s3")
s3_client.upload_file(
    cached_path(PATHS["openalex_works_counts"]),
    "ai2-s2-research-public",
    "s2aff-release/openalex_works_counts.csv",
    ExtraArgs={
        "ContentType": "text/csv",
        "ACL": "public-read",
    },
)
