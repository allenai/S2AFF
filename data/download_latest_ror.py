import argparse
import os
import zipfile
from pathlib import Path

import requests

ZENODO_URL = "https://zenodo.org/api/records/?communities=ror-data&sort=mostrecent"
ZIP_SUFFIX = "-ror-data.zip"
TIMEOUT = 60


def _get_latest_record():
    response = requests.get(ZENODO_URL, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    hits = payload.get("hits", {}).get("hits", [])
    if not hits:
        raise RuntimeError("Zenodo response did not contain any records.")
    return hits[0]


def main():
    parser = argparse.ArgumentParser(description="Download the latest ROR data zip from Zenodo.")
    default_out = Path(__file__).resolve().parent
    parser.add_argument("--out-dir", default=str(default_out), help="Directory to write the ROR data into.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    record = _get_latest_record()
    files = record.get("files", [])
    if not files:
        raise RuntimeError("Zenodo record did not include any files.")

    zip_files = [f for f in files if str(f.get("key", "")).endswith(ZIP_SUFFIX)]
    if not zip_files:
        raise RuntimeError("Zenodo record did not include a *-ror-data.zip file.")

    download_url = zip_files[0]["links"]["self"]
    file_name = zip_files[0]["key"]

    response = requests.get(download_url, timeout=TIMEOUT)
    response.raise_for_status()

    zip_path = out_dir / file_name
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    zip_path.unlink()

    json_files = [
        p.name
        for p in out_dir.iterdir()
        if p.is_file() and p.name.endswith(".json") and "ror-data" in p.name
    ]
    json_files.sort(reverse=True)

    if not json_files:
        raise RuntimeError("No ror-data JSON files found after extraction.")

    print("Downloaded this file: " + json_files[0])
    print("Make sure to do these things and open a PR:")
    print("(1) Update s2aff/consts.py with the new ROR version (or set S2AFF_ROR_VERSION).")
    print(
        f"(2) Upload to the S3 bucket with the new ROR json file: aws s3 cp {json_files[0]} s3://ai2-s2-research-public/s2aff-release/"
    )
    if len(json_files) > 1:
        print(
            f"(3) Delete from the S3 bucket with the old ROR json file: aws s3 rm s3://ai2-s2-research-public/s2aff-release/{json_files[1]}"
        )
    else:
        print("(3) Delete the old ROR json file from S3 (previous version not detected locally).")


if __name__ == "__main__":
    main()
