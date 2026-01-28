import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import boto3
import requests

from s2aff.util.s3 import get_ror_version

ZENODO_URL = "https://zenodo.org/api/records/?communities=ror-data&sort=mostrecent"
ZIP_SUFFIX = "-ror-data.zip"
JSON_SUFFIX = "-ror-data.json"
TIMEOUT = 60

_VERSION_RE = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})$")


def _parse_version(version: str):
    match = _VERSION_RE.match(version)
    if not match:
        return None
    return (
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
        int(match.group("major")),
        int(match.group("minor")),
    )


def _select_zenodo_file(files):
    candidates = []
    for entry in files:
        key = entry.get("key", "")
        if key.endswith(ZIP_SUFFIX):
            version = key[: -len(ZIP_SUFFIX)]
            candidates.append((_parse_version(version), entry))
    if not candidates:
        raise RuntimeError("Zenodo record did not include a *-ror-data.zip file.")
    parsed = [c for c in candidates if c[0] is not None]
    if parsed:
        parsed.sort(key=lambda item: item[0])
        return parsed[-1][1]
    return candidates[0][1]


def _select_json_member(members):
    preferred = [
        m for m in members if m.endswith(JSON_SUFFIX) and not m.endswith("_schema_v2.json")
    ]
    if preferred:
        if len(preferred) == 1:
            return preferred[0]
        parsed = []
        for name in preferred:
            version = Path(name).name[: -len(JSON_SUFFIX)]
            parsed_version = _parse_version(version)
            if parsed_version:
                parsed.append((parsed_version, name))
        if parsed:
            parsed.sort(key=lambda item: item[0])
            return parsed[-1][1]
        return preferred[0]

    fallback = [m for m in members if m.endswith(".json")]
    if not fallback:
        raise RuntimeError("No JSON files found inside the ROR zip.")
    return fallback[0]


def _get_latest_record():
    response = requests.get(ZENODO_URL, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    hits = payload.get("hits", {}).get("hits", [])
    if not hits:
        raise RuntimeError("Zenodo response did not contain any records.")
    return hits[0]


record = _get_latest_record()
zenodo_file = _select_zenodo_file(record.get("files", []))
download_url = zenodo_file["links"]["self"]
file_name = zenodo_file["key"]
latest_version = file_name[: -len(ZIP_SUFFIX)]

current_version = get_ror_version()
if current_version:
    print("Currently used version is " + current_version)
else:
    print("Currently used version is UNKNOWN (S3 lookup failed).")
print("Latest version is " + latest_version)

if current_version and latest_version == current_version:
    print("ROR up to date. No action required.")
    raise SystemExit(0)

print("Downloading latest version from Zenodo...")
download_response = requests.get(download_url, timeout=TIMEOUT)
download_response.raise_for_status()

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    zip_path = tmpdir_path / file_name
    zip_path.write_bytes(download_response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        zip_ref.extractall(tmpdir_path)

    json_member = _select_json_member(members)
    src_path = tmpdir_path / json_member
    if not src_path.exists():
        raise RuntimeError(f"Expected JSON file not found after unzip: {json_member}")

    new_filename = f"{latest_version}{JSON_SUFFIX}"
    dst_path = Path.cwd() / new_filename
    if dst_path.exists():
        dst_path.unlink()
    shutil.move(str(src_path), dst_path)

# upload to S3
s3 = boto3.client("s3")

bucket_name = "ai2-s2-research-public"
folder_name = "s2aff-release"

upload_key = f"{folder_name}/{new_filename}"
print(f"Uploading {new_filename} to https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/{upload_key}")

s3.upload_file(
    new_filename,
    bucket_name,
    upload_key,
    ExtraArgs={
        "ContentType": "application/json",
        "ACL": "public-read",
    },
)
print("Upload complete.")
