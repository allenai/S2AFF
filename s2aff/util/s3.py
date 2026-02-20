import re
from typing import Optional, Tuple, Iterable, Dict, Any

import boto3
from botocore.config import Config
from botocore import UNSIGNED
from botocore.exceptions import BotoCoreError, ClientError

_ROR_VERSION_RE = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})$")


def _parse_ror_version(version: str) -> Optional[Tuple[int, int, int, int, int]]:
    match = _ROR_VERSION_RE.match(version)
    if not match:
        return None
    return (
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
        int(match.group("major")),
        int(match.group("minor")),
    )


def _iter_s3_objects(s3, bucket: str, prefix: str) -> Iterable[Dict[str, Any]]:
    continuation = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        response = s3.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []):
            yield obj
        if not response.get("IsTruncated"):
            break
        continuation = response.get("NextContinuationToken")

def get_ror_version():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    suffix = "-ror-data.json"
    bucket = "ai2-s2-research-public"
    prefix = "s2aff-release"
    try:
        files = [obj for obj in _iter_s3_objects(s3, bucket=bucket, prefix=prefix) if obj["Key"].endswith(suffix)]
    except (BotoCoreError, ClientError):
        return None

    if not files:
        return None

    parsed = []
    for obj in files:
        filename = obj["Key"].split("/")[-1]
        version = filename[: -len(suffix)]
        parsed_version = _parse_ror_version(version)
        if parsed_version:
            parsed.append((parsed_version, obj))

    if parsed:
        parsed.sort(key=lambda item: (item[0], item[1].get("LastModified")))
        most_recent_file = parsed[-1][1]
    else:
        most_recent_file = max(files, key=lambda x: x.get("LastModified"))

    return most_recent_file["Key"].split("/")[-1].split(suffix)[0]
