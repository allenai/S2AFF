import boto3
from botocore.config import Config
from botocore import UNSIGNED
from botocore.exceptions import BotoCoreError, ClientError

def get_ror_version():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    suffix = "-ror-data.json"
    try:
        response = s3.list_objects_v2(
            Bucket="ai2-s2-research-public", Prefix="s2aff-release")
    except (BotoCoreError, ClientError):
        return None

    if 'Contents' in response:
        files = [obj for obj in response['Contents']
                 if obj['Key'].endswith(suffix)]
        if files:
            most_recent_file = max(files, key=lambda x: x['LastModified'])
            return most_recent_file['Key'].split('/')[-1].split(suffix)[0]
    return None
