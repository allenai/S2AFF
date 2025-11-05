import requests
import zipfile
import os
import re
import boto3
from s2aff.util.s3 import get_ror_version

# Make a GET request to the Zenodo API endpoint
response = requests.get("https://zenodo.org/api/records/?communities=ror-data&sort=mostrecent")


# Get the download URL of the most recent ROR record
download_url = response.json()["hits"]["hits"][0]["files"][0]["links"]["self"]
file_name = response.json()["hits"]["hits"][0]["files"][0]["key"]

latest_version = re.sub("-ror-data.zip", "", file_name)

current_version = get_ror_version()
print("Currently used version is " + current_version)
print("Latest version is " + latest_version)
if latest_version == current_version:
    print("ROR up to date. No action required.")
else:
    print("Downloading latest version from Zenodo...")
    response = requests.get(download_url)
    with open(file_name, "wb") as f:
        f.write(response.content)

    # unzip the file_name and delete the zip file
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall()
        extracted_files = zip_ref.namelist()
    unzipped_filename = [name for name in extracted_files if name.endswith(".json")][0]
    os.remove(file_name)

    # we need to rename the unzipped file name so that it's exactly
    # f"{ROR_VERSION}-ror-data.json"
    # to match what's in paths.py
    new_filename = f"{latest_version}-ror-data.json"
    os.rename(unzipped_filename, new_filename)

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
