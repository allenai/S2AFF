import requests
import zipfile
import os
import re 
# Make a GET request to the Zenodo API endpoint
response = requests.get("https://zenodo.org/api/records/?communities=ror-data&sort=mostrecent")

# Get the download URL of the most recent ROR record
download_url = response.json()["hits"]["hits"][0]["files"][0]["links"]["self"]
file_name = response.json()["hits"]["hits"][0]["files"][0]["key"]

# Download the record
response = requests.get(download_url)

# Save the record to a file
with open(file_name, "wb") as f:
    f.write(response.content)

# unzip the file_name and delete the zip file
with zipfile.ZipFile(file_name, "r") as zip_ref:
    zip_ref.extractall()
os.remove(file_name)

# get a list of all of the json files in the directory that that have the text 'ror-data' in them
# and sort it in order of most recent version to least recent version
json_files = [f for f in os.listdir(".") if os.path.isfile(f) and f.endswith(".json") and "ror-data" in f]
json_files.sort(reverse=True)

print("Downloaded this file: " + json_files[0])
print(f"Version is: {re.sub('-ror-data.json','',json_files[0])}")
print("Make sure to do these things and open a PR:")
print("(1) Update s2aff/conts.py with the new ROR_VERSION")
print(
    f"(2) Upload to the S3 bucket with the new ROR json file: aws s3 cp {json_files[0]} s3://ai2-s2-research-public/s2aff-release/"
)
print(
    f"(3) Delete from the S3 bucket with the old ROR json file: aws s3 rm s3://ai2-s2-research-public/s2aff-release/{json_files[1]}"
)
