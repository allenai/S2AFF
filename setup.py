import re
import setuptools
from os import path

requirements_file = path.join(path.dirname(__file__), "requirements.in")
requirements = [r for r in open(requirements_file).read().split("\n") if not re.match(r"^\-", r)]

setuptools.setup(
    name="s2aff",
    version="0.12",
    url="https://github.com/allenai/S2AFF",
    packages=setuptools.find_packages(),
    install_requires=requirements,  # dependencies specified in requirements.in
    description='Semantic Scholar\'s Affiliation Extraction: Link Your Raw Affiliations to ROR IDs',
)
