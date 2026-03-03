#!/usr/bin/env bash
# AI2 Internal: Builds and uploads S2AFF Python package to AI2 PyPI
# Requires: TWINE_REPOSITORY_URL environment variable

set -euo pipefail
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Validate required environment variables
: "${TWINE_REPOSITORY_URL:?Error: TWINE_REPOSITORY_URL must be set. See DEPLOYMENT.md for details.}"
: "${TWINE_USERNAME:?Error: TWINE_USERNAME must be set. See DEPLOYMENT.md for details.}"
: "${TWINE_PASSWORD:?Error: TWINE_PASSWORD must be set. See DEPLOYMENT.md for details.}"

rm -rf dist

export TWINE_NON_INTERACTIVE=1

python -m pip install --upgrade pip setuptools wheel build twine
python -m build
twine upload dist/*
