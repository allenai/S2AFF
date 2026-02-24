#!/usr/bin/env bash
# AI2 Internal: Cross-compiles Rust extension for Linux x86_64 and uploads to AI2 PyPI
# Requires: TWINE_REPOSITORY_URL environment variable, Docker

set -euo pipefail
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Validate required environment variables
: "${TWINE_REPOSITORY_URL:?Error: TWINE_REPOSITORY_URL must be set. See DEPLOYMENT.md for details.}"
: "${TWINE_USERNAME:?Error: TWINE_USERNAME must be set. See DEPLOYMENT.md for details.}"
: "${TWINE_PASSWORD:?Error: TWINE_PASSWORD must be set. See DEPLOYMENT.md for details.}"

rm -rf target/wheels

export TWINE_NON_INTERACTIVE=1

# Build for Linux x86_64 using maturin Docker image
docker run --rm --platform linux/amd64 -v $(pwd):/io \
  ghcr.io/pyo3/maturin \
  build --release --manylinux 2014 -i python3.11 --out /io/target/wheels

# Install twine if needed
pip install --upgrade twine

# Upload only the Linux wheel
twine upload target/wheels/*manylinux*.whl
