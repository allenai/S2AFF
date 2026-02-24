# Deployment Guide (AI2 Internal)

This document contains deployment scripts and instructions for AI2 internal use.

## Prerequisites

### Required Environment Variables

```bash
export TWINE_REPOSITORY_URL="<your-internal-pypi-url>"
export TWINE_USERNAME="<username>"
export TWINE_PASSWORD="<password>"
```

**Note**: Contact your team for the correct values for `TWINE_REPOSITORY_URL`, `TWINE_USERNAME`, and `TWINE_PASSWORD`.

### Optional: Using .env file

You can create a `.env` file in the repository root (see `.env.example`) and source it:

```bash
source .env
```

## Deployment Scripts

### `upload.sh`

**Purpose**: Builds and uploads the main S2AFF Python package to AI2 PyPI.

**Requirements**:
- Python 3.11+
- `TWINE_REPOSITORY_URL`, `TWINE_USERNAME`, `TWINE_PASSWORD` environment variables

**What it does**:
1. Cleans previous build artifacts (`dist/`)
2. Upgrades build tools (pip, setuptools, wheel, build, twine)
3. Builds the package using `python -m build`
4. Uploads to AI2 PyPI using `twine upload`

**Usage**:
```bash
# Set all required env vars (or source .env)
export TWINE_REPOSITORY_URL="<your-internal-pypi-url>"
export TWINE_USERNAME="<username>"
export TWINE_PASSWORD="<password>"
./upload.sh
```

### `s2aff_rust/upload-linux.sh`

**Purpose**: Cross-compiles the Rust extension for Linux x86_64 and uploads the wheel to AI2 PyPI.

**Requirements**:
- Docker
- `TWINE_REPOSITORY_URL`, `TWINE_USERNAME`, `TWINE_PASSWORD` environment variables

**What it does**:
1. Cleans previous build artifacts (`target/wheels/`)
2. Uses the official PyO3 maturin Docker image to cross-compile for Linux x86_64
3. Builds manylinux wheels compatible with Python 3.11
4. Installs/upgrades twine
5. Uploads only the Linux manylinux wheel to AI2 PyPI

**Usage**:
```bash
# Set all required env vars (or source .env)
export TWINE_REPOSITORY_URL="<your-internal-pypi-url>"
export TWINE_USERNAME="<username>"
export TWINE_PASSWORD="<password>"
cd s2aff_rust
./upload-linux.sh
```

**Note**: This script is typically run from macOS to produce Linux wheels. Use the maturin Docker image to ensure compatibility with production Linux environments.

## Deployment Workflow

### Standard Release (macOS users)

1. **Update version** in `pyproject.toml` and `s2aff_rust/Cargo.toml`
2. **Build and upload Linux Rust extension**:
   ```bash
   cd s2aff_rust
   ./upload-linux.sh
   ```
3. **Build and upload Python package**:
   ```bash
   # Set all required env vars (or source .env)
   export TWINE_REPOSITORY_URL="<your-internal-pypi-url>"
   export TWINE_USERNAME="<username>"
   export TWINE_PASSWORD="<password>"
   ./upload.sh
   ```

### Linux Native Build

If you're already on Linux x86_64, you can build natively instead of using Docker:

```bash
# Set all required env vars (or source .env)
export TWINE_REPOSITORY_URL="<your-internal-pypi-url>"
export TWINE_USERNAME="<username>"
export TWINE_PASSWORD="<password>"

cd s2aff_rust
maturin build --release --manylinux 2014 -i python3.11
twine upload target/wheels/*manylinux*.whl
```

## Troubleshooting

### Error: Environment variables must be set

Make sure you've exported all required environment variables before running the scripts:

```bash
export TWINE_REPOSITORY_URL="<your-internal-pypi-url>"
export TWINE_USERNAME="<username>"
export TWINE_PASSWORD="<password>"
```

Or source your `.env` file if you created one.

### Docker Issues

If you encounter Docker-related errors with `upload-linux.sh`:
- Ensure Docker is running
- Verify you have access to `ghcr.io/pyo3/maturin` image
- Check that the current directory is mounted correctly in the Docker command

### Upload Failures

If twine upload fails:
- Verify you're connected to the AI2 network or VPN
- Check that the package version doesn't already exist in the repository
- Ensure the build artifacts were created successfully before upload
