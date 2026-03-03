# S2AFF Deployment Guide

Complete guide for updating and deploying S2AFF to production (AI2 Internal).

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Deployment Workflow](#deployment-workflow)
3. [Post-Deployment Configuration](#post-deployment-configuration)

---

## Prerequisites

### Required Tools
- Python 3.11+
- Docker (for Linux wheel builds)
- AWS CLI (configured with AI2 credentials)
- Access to AI2 internal PyPI repository

### Required Environment Variables

See the [deployment document](https://docs.google.com/document/d/1VoDK3e6PxAEqdvORcXKyeZdS7toiqIVZXbWJYiYdhh8/edit?usp=sharing) for required environment variables (`TWINE_REPOSITORY_URL`, `TWINE_USERNAME`, `TWINE_PASSWORD`, etc.).

Create a `.env` file (see `.env.example`) and source it:
```bash
source .env
```

### Access Requirements
- AI2 network/VPN connection
- Write access to `s3://ai2-timo-registry/`
- Write access to `s3://ai2-s2-research/`
- TeamCity permissions

---

## S2 Pypi Deployment Workflow

### Step 1: Decide on Version Number

Current version: **0.202**

Choose your new version number (e.g., `0.203`).

### Step 2: Prepare Artifacts (if new model artifacts created)

**Only required if you have new model files (ROR data, OpenAlex counts, trained models, etc.)**

```bash
# Download release artifacts
aws s3 sync s3://ai2-s2-research-public/s2aff-release/ ./artifacts/

# Create tarball
tar -czf model_artifacts.tar.gz -C artifacts/ .

# Upload to internal S3
aws s3 cp model_artifacts.tar.gz s3://ai2-s2-research/s2aff/model_artifacts.tar.gz
```

**If the S3 path changes**: Update `artifacts_s3_path` in `s2aff/timo/config.yaml`

### Step 3: Update Version Numbers

Update version in **all three files**:

1. **s2aff/pyproject.toml**
   ```toml
   [project]
   version = "0.203"  # Update this
   ```

2. **s2aff_rust/pyproject.toml**
   ```toml
   [project]
   version = "0.203"  # Update this
   ```

3. **s2aff_rust/Cargo.toml**
   ```toml
   [package]
   version = "0.2.1"  # Update this
   ```

4. **Update Cargo.lock**
   ```bash
   cd s2aff_rust
   cargo update
   git add Cargo.lock
   ```

### Step 4: Build and Publish Packages

**IMPORTANT**: Publish s2aff_rust **BEFORE** s2aff (dependency order).

**Reference**: See [this branch](https://github.com/allenai/S2AFF/tree/add-rust-upgrade-version-0.2.1) for example upload scripts.

#### Prepare Virtual Environments

Create new virtual environments in both directories:

```bash
# In s2aff_rust/
cd s2aff_rust
python3.11 -m venv .venv

# In s2aff/ (repository root)
cd ..
python3.11 -m venv .venv
```

#### 4a. Build and Upload s2aff_rust (Linux wheels)

```bash
# Ensure env vars are set (see deployment document)
cd s2aff_rust
./upload-linux.sh
```

**What this does**:
- Uses Docker to cross-compile for Linux x86_64
- Builds manylinux wheels for Python 3.11
- Uploads wheel to AI2 PyPI

#### 4b. Build and Upload s2aff

```bash
# From repository root
./upload.sh
```

**What this does**:
- Cleans previous build artifacts
- Builds Python package
- Uploads to AI2 PyPI

#### 4c. Clean Up PyPI Artifacts

**Delete the `.tar.gz` source distribution files** from PyPI (keep only `.whl` files):

```bash
# Example: delete s2aff-0.203.tar.gz
# Do this via the PyPI web interface at https://pip.s2.allenai.org/
```

### Step 5: Upload Config to TIMO Registry

```bash
# Upload config YAML to S3 (modify version in path)
aws s3 cp s2aff/timo/config.yaml \
  s3://ai2-timo-registry/model-configs/s2aff/0.203.yaml

# Verify upload
aws s3 ls s3://ai2-timo-registry/model-configs/s2aff/
```

---

## S2 Production Deployment Workflow

### Step 6: Update TIMO Service Config

Update the [TIMO config](https://github.com/allenai/timo/blob/main/timo_services/configs/s2aff_v2.py) to point to the new S2AFF version.

**For existing endpoint (version bump)**:
- Update version number in `timo_services/configs/s2aff_v2.py`
- Submit PR and merge

**For new endpoint (major version)**:
- Create new config file (e.g., `s2aff_v3.py`)
- Reference: [Example PR #355](https://github.com/allenai/timo/pull/355)

### Step 7: Wait for TIMO CI

The merged TIMO change must pass through TIMO CI. Monitor the build and ensure it succeeds.

See the [deployment document](https://docs.google.com/document/d/1VoDK3e6PxAEqdvORcXKyeZdS7toiqIVZXbWJYiYdhh8/edit?usp=sharing) for CI links.

### Step 8: Configure TeamCity Endpoints (new endpoint only)

**If creating a new endpoint**, set up build and deploy pipelines:

1. **Create Build Endpoint**
   - Copy from existing endpoint (e.g., s2aff_v2 build)
   - Update `config_name` parameter to new endpoint name
   - Use `s2` user for permissions
   - Update VCS branch if needed

2. **Create Deploy Endpoint**
   - Copy from existing endpoint (e.g., s2aff_v2 deploy)
   - Update `config_name` parameter to new endpoint name
   - Update VCS branch if needed

3. **Validate Deployment**
   - Run the TeamCity build and deploy pipelines for your endpoint and verify they pass.

See the [deployment document](https://docs.google.com/document/d/1VoDK3e6PxAEqdvORcXKyeZdS7toiqIVZXbWJYiYdhh8/edit?usp=sharing) for TeamCity pipeline links.

### Step 9: Update Scholar Repository (new endpoint only)

**If you created a new endpoint**, update the scholar repository to point to it:

Update these two config files:
1. [PaperAuthorData/base.conf](https://github.com/allenai/scholar/blob/main/authors/src/main/resources/PaperAuthorData/base.conf#L36)
2. [AltAuthorDisambiguation/base.conf](https://github.com/allenai/scholar/blob/main/authors/src/main/resources/AltAuthorDisambiguation/base.conf#L112)

---

**For detailed deployment document**: See [Google Doc](https://docs.google.com/document/d/1VoDK3e6PxAEqdvORcXKyeZdS7toiqIVZXbWJYiYdhh8/edit?usp=sharing)
