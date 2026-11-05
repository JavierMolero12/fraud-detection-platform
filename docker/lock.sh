#!/bin/sh
# Regenerate the Docker lock files from pyproject.toml, inside the same base image as the
# Dockerfile, constrained to the package versions of the environment the model was trained in
# (docker/training-env-constraints.txt, `pip freeze` of that environment). A served model must
# run on the library versions it was trained with: `fraud serve verify-bundle` checks it.
#
#   sh docker/lock.sh            (from the repository root; needs Docker)
set -eu
export MSYS_NO_PATHCONV=1                  # Git Bash on Windows: keep /repo as it is
HOST_DIR=$(pwd -W 2>/dev/null || pwd)       # Windows path under Git Bash, POSIX elsewhere

BASE=$(sed -n 's/^ARG PYTHON_IMAGE=//p' Dockerfile)

docker run --rm -v "$HOST_DIR:/repo" "$BASE" sh -euc '
    cp -r /repo /tmp/src && cd /tmp/src
    python -m venv /tmp/app && /tmp/app/bin/pip install -q -c docker/training-env-constraints.txt .
    /tmp/app/bin/pip freeze --exclude fraudplatform > /repo/docker/requirements.lock
    python -m venv /tmp/train && /tmp/train/bin/pip install -q -c docker/training-env-constraints.txt ".[train]"
    /tmp/train/bin/pip freeze --exclude fraudplatform > /repo/docker/requirements-train.lock
'
wc -l docker/requirements.lock docker/requirements-train.lock
