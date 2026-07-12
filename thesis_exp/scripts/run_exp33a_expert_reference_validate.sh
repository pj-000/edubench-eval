#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=42

python thesis_exp/exp33_expert_reference/validate_exp33a_expert_reference.py "$@"
