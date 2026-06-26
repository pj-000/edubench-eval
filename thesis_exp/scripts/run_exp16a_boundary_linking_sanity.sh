#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python -m thesis_exp.src.edujudge.exp16_boundary_linking.sanity_check_boundary_linking \
  --variant "${VARIANT:-qmr_meta}"
