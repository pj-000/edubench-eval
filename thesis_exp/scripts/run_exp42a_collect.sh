#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
SEEDS_STRING="${SEEDS:-42 43 44}"
read -r -a RUN_SEEDS <<< "${SEEDS_STRING}"

"${PYTHON}" thesis_exp/exp42_rubidist/collect_exp42a_multiseed.py --seeds "${RUN_SEEDS[@]}"
"${PYTHON}" thesis_exp/exp42_rubidist/bootstrap_exp42a_crossed_seed_question.py \
  --seeds "${RUN_SEEDS[@]}" --question-resamples 2000 --crossed-resamples 5000
"${PYTHON}" thesis_exp/exp42_rubidist/analyze_exp42a_factorial_effects.py

echo "Exp42A collection, bootstrap, factorial analysis, and locked decision complete."
