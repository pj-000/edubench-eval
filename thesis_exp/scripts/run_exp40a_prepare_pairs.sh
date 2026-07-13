#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python thesis_exp/exp40_edupair_cf/resolve_exp39a_pair_inputs.py
python thesis_exp/exp40_edupair_cf/prepare_exp40a_pairwise_verification_packets.py
python thesis_exp/exp40_edupair_cf/run_exp40a_deepseek_pairwise_verification.py --max-pairs 1 --dry-run

echo "Exp40A resolved 240 private Exp39A pairs and prepared 480 target-blind packets."
