#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

python -m thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.test_monotone_projection
EXP12_PREFLIGHT_ONLY=1 ./thesis_exp/scripts/run_exp12_monotonic_projection_map_oc.sh >/tmp/exp12_smoke_preflight.log
python -m thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.smoke_check_exp12
python -m thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.collect_exp12_results
python -m thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.readability_check_exp12
cat thesis_exp/outputs/exp12_monotonic_projection_map_oc/smoke_test/reports/exp12_smoke_check.md
