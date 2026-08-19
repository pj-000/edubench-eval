"""Aggregate all three real-model no-update preflights without authorizing training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp61_soft_sts15_external_confirmation import OUTPUT_ROOT, SEEDS, VARIANTS
from thesis_exp.exp61_soft_sts15_external_confirmation.contract import SOURCE_LOCK


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize() -> dict[str, Any]:
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    records: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for seed in SEEDS:
        path = OUTPUT_ROOT / "preflight" / f"seed_{seed}" / "real_model_no_update_preflight.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        records[str(seed)] = {
            "sha256": sha256(path),
            "status": record["status"],
            "identity_error": record["soft_ce_identity"]["absolute_identity_error"],
            "parameter_sha256": record["initial_parameter_sha256"],
            "model_manifest_sha256": record["model_manifest_sha256"],
        }
        checks[f"seed_{seed}_status_pass"] = record["status"] == "EXP61_REAL_MODEL_NO_UPDATE_PREFLIGHT_PASS"
        checks[f"seed_{seed}_all_checks_pass"] = all(record["checks"].values())
        checks[f"seed_{seed}_all_arms_present"] = set(record["arm_geometry_audits"]) == set(VARIANTS)
        checks[f"seed_{seed}_model_matches_lock"] = record["model_manifest_sha256"] == lock["model"]["manifest_sha256"]
    checks["source_lock_remains_candidate"] = lock["status"] == "EXP61_SOURCE_LOCK_CANDIDATE_AWAITING_FINAL_PREFLIGHT_REVIEW"
    checks["formal_training_remains_disabled"] = lock["formal_training_authorized"] is False
    result = {
        "status": "EXP61_PREFLIGHT_IMPLEMENTATION_PASS_AWAITING_INDEPENDENT_REVIEW" if all(checks.values()) else "EXP61_PREFLIGHT_IMPLEMENTATION_FAIL",
        "seeds": records,
        "checks": checks,
        "optimizer_step_count": 0,
        "test_access_count": 0,
        "formal_training_authorized": False,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    output = OUTPUT_ROOT / "decision/preflight_implementation_decision.json"
    result = finalize()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
