"""Create and verify the Exp63 immutable source lock."""

from __future__ import annotations

import argparse
import json

from thesis_exp.exp63_same_state_counterfactual import REPO_ROOT, SOURCE_LOCK_PATH
from thesis_exp.exp63_same_state_counterfactual.runtime import sha256_file, write_json


LOCKED_FILES = (
    "thesis_exp/configs/exp63_same_state_counterfactual/protocol.json",
    "thesis_exp/exp63_same_state_counterfactual/__init__.py",
    "thesis_exp/exp63_same_state_counterfactual/runtime.py",
    "thesis_exp/exp63_same_state_counterfactual/train_base.py",
    "thesis_exp/exp63_same_state_counterfactual/counterfactual.py",
    "thesis_exp/exp63_same_state_counterfactual/analyze.py",
    "thesis_exp/exp63_same_state_counterfactual/preflight.py",
    "thesis_exp/exp57_cbrd/data_audit.py",
    "thesis_exp/exp57_cbrd/losses.py",
    "thesis_exp/exp57_cbrd/method.py",
    "thesis_exp/exp57_cbrd/model.py",
    "thesis_exp/exp57_cbrd/train.py",
    "thesis_exp/exp59_residual_geometry/train.py",
    "thesis_exp/src/edujudge/exp02/train_ce_baseline.py",
    "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py",
    "thesis_exp/src/edujudge/utils/io.py",
    "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl",
    "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"
)


def current_hashes() -> dict[str, str]:
    return {relative: sha256_file(REPO_ROOT / relative) for relative in LOCKED_FILES}


def verify() -> dict[str, object]:
    lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    actual = current_hashes()
    mismatches = {
        relative: {"expected": expected, "actual": actual.get(relative)}
        for relative, expected in lock["files"].items()
        if actual.get(relative) != expected
    }
    if mismatches:
        raise RuntimeError(f"Exp63 source-lock mismatch: {mismatches}")
    return {"status": "PASS", "files": len(actual), "source_lock": str(SOURCE_LOCK_PATH)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_json(
            SOURCE_LOCK_PATH,
            {
                "status": "EXP63_SOURCE_LOCK_FROZEN_BEFORE_REAL_MODEL_UPDATE_RESULTS",
                "files": current_hashes(),
            },
        )
    print(json.dumps(verify(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

