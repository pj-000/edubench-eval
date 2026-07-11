"""Validate that Exp27R completed once and is permanently closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.src.edujudge.exp27r import OUTPUT_DIR


def validate(out_dir: Path) -> dict[str, object]:
    decision = json.loads((out_dir / "decision/exp27r_final_test_decision.json").read_text(encoding="utf-8"))
    required = (
        decision["lock_pass"] and decision["methods_frozen"] and decision["training_frozen"]
        and decision["data_frozen"] and decision["test_campaign_completed"]
        and decision["test_access_count"] == 1 and decision["all_variants_evaluated"]
        and decision["all_seeds_evaluated"] and decision["selected_checkpoint_results_complete"]
        and decision["primary_comparisons_complete"] and decision["crossed_bootstrap_complete"]
        and decision["recommend_more_training"] is False and decision["final_test_closed"] is True
    )
    if not required:
        raise ValueError("Exp27R final close validation failed")
    return {"status": "PASS", "test_access_count": 1, "final_test_closed": True,
            "final_paper_position": decision["final_paper_position"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    print(json.dumps(validate(parser.parse_args().out_dir), sort_keys=True))

