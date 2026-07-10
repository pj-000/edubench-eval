"""Validate Exp27N returned-adjudication aggregate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import read_csv, read_jsonl


DEFAULT_OUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27n_selective_model_adjudication_seed42"
)


def validate(args: argparse.Namespace) -> dict[str, object]:
    paths = {
        "completion": args.out_dir / "tables" / "exp27n_adjudication_completion.csv",
        "distribution": args.out_dir / "tables" / "exp27n_adjudication_distribution.csv",
        "comparison": args.out_dir / "tables" / "exp27n_source_disagreement_to_model_review.csv",
        "qc": args.out_dir / "tables" / "exp27n_adjudication_qc.csv",
        "decision": args.out_dir / "decision" / "exp27n_selective_adjudication_ingest_decision.json",
        "report": args.out_dir / "reports" / "exp27n_selective_adjudication_ingest_report.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    counts = {row["item"]: int(row["count"]) for row in read_csv(paths["completion"])}
    expected = {
        "blind_packet_rows": 54,
        "returned_rows": 54,
        "returned_unique_sample_ids": 54,
        "schema_valid_rows": 54,
        "exact_evidence_rows": 52,
        "missing_evidence_explained_rows": 1,
        "exact_substring_repair_rows": 1,
        "new_resolved_model_silver": 47,
        "new_review_only": 7,
        "existing_required_reviews": 16,
        "all_required_reviews_completed": 70,
        "all_required_resolved_model_silver": 62,
        "all_required_review_only": 8,
    }
    if counts != expected:
        raise ValueError(f"Exp27N completion counts changed: {counts}")
    qc = read_csv(paths["qc"])
    if len(qc) != 10 or any(row["status"] != "PASS" for row in qc):
        raise ValueError(f"Exp27N QC failed: {qc}")
    comparison = read_csv(paths["comparison"])
    if len(comparison) != 8 or {row["population"] for row in comparison} != {
        "all_returned",
        "resolved_only",
    }:
        raise ValueError("Exp27N source-comparison table is incomplete")
    decision = json.loads(paths["decision"].read_text(encoding="utf-8"))
    if (
        decision.get("status") != "PASS"
        or decision.get("returned_rows") != 54
        or decision.get("all_required_adjudications_completed") != 70
        or decision.get("model_review_is_silver") is not True
        or decision.get("dev_test_labels_read") is not False
        or decision.get("proceed_to_361_downstream_dataset_construction") is not True
        or decision.get("proceed_to_training") is not False
        or decision.get("proceed_to_full_3326_expansion") is not False
        or decision.get("proceed_to_dev_test_relabeling") is not False
    ):
        raise ValueError(f"Exp27N ingest decision is invalid: {decision}")
    if args.require_private:
        normalized = read_jsonl(
            args.out_dir / "private" / "exp27n_gpt56pro_selective_adjudication_54.normalized.jsonl"
        )
        consolidated = read_jsonl(
            args.out_dir / "private" / "exp27n_adjudication_reference_70.jsonl"
        )
        if len(normalized) != 54 or len({row["sample_id"] for row in normalized}) != 54:
            raise ValueError("Private normalized Exp27N annotations are incomplete")
        if len(consolidated) != 70 or len({row["sample_id"] for row in consolidated}) != 70:
            raise ValueError("Private consolidated Exp27N reference is incomplete")
    return {
        "status": "PASS",
        "returned_rows": 54,
        "required_adjudications_completed": 70,
        "resolved_model_silver": 62,
        "review_only": 8,
        "next_step": "prepare_361_in_place_downstream_pilot_datasets",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-private", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), ensure_ascii=False, sort_keys=True))
