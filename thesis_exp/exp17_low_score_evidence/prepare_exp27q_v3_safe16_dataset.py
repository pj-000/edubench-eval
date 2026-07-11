"""Build the locked Exp27Q dataset by restoring exactly 16 original low labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.exp27p.common import input_source_hash, read_jsonl, stable_hash, write_csv, write_jsonl
from thesis_exp.src.edujudge.exp27q import EXP27O_DIR, OUTPUT_DIR, SAFE16_DATASET


V3_DATASET = EXP27O_DIR / "private/data/exp27o_v3_selective_soft_audit_train.jsonl"
MANIFEST = EXP27O_DIR / "tables/exp27o_high_impact16_manifest_light.csv"


def one_hot(label: int) -> list[float]:
    return [1.0 if value == label else 0.0 for value in range(1, 6)]


def supervision_mass(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    output = []
    for label in range(1, 6):
        subset = [row for row in rows if int(row["original_label_5"]) == label]
        output.append({
            "variant": variant,
            "original_label_5": label,
            "row_count": len(subset),
            "active_row_count": sum(float(row["sample_weight"]) > 0 for row in subset),
            "sample_weight_sum": sum(float(row["sample_weight"]) for row in subset),
            **{
                f"weighted_target_mass_{target}": sum(
                    float(row["sample_weight"]) * float(row["soft_target_5"][target - 1]) for row in subset
                )
                for target in range(1, 6)
            },
        })
    return output


def build(v3_path: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(v3_path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    manifest_ids = [row["sample_id"] for row in manifest]
    if len(manifest_ids) != 16 or len(set(manifest_ids)) != 16:
        raise ValueError("Safe16 manifest must contain exactly 16 unique sample IDs")
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    row_ids = [str(row["sample_id"]) for row in rows]
    if len(rows) != 3326 or len(set(row_ids)) != 3326 or not set(manifest_ids) <= set(row_ids):
        raise ValueError("V3 dataset identity/size does not match the locked design")

    output: list[dict[str, Any]] = []
    transitions = []
    for row in rows:
        sid = str(row["sample_id"])
        new_row = dict(row)
        if sid in manifest_by_id:
            original = int(row["original_label_5"])
            current = int(row["label_5"])
            reference = str(row.get("exp27o_reference_status", ""))
            if not (
                original <= 2
                and current >= 4
                and row.get("exp27o_training_tier") == "resolved_adjudication"
                and reference == "model_review_silver_not_human_gold"
            ):
                raise ValueError(f"Safe16 precondition failed: {sid}")
            old_target = list(row["soft_target_5"])
            new_row["label_5"] = original
            new_row["soft_target_5"] = one_hot(original)
            new_row["exp27q_override"] = "restore_original_low_one_hot_directional_safety"
            new_row["exp27q_original_v3_target"] = {
                "label_5": current,
                "soft_target_hash": stable_hash(old_target),
            }
            transitions.append({
                "sample_id_hash": stable_hash(sid),
                "question_key_hash": stable_hash(row.get("question_key")),
                "original_label_5": original,
                "v3_label_5": current,
                "safe16_label_5": original,
                "v3_soft_target_hash": stable_hash(old_target),
                "safe16_soft_target_5": json.dumps(one_hot(original), separators=(",", ":")),
                "sample_weight": row["sample_weight"],
                "reference_status": reference,
            })
        output.append(new_row)

    if len(transitions) != 16:
        raise ValueError(f"Expected 16 transitions, found {len(transitions)}")
    output_path = output_dir / "private/data/exp27q_v3_safe16_original_low_anchor_train.jsonl"
    write_jsonl(output_path, output)
    tables = output_dir / "tables"
    write_csv(tables / "exp27q_safe16_target_transition.csv", transitions)

    changed = [
        sid for sid, left, right in zip(row_ids, rows, output)
        if stable_hash(left) != stable_hash(right)
    ]
    equivalence = [{
        "rows": len(rows),
        "unique_sample_id": len(set(row_ids)),
        "changed_rows_vs_v3": len(changed),
        "changed_input_hashes": sum(input_source_hash(a) != input_source_hash(b) for a, b in zip(rows, output)),
        "changed_sample_weights": sum(float(a["sample_weight"]) != float(b["sample_weight"]) for a, b in zip(rows, output)),
        "changed_raw_quality_weights": sum(float(a["raw_quality_weight"]) != float(b["raw_quality_weight"]) for a, b in zip(rows, output)),
        "non_safe16_target_mismatches": sum(
            (a["label_5"], a["soft_target_5"]) != (b["label_5"], b["soft_target_5"])
            for a, b in zip(rows, output) if a["sample_id"] not in manifest_by_id
        ),
        "safe16_id_symmetric_difference": len(set(changed) ^ set(manifest_ids)),
        "v3_global_weight_sum": sum(float(row["sample_weight"]) for row in rows),
        "safe16_global_weight_sum": sum(float(row["sample_weight"]) for row in output),
        "v3_mean_sample_weight": sum(float(row["sample_weight"]) for row in rows) / len(rows),
        "safe16_mean_sample_weight": sum(float(row["sample_weight"]) for row in output) / len(output),
    }]
    write_csv(tables / "exp27q_safe16_dataset_equivalence.csv", equivalence)
    write_csv(
        tables / "exp27q_safe16_effective_supervision_mass.csv",
        supervision_mass(rows, "v3_selective_soft_audit") + supervision_mass(output, "v3_safe16_original_low_anchor"),
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        if row["sample_id"] in manifest_by_id:
            grouped[str(row["question_key"])].append(row)
    write_csv(tables / "exp27q_safe16_question_key_summary.csv", [{
        "question_key_hash": stable_hash(key),
        "safe16_row_count": len(group),
        "original_label_counts": json.dumps(Counter(int(row["original_label_5"]) for row in group), sort_keys=True),
    } for key, group in sorted(grouped.items())])
    return {"rows": len(output), "changed_rows": 16, "changed_question_keys": len(grouped), "output": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-jsonl", type=Path, default=V3_DATASET)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(args.v3_jsonl, args.manifest, args.out_dir), sort_keys=True))
