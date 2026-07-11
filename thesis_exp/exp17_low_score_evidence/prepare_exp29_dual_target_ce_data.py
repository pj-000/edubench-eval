"""Build train-only uncertainty-preserving dual-target CE datasets for Exp29."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXP28_DATA = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/private/datasets"
)
DEFAULT_FUSION = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/"
    "tables/exp28e_fusion_decisions_light.csv"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp29_dual_target_ce_seed42"
)
VARIANTS = (
    "c1_audited_dual_target",
    "c2_selected_exposure_control",
    "c3_random_dual_target_control",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duplicate(row: dict[str, Any], target: int, suffix: str, provenance: str) -> dict[str, Any]:
    copied = dict(row)
    copied["id"] = f"{row['id']}::{suffix}"
    copied["label"] = target - 1
    copied["label_5"] = target
    copied["target_provenance"] = provenance
    copied["dual_target_duplicate"] = True
    return copied


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    for variant in ("b0_original_human", "b2_selective_dual_teacher", "b4_random_transition_control"):
        for split in ("train", "dev"):
            path = args.exp28_data / variant / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
        if (args.exp28_data / variant / "test.jsonl").exists():
            raise ValueError("Exp29 scout must not read or inherit test data")
    baseline = read_jsonl(args.exp28_data / "b0_original_human" / "train.jsonl")
    selective = read_jsonl(args.exp28_data / "b2_selective_dual_teacher" / "train.jsonl")
    random_control = read_jsonl(args.exp28_data / "b4_random_transition_control" / "train.jsonl")
    dev = read_jsonl(args.exp28_data / "b0_original_human" / "dev.jsonl")
    if not (len(baseline) == len(selective) == len(random_control) == 2654 and len(dev) == 664):
        raise ValueError("Exp29 requires the locked Exp28 paper train/dev datasets")
    by_id = {str(row["record_id"]): row for row in baseline}
    selective_by_id = {str(row["record_id"]): row for row in selective}
    random_by_id = {str(row["record_id"]): row for row in random_control}
    if not (set(by_id) == set(selective_by_id) == set(random_by_id)):
        raise ValueError("Exp28 training identities differ")
    accepted = {
        row["sample_id"]
        for row in read_csv(args.fusion)
        if str(row.get("accepted_change") or "").lower() == "true"
    }
    selective_changed = {
        sample_id
        for sample_id in by_id
        if int(selective_by_id[sample_id]["label_5"]) != int(by_id[sample_id]["label_5"])
    }
    random_changed = {
        sample_id
        for sample_id in by_id
        if int(random_by_id[sample_id]["label_5"]) != int(by_id[sample_id]["label_5"])
    }
    if accepted != selective_changed or len(accepted) != 518 or len(random_changed) != len(accepted):
        raise ValueError("Locked accepted/random transition identities are inconsistent")

    variants = {name: [dict(row) for row in baseline] for name in VARIANTS}
    for sample_id in sorted(accepted):
        original = by_id[sample_id]
        variants["c1_audited_dual_target"].append(
            duplicate(
                original,
                int(selective_by_id[sample_id]["label_5"]),
                "audited-teacher-target",
                "qwen_deepseek_consensus_secondary_target",
            )
        )
        variants["c2_selected_exposure_control"].append(
            duplicate(
                original,
                int(original["label_5"]),
                "selected-original-target",
                "selected_row_original_target_exposure_control",
            )
        )
    for sample_id in sorted(random_changed):
        original = by_id[sample_id]
        variants["c3_random_dual_target_control"].append(
            duplicate(
                original,
                int(random_by_id[sample_id]["label_5"]),
                "random-teacher-target",
                "matched_random_transition_secondary_target",
            )
        )

    output_root = args.out_dir / "private" / "datasets"
    dev_hashes = set()
    summary = []
    for name, rows in variants.items():
        if len(rows) != 3172:
            raise ValueError(f"Unexpected {name} row count: {len(rows)}")
        train_path = output_root / name / "train.jsonl"
        dev_path = output_root / name / "dev.jsonl"
        write_jsonl(train_path, rows)
        write_jsonl(dev_path, dev)
        dev_hashes.add(sha256(dev_path))
        counts = Counter(int(row["label_5"]) for row in rows)
        summary.append(
            {
                "variant": name,
                "train_rows": len(rows),
                "base_rows": 2654,
                "duplicate_rows": len(rows) - 2654,
                **{f"label_{label}": counts[label] for label in range(1, 6)},
                "loss": "ordinary_cross_entropy",
                "test_read": False,
            }
        )
    if len(dev_hashes) != 1 or any((output_root / name / "test.jsonl").exists() for name in VARIANTS):
        raise RuntimeError("Exp29 dev identity or no-test invariant failed")
    write_csv(
        args.out_dir / "tables" / "exp29_dataset_summary.csv",
        summary,
        ["variant", "train_rows", "base_rows", "duplicate_rows", "label_1", "label_2", "label_3", "label_4", "label_5", "loss", "test_read"],
    )
    decision = {
        "status": "READY_FOR_SEED42_DEV_SCOUT",
        "variants": {row["variant"]: row["train_rows"] for row in summary},
        "accepted_teacher_disagreements": len(accepted),
        "original_label_always_retained": True,
        "teacher_target_used_as_additional_observation": True,
        "loss": "ordinary_cross_entropy",
        "test_read": False,
        "test_written": False,
    }
    decision_path = args.out_dir / "decision" / "exp29_dataset_decision.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = """# Exp29 Uncertainty-Preserving Dual-Target CE Data

Every original benchmark target is retained. For the 518 high-confidence adjacent disagreements
accepted by the locked Qwen/DeepSeek audit, C1 adds a second observation carrying the teacher
target. C2 repeats the same selected rows with the original target, controlling for exposure.
C3 adds the identical transition multiset at matched random positions, controlling for targeting.
All variants use ordinary cross-entropy; dev is identical and test is neither read nor written.
"""
    report_path = args.out_dir / "reports" / "exp29_dataset_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp28-data", type=Path, default=DEFAULT_EXP28_DATA)
    parser.add_argument("--fusion", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), ensure_ascii=False, sort_keys=True))
