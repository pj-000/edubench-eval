"""Stage 0 audit without parsing the sealed paper-like test split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

from thesis_exp.exp43_rubimor.common import (
    DEV_PATH, EXPECTED_FOLD_HASH, PROCESSED_PATH, ROOT, TEST_PATH, TRAIN_PATH,
    canonical_metric, ensure_dirs, human_stats, normalize_text, raw_rubric,
    read_jsonl, sample_id, sha256_file, stable_hash, triple_hash, write_csv, write_json,
)


LOCKED_EXP42_FOLDS = Path("thesis_exp/exp42_rubidist/outputs/exp42a_rubidist_multiseed/private/data/exp42a_groupcv_fold_assignment.csv")
PUBLIC_SPLIT_MANIFEST = Path("thesis_exp/outputs/exp00_data/tables/split_stats.csv")


def fold_assignment(rows: list[dict]) -> tuple[list[dict], str]:
    if LOCKED_EXP42_FOLDS.exists():
        with LOCKED_EXP42_FOLDS.open("r", encoding="utf-8", newline="") as handle:
            source = [{"sample_id": row["sample_id"], "question_key": row["question_key"], "fold": int(row["fold"])} for row in csv.DictReader(handle)]
        by_id = {row["sample_id"]: row for row in source}
        if set(by_id) != {sample_id(row) for row in rows}:
            raise RuntimeError("Locked Exp42 fold assignment does not match Exp43 train membership")
        assignments = [{"sample_id": sample_id(row), "question_key": str(row["question_key"]), "fold": int(by_id[sample_id(row)]["fold"])} for row in rows]
        return assignments, "copied_from_locked_exp42_assignment"
    labels = [f"{int(row['label_5'])}|{row.get('language') or 'unknown'}" for row in rows]
    groups = [str(row["question_key"]) for row in rows]
    by_id: dict[str, int] = {}
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, heldout) in enumerate(splitter.split(rows, labels, groups)):
        for index in heldout:
            by_id[sample_id(rows[index])] = fold
    return ([{"sample_id": sample_id(row), "question_key": str(row["question_key"]), "fold": by_id[sample_id(row)]} for row in rows], "regenerated_current_sklearn_fallback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--dev", type=Path, default=DEV_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--processed", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    train, dev = read_jsonl(args.train), read_jsonl(args.dev)
    split_rows = {"train": train, "dev": dev}
    counts, labels, metrics, coverage, humans, generators = [], [], [], [], [], []
    valid_humans = True
    metric_values: set[str] = set()
    for split, rows in split_rows.items():
        ids = [sample_id(row) for row in rows]
        counts.append({"split": split, "rows": len(rows), "unique_sample_ids": len(set(ids)), "duplicate_sample_ids": len(ids)-len(set(ids)), "unique_question_keys": len({str(r['question_key']) for r in rows}), "unique_triple_hashes": len({triple_hash(r) for r in rows})})
        for label, count in sorted(Counter(int(row["label_5"]) for row in rows).items()):
            labels.append({"split": split, "label": label, "count": count, "rate": count / len(rows)})
        for metric, count in sorted(Counter(canonical_metric(row) for row in rows).items()):
            metric_values.add(metric)
            metrics.append({"split": split, "canonical_metric": metric, "count": count, "rate": count / len(rows)})
        fields = {
            "rubric": lambda r: bool(raw_rubric(r)), "language": lambda r: bool(r.get("language")),
            "subject": lambda r: bool(r.get("subject_canonical") or r.get("subject_raw")),
            "scenario": lambda r: bool(r.get("scenario_canonical") or r.get("scenario_raw")),
            "education_level": lambda r: bool(r.get("education_level_canonical") or r.get("education_level_raw")),
        }
        for field, present in fields.items():
            n = sum(present(row) for row in rows)
            coverage.append({"split": split, "field": field, "covered": n, "rows": len(rows), "coverage": n / len(rows)})
        for row in rows:
            try:
                stats = human_stats(row)
                legal = True
            except (KeyError, TypeError, ValueError):
                stats, legal, valid_humans = {}, False, False
            humans.append({"split": split, "sample_id_hash": stable_hash(sample_id(row)), "three_valid_scores": legal, "label_matches_half_up": legal and int(row["label_5"]) == stats["gold_label_5"]})
            generators.append({"split": split, "has_generator_model": bool(row.get("generator_model") or row.get("answer_model")), "has_judge_scores": bool(row.get("judge_scores")), "serialized_into_exp43_input": False})
    test_manifest_rows = []
    if PUBLIC_SPLIT_MANIFEST.exists():
        with PUBLIC_SPLIT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
            test_manifest_rows = [row for row in csv.DictReader(handle) if row.get("split_name") == "paper_like_triple_seed42" and row.get("split") == "test"]
    test_manifest_count = int(test_manifest_rows[0]["num_rows"]) if len(test_manifest_rows) == 1 else None
    counts.append({"split": "test_sealed_public_manifest", "rows": test_manifest_count, "unique_sample_ids": "SEALED", "duplicate_sample_ids": "SEALED", "unique_question_keys": "SEALED", "unique_triple_hashes": "SEALED"})
    train_triples, dev_triples = {triple_hash(r) for r in train}, {triple_hash(r) for r in dev}
    train_qkeys, dev_qkeys = {str(r["question_key"]) for r in train}, {str(r["question_key"]) for r in dev}
    assignments, fold_source = fold_assignment(train)
    fold_hash = stable_hash(assignments)
    qkey_folds: dict[str, set[int]] = {}
    for row in assignments:
        qkey_folds.setdefault(row["question_key"], set()).add(int(row["fold"]))
    if any(len(folds) != 1 for folds in qkey_folds.values()):
        raise RuntimeError("Locked assignment splits a question_key across folds")
    write_csv(args.out_dir / "private/data/exp43_groupcv_fold_assignment.csv", assignments)
    overlap = [
        {"key_type": "sample_id", "overlap": len({sample_id(r) for r in train} & {sample_id(r) for r in dev}), "required_zero": True},
        {"key_type": "normalized_question_answer_metric", "overlap": len(train_triples & dev_triples), "required_zero": True},
        {"key_type": "question_key", "overlap": len(train_qkeys & dev_qkeys), "required_zero": False},
    ]
    test_status = "PUBLIC_MANIFEST_ONLY_TEST_CONTENT_NOT_PARSED" if test_manifest_count == 2218 else "TEST_MANIFEST_DEFERRED_UNTIL_FINAL_UNLOCK"
    data_manifest = {
        "locked_start_commit": "4843230", "processed_path": str(args.processed),
        "train": {"path": str(args.train), "rows": len(train), "sha256": sha256_file(args.train)},
        "dev": {"path": str(args.dev), "rows": len(dev), "sha256": sha256_file(args.dev)},
        "training_pool_rows": len(train) + len(dev),
        "test": {"path": str(args.test), "exists": args.test.exists(), "rows_from_public_manifest": test_manifest_count, "manifest_path": str(PUBLIC_SPLIT_MANIFEST), "parsed": False, "status": test_status},
        "split_procedure": "original heldout flag repaired for triple isolation; train pool grouped by triple_key with seed42 into 80/20",
    }
    mapping = {metric: index for index, metric in enumerate(sorted(metric_values))}
    decisions = {
        "train_rows": len(train) == 2654, "dev_rows": len(dev) == 664,
        "training_pool_rows": len(train) + len(dev) == 3318,
        "train_dev_triple_overlap_zero": not (train_triples & dev_triples),
        "human_scores_valid": valid_humans, "canonical_metrics_nonempty_unique": bool(mapping) and "" not in mapping,
        "expected_or_audited_metric_count": len(mapping) > 0,
        "exp42_fold_hash_reproduced": fold_hash == EXPECTED_FOLD_HASH,
        "test_not_parsed": True,
        "test_public_manifest_rows": test_manifest_count == 2218,
    }
    status = "GO" if all(decisions.values()) else "DATA_AUDIT_NO_GO"
    write_csv(args.out_dir / "tables/exp43_stage0_split_counts.csv", counts)
    write_csv(args.out_dir / "tables/exp43_stage0_train_dev_overlap.csv", overlap)
    write_csv(args.out_dir / "tables/exp43_stage0_overlap_audit.csv", overlap)
    write_csv(args.out_dir / "tables/exp43_stage0_label_distribution.csv", labels)
    write_csv(args.out_dir / "tables/exp43_stage0_metric_distribution.csv", metrics)
    write_csv(args.out_dir / "tables/exp43_stage0_metadata_coverage.csv", coverage)
    write_csv(args.out_dir / "tables/exp43_stage0_human_score_audit.csv", humans)
    write_csv(args.out_dir / "tables/exp43_stage0_generator_field_audit.csv", generators)
    write_csv(args.out_dir / "tables/exp43_stage0_exp42_parity.csv", [{"item": "fold_assignment", "expected": EXPECTED_FOLD_HASH, "actual": fold_hash, "match": fold_hash == EXPECTED_FOLD_HASH}])
    write_json(args.out_dir / "configs/exp43_data_manifest.json", data_manifest)
    write_json(args.out_dir / "configs/exp43_metric_mapping.json", {"metrics": mapping, "count": len(mapping), "unknown_metric_id": -1})
    write_json(args.out_dir / "hashes/exp43_split_hashes.json", {"train_sha256": sha256_file(args.train), "dev_sha256": sha256_file(args.dev), "test_sha256": "SEALED_UNTIL_FINAL_UNLOCK", "fold_assignment_sha256": fold_hash})
    write_json(args.out_dir / "hashes/exp43_fold_hashes.json", {"expected": EXPECTED_FOLD_HASH, "actual": fold_hash, "match": fold_hash == EXPECTED_FOLD_HASH, "source": fold_source})
    code_files = sorted(Path("thesis_exp/exp43_rubimor").glob("*.py"))
    write_json(args.out_dir / "hashes/exp43_code_hashes.json", {str(path): sha256_file(path) for path in code_files})
    report = ["# Exp43 Stage 0 Data Audit", "", f"- Decision: **{status}**", f"- Train/dev/training pool: {len(train)}/{len(dev)}/{len(train)+len(dev)}", f"- Canonical metrics: {len(mapping)}", f"- Train/dev triple overlap: {len(train_triples & dev_triples)}", f"- Train/dev question-key overlap (allowed by paper protocol): {len(train_qkeys & dev_qkeys)}", f"- Fold hash reproduced: {fold_hash == EXPECTED_FOLD_HASH}", f"- Test status: {test_status}; labels and answers were not parsed.", "", "The formal paper-like split is triple-disjoint, while Stage 2-6 GroupCV is additionally question-key-disjoint."]
    (args.out_dir / "reports/exp43_stage0_data_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(args.out_dir / "decision/exp43_stage0_decision.json", {"status": status, "checks": decisions, "test_access_count": 0, "test_status": test_status})
    print(json.dumps({"status": status, "checks": decisions, "metrics": len(mapping), "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
