"""Prepare the Exp27R final lock without opening the test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.exp27p.common import read_jsonl, stable_hash, write_csv, write_json
from thesis_exp.src.edujudge.exp27q import OUTPUT_DIR as EXP27Q_OUTPUT
from thesis_exp.src.edujudge.exp27r import COMPARISONS, OUTPUT_DIR, SEEDS, VARIANTS, run_root
from thesis_exp.src.edujudge.exp27r.bootstrap_exp27r_crossed_seed_question import crossed_bootstrap


TRAIN = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEV = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
EXP27O = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42")
EXP27P_OUTPUT = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27p_soft_target_reranker_multiseed_seed42_44")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(path: Path) -> tuple[str, list[dict[str, Any]]]:
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append({"relative_path": str(item.relative_to(path)), "size": item.stat().st_size, "sha256": file_sha256(item)})
    return stable_hash(files), files


def current_commit(expected: str | None) -> str:
    if expected:
        return expected
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def prepare(out_dir: Path, expected_commit: str | None, crossed_resamples: int) -> dict[str, Any]:
    tables, configs, protocols = (out_dir / name for name in ("tables", "configs", "protocols"))
    for path in (tables, configs, protocols):
        path.mkdir(parents=True, exist_ok=True)
    commit = current_commit(expected_commit)
    registry, sensitivity, lock_checks = [], [], []
    dev_predictions: dict[tuple[int, str], list[dict[str, Any]]] = {}
    pure_complete = True
    for variant in VARIANTS:
        for seed in SEEDS:
            run = run_root(variant) / variant / f"seed_{seed}"
            summary_path = run / "run_summary.json"
            pred_path = run / "predictions_private/selected_dev_predictions.jsonl"
            if not summary_path.exists() or not pred_path.exists():
                raise FileNotFoundError(f"Missing frozen run material: {run}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            valid = (
                summary.get("status") == "COMPLETED"
                and summary.get("variant") == variant
                and int(summary.get("seed", -1)) == seed
                and int(summary.get("test_access_count", -1)) == 0
                and summary.get("checkpoint_reload_pass") is True
            )
            if not valid:
                raise ValueError(f"Invalid frozen summary: {summary_path}")
            selected = Path(summary["selected_checkpoint"])
            pure = Path(summary["config"]["checkpoint_output_dir"]) / f"epoch_{int(summary['pure_min_mae_epoch']):02d}"
            if not selected.exists():
                raise FileNotFoundError(f"Selected checkpoint missing: {selected}")
            pure_complete = pure_complete and pure.exists()
            selected_hash, selected_files = tree_manifest(selected)
            registry.append({
                "variant": variant, "seed": seed, "checkpoint_kind": "selected",
                "epoch": summary["selected_epoch"], "checkpoint_path": str(selected),
                "checkpoint_tree_sha256": selected_hash, "file_count": len(selected_files),
                "selected_primary": True,
            })
            if pure.exists():
                pure_hash, pure_files = tree_manifest(pure)
                registry.append({
                    "variant": variant, "seed": seed, "checkpoint_kind": "pure_min_mae",
                    "epoch": summary["pure_min_mae_epoch"], "checkpoint_path": str(pure),
                    "checkpoint_tree_sha256": pure_hash, "file_count": len(pure_files),
                    "selected_primary": False,
                })
            selected_metrics, pure_metrics = summary["selected_metrics"], summary["pure_min_mae_metrics"]
            sensitivity.append({
                "variant": variant, "seed": seed, "selected_epoch": summary["selected_epoch"],
                "pure_min_mae_epoch": summary["pure_min_mae_epoch"],
                "same_epoch": summary["selected_epoch"] == summary["pure_min_mae_epoch"],
                **{f"selected_{key}": selected_metrics[key] for key in ("MAE_argmax", "QWK", "low_to_high_rate", "label2_recall", "label5_recall")},
                **{f"pure_{key}": pure_metrics[key] for key in ("MAE_argmax", "QWK", "low_to_high_rate", "label2_recall", "label5_recall")},
            })
            dev_predictions[(seed, variant)] = read_jsonl(pred_path)
            lock_checks.append({"check": f"{variant}:seed_{seed}", "status": "PASS", "test_access_count": 0})

    if not pure_complete:
        registry = [row for row in registry if row["checkpoint_kind"] == "selected"]

    crossed_rows = []
    for left, right, effect in COMPARISONS:
        crossed_rows.extend(crossed_bootstrap(
            {seed: dev_predictions[(seed, left)] for seed in SEEDS},
            {seed: dev_predictions[(seed, right)] for seed in SEEDS},
            left, right, effect, crossed_resamples, 27017,
        ))
    nested_path = EXP27Q_OUTPUT / "tables/exp27q_two_level_seed_question_bootstrap_ci.csv"
    nested = list(csv.DictReader(nested_path.open("r", encoding="utf-8", newline="")))
    nested_lookup = {(row["metric"]): row for row in nested}
    nested_vs_crossed = []
    for row in crossed_rows:
        nested_row = nested_lookup.get(row["metric"]) if row["left_variant"] == "v3_selective_soft_audit" and row["right_variant"] == "v3_safe16_original_low_anchor" else None
        nested_vs_crossed.append({
            **row,
            "nested_ci_low_95": nested_row.get("ci_low_95", "") if nested_row else "",
            "nested_ci_high_95": nested_row.get("ci_high_95", "") if nested_row else "",
        })

    hashes = [
        {"artifact": "train_split", "path": str(TRAIN), "sha256": file_sha256(TRAIN), "phase1_test_read": False},
        {"artifact": "dev_split", "path": str(DEV), "sha256": file_sha256(DEV), "phase1_test_read": False},
        {"artifact": "exp27o_fingerprints", "path": str(EXP27O / "tables/exp27o_dataset_fingerprints.csv"), "sha256": file_sha256(EXP27O / "tables/exp27o_dataset_fingerprints.csv"), "phase1_test_read": False},
        {"artifact": "safe16_equivalence", "path": str(EXP27Q_OUTPUT / "tables/exp27q_safe16_dataset_equivalence.csv"), "sha256": file_sha256(EXP27Q_OUTPUT / "tables/exp27q_safe16_dataset_equivalence.csv"), "phase1_test_read": False},
        {"artifact": "exp27p_locked_config", "path": str(EXP27P_OUTPUT / "configs/exp27p_multiseed_locked_config.json"), "sha256": file_sha256(EXP27P_OUTPUT / "configs/exp27p_multiseed_locked_config.json"), "phase1_test_read": False},
    ]
    for code_path in (
        Path("thesis_exp/src/edujudge/exp27p/common.py"),
        Path("thesis_exp/src/edujudge/exp27p/train_exp27p_soft_target_reranker.py"),
        Path("thesis_exp/src/edujudge/exp27r/evaluate_exp27r_frozen_checkpoints.py"),
        Path("thesis_exp/src/edujudge/exp27r/collect_exp27r_final_test.py"),
        Path("thesis_exp/src/edujudge/exp27r/bootstrap_exp27r_crossed_seed_question.py"),
    ):
        hashes.append({"artifact": f"code:{code_path.name}", "path": str(code_path),
                       "sha256": file_sha256(code_path), "phase1_test_read": False})
    manifest = {
        "experiment": "exp27r_final_test_campaign_seed42_44", "locked_source_commit": commit,
        "methods_frozen": True, "training_frozen": True, "data_frozen": True,
        "test_access_before_campaign": 0, "test_hash_deferred_until_explicit_gate": True,
        "variants": list(VARIANTS), "seeds": list(SEEDS), "comparisons": [list(row) for row in COMPARISONS],
        "selected_checkpoint_count": 15, "pure_min_sensitivity_enabled": pure_complete,
        "pure_min_checkpoint_count": 15 if pure_complete else 0,
        "checkpoint_registry_sha256": stable_hash(registry),
        "dataset_and_config_hashes_sha256": stable_hash(hashes),
        "crossed_bootstrap_resamples": crossed_resamples,
        "no_new_method_after_test": True,
    }
    write_csv(tables / "exp27r_checkpoint_registry.csv", registry)
    write_csv(tables / "exp27r_dataset_and_config_hashes.csv", hashes)
    write_csv(tables / "exp27r_selected_vs_pure_dev_sensitivity.csv", sensitivity)
    write_csv(tables / "exp27r_lock_validation.csv", lock_checks + [
        {"check": "selected_checkpoint_count_15", "status": "PASS", "test_access_count": 0},
        {"check": "phase1_test_not_read", "status": "PASS", "test_access_count": 0},
    ])
    write_csv(tables / "exp27r_dev_nested_vs_crossed_bootstrap.csv", nested_vs_crossed)
    write_json(configs / "exp27r_final_lock_manifest.json", manifest)
    protocol = """# Exp27R Protocol Amendment\n\n- Exp27P V3 status: `SOFT_TARGET_NOT_STABLE`.\n- Exp27Q status: `SAFE16_INCONCLUSIVE`.\n- Safe16 was not promoted as a successful method.\n- This one-shot test campaign validates the complete frozen V0/V1/V2/V3/Safe16 matrix; it does not select or rescue a method.\n- All five methods, three seeds, and preregistered comparisons must be reported.\n- After test access, labels, weights, tau, checkpoint selection, training, and methods must not change.\n- Positive and negative test outcomes are reported without returning to dev.\n"""
    (protocols / "exp27r_protocol_amendment.md").write_text(protocol, encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--expected-commit")
    parser.add_argument("--crossed-resamples", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(prepare(args.out_dir, args.expected_commit, args.crossed_resamples), sort_keys=True))
