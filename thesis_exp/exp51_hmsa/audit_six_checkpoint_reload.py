#!/usr/bin/env python3
"""Verify that all six frozen Exp51 checkpoints reproduce formal dev outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HARD_KEYS = [
    "n",
    "Exact_rounded",
    "MAE_human_mean",
    "Bias_human_mean",
    "Kendall_human_mean",
    "BinAgreement_paper_3way",
    "L2H_count",
    "L2H_rounded",
    "L2H_n",
    "H2L_count",
    "H2L_rounded",
    "H2L_n",
    "QWK_rounded",
]
for label in range(1, 6):
    HARD_KEYS.extend(
        [
            f"Recall_{label}_rounded",
            f"Recall_{label}_correct",
            f"Recall_{label}_total",
            f"Predicted_{label}_count",
        ]
    )

PROBABILITY_KEYS = [
    "MAE_expected_human_mean",
    "NLL_rounded",
    "Brier_rounded",
    "ECE_rounded",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_vector_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{row['record_id']}\t{row['pred_label_5']}" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    formal_roots = {
        "b0": root / "thesis_exp/outputs/exp49_cphce/runs/b0_hard_ce",
        "exp51": root / "thesis_exp/outputs/exp51_hmsa/runs/hmsa_lambda1",
    }
    reload_root = (
        root / "thesis_exp/outputs/exp51_hmsa/audit/final_test_dev_smoke"
    )
    lock_path = root / "thesis_exp/configs/exp51_hmsa/final_test_lock.json"
    decision_path = (
        root / "thesis_exp/outputs/exp51_hmsa/decision/formal_decision.json"
    )
    state_path = (
        root / "thesis_exp/outputs/exp51_hmsa/final_test/campaign_state.json"
    )
    output_path = (
        root
        / "thesis_exp/outputs/exp51_hmsa/audit"
        / "six_checkpoint_dev_reload_integrity.json"
    )
    lock = load_json(lock_path)
    formal_decision = load_json(decision_path)
    expected_dev_text_hash = formal_decision["formal_protocol"]["dev_text_hash"]

    records: list[dict[str, Any]] = []
    all_passed = True
    for arm in ("b0", "exp51"):
        for seed in ("42", "43", "44"):
            formal_dir = formal_roots[arm] / f"seed_{seed}"
            reload_dir = reload_root / arm / f"seed_{seed}"
            formal_metrics_path = formal_dir / "selected_dev_metrics.json"
            reload_metrics_path = reload_dir / "dev_metrics.json"
            formal_predictions_path = formal_dir / "predictions_dev_best.jsonl"
            reload_predictions_path = reload_dir / "predictions_dev.jsonl"

            formal_metrics = load_json(formal_metrics_path)
            reload_metrics = load_json(reload_metrics_path)
            formal_predictions = load_jsonl(formal_predictions_path)
            reload_predictions = load_jsonl(reload_predictions_path)
            run_summary = load_json(formal_dir / "run_summary.json")
            locked = lock["checkpoints"][f"{arm}/seed_{seed}"]

            ids_exact = [
                row["record_id"] for row in formal_predictions
            ] == [row["record_id"] for row in reload_predictions]
            prediction_vector_exact = [
                row["pred_label_5"] for row in formal_predictions
            ] == [row["pred_label_5"] for row in reload_predictions]

            hard_differences: dict[str, float | None] = {}
            for key in HARD_KEYS:
                if key not in formal_metrics or key not in reload_metrics:
                    hard_differences[key] = None
                else:
                    hard_differences[key] = abs(
                        float(formal_metrics[key]) - float(reload_metrics[key])
                    )
            hard_metrics_exact = all(
                difference is not None and difference <= 1e-12
                for difference in hard_differences.values()
            )

            probability_differences = {
                key: abs(float(formal_metrics[key]) - float(reload_metrics[key]))
                for key in PROBABILITY_KEYS
                if key in formal_metrics and key in reload_metrics
            }

            checkpoint_path = root / locked["path"]
            checkpoint_exact = (
                sha256(checkpoint_path) == locked["sha256"]
            )
            epoch_exact = int(run_summary["selected_epoch"]) == int(
                locked["selected_epoch"]
            )
            input_exact = (
                reload_metrics.get("input_text_hash") == expected_dev_text_hash
            )
            expected_inference = (
                lock["b0_inference"]
                if arm == "b0"
                else lock["exp51_inference"]
            )
            inference_exact = reload_metrics.get("inference") == expected_inference
            test_access_zero = (
                reload_metrics.get("test_access_count") == 0
                and reload_metrics.get("test_file_sha256") is None
                and not state_path.exists()
            )

            passed = all(
                [
                    ids_exact,
                    prediction_vector_exact,
                    hard_metrics_exact,
                    checkpoint_exact,
                    epoch_exact,
                    input_exact,
                    inference_exact,
                    test_access_zero,
                ]
            )
            all_passed = all_passed and passed
            records.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "passed": passed,
                    "ids_exact": ids_exact,
                    "hard_prediction_vector_exact": prediction_vector_exact,
                    "formal_prediction_vector_sha256": prediction_vector_sha256(
                        formal_predictions
                    ),
                    "reload_prediction_vector_sha256": prediction_vector_sha256(
                        reload_predictions
                    ),
                    "hard_official_metrics_exact": hard_metrics_exact,
                    "hard_metric_max_abs_diff": max(
                        difference
                        for difference in hard_differences.values()
                        if difference is not None
                    ),
                    "probability_metric_max_abs_diff_diagnostic": max(
                        probability_differences.values(), default=None
                    ),
                    "max_abs_logit_diff_diagnostic": reload_metrics.get(
                        "integrity", {}
                    ).get("max_abs_logit_diff"),
                    "checkpoint_sha256_exact": checkpoint_exact,
                    "selected_epoch_exact": epoch_exact,
                    "dev_input_sha256_exact": input_exact,
                    "hard_head_inference_exact": inference_exact,
                    "test_access_still_zero": test_access_zero,
                    "formal_metrics_sha256": sha256(formal_metrics_path),
                    "reload_metrics_sha256": sha256(reload_metrics_path),
                    "formal_predictions_sha256": sha256(
                        formal_predictions_path
                    ),
                    "reload_predictions_sha256": sha256(
                        reload_predictions_path
                    ),
                }
            )

    report = {
        "decision": (
            "PASS_TEST_UNLOCK"
            if all_passed
            else "BLOCKED_BY_FREEZE_INTEGRITY"
        ),
        "all_six_passed": all_passed,
        "comparison_policy": {
            "hard_prediction_vectors": "exact",
            "official_hard_metrics_abs_tolerance": 1e-12,
            "probability_metrics": "diagnostic_only",
            "test_access_required": 0,
        },
        "final_test_lock_sha256": sha256(lock_path),
        "campaign_state_absent": not state_path.exists(),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(
        json.dumps(
            {
                "decision": report["decision"],
                "all_six_passed": all_passed,
                "campaign_state_absent": report["campaign_state_absent"],
                "records": [
                    {
                        key: record[key]
                        for key in (
                            "arm",
                            "seed",
                            "passed",
                            "hard_prediction_vector_exact",
                            "hard_official_metrics_exact",
                            "hard_metric_max_abs_diff",
                            "probability_metric_max_abs_diff_diagnostic",
                            "checkpoint_sha256_exact",
                            "test_access_still_zero",
                        )
                    }
                    for record in records
                ],
                "report": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
