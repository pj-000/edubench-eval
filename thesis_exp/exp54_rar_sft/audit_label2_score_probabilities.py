"""Independently audit frozen Label-2 score-probability extraction outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT


ARMS = ("R3", "P1_FIELD_DPO")
SEEDS = (42, 43, 44)
SCORES = (1, 2, 3, 4, 5)
EXPECTED_ROW_KEYS = {
    "row_position",
    "record_id",
    "question_key",
    "metric_id",
    "language",
    "label_5",
    "human_mean_5",
    "human_1_5",
    "human_2_5",
    "human_3_5",
    "parsed_score",
    "forced_completion",
    "prompt_token_ids_sha256",
    "canonical_score_option_logprobs",
    "canonical_score_option_probabilities",
    "forced_choice_score",
}
DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
)
DEFAULT_PRIVATE_ROOT = (
    DEFAULT_ARTIFACT_ROOT
    / "label2_identification_audit/private/score_probabilities"
)
DEFAULT_INVENTORY = (
    DEFAULT_ARTIFACT_ROOT / "label2_identification_audit/inventory_report.json"
)
DEFAULT_DEV = (
    REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"
)
DEFAULT_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "label2_identification_audit/score_probability_extraction_report.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected JSON objects")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def normalized_probabilities(logprobabilities: list[float]) -> list[float]:
    if len(logprobabilities) != 5 or any(
        not math.isfinite(value) for value in logprobabilities
    ):
        raise ValueError("five finite log-probabilities are required")
    maximum = max(logprobabilities)
    terms = [math.exp(value - maximum) for value in logprobabilities]
    denominator = sum(terms)
    return [value / denominator for value in terms]


def validate_probability_row(
    row: dict[str, Any],
    *,
    dev_row: dict[str, Any],
    prediction_row: dict[str, Any],
    row_position: int,
) -> None:
    if set(row) != EXPECTED_ROW_KEYS:
        raise ValueError(f"row {row_position}: probability row keys differ")
    exact = {
        "row_position": row_position,
        "record_id": str(dev_row["record_id"]),
        "question_key": str(dev_row["question_key"]),
        "metric_id": str(dev_row["metric_id"]),
        "language": str(dev_row["language"]),
        "label_5": int(dev_row["label_5"]),
        "parsed_score": int(prediction_row["prediction"]["score"]),
        "forced_completion": bool(prediction_row["forced_completion"]),
        "prompt_token_ids_sha256": str(
            prediction_row["prompt_token_ids_sha256"]
        ),
    }
    for field, expected in exact.items():
        if row.get(field) != expected:
            raise ValueError(f"row {row_position}: {field} differs")
    for field in ("human_mean_5", "human_1_5", "human_2_5", "human_3_5"):
        if float(row[field]) != float(dev_row[field]):
            raise ValueError(f"row {row_position}: {field} differs")

    expected_score_keys = {str(score) for score in SCORES}
    logprob_map = row["canonical_score_option_logprobs"]
    probability_map = row["canonical_score_option_probabilities"]
    if set(logprob_map) != expected_score_keys or set(probability_map) != expected_score_keys:
        raise ValueError(f"row {row_position}: score keys differ")
    logprobs = [float(logprob_map[str(score)]) for score in SCORES]
    probabilities = [float(probability_map[str(score)]) for score in SCORES]
    recomputed = normalized_probabilities(logprobs)
    if any(
        not math.isfinite(value) or abs(value - expected) > 1e-7
        for value, expected in zip(probabilities, recomputed, strict=True)
    ):
        raise ValueError(f"row {row_position}: probability normalization differs")
    forced_choice = max(SCORES, key=lambda score: probabilities[score - 1])
    if int(row["forced_choice_score"]) != forced_choice:
        raise ValueError(f"row {row_position}: forced choice differs")


def _paths(
    artifact_root: Path,
    private_root: Path,
    arm: str,
    seed: int,
) -> tuple[Path, Path, Path]:
    private_arm = arm.lower()
    probability_dir = private_root / private_arm / f"seed_{seed}"
    if arm == "R3":
        prediction = (
            artifact_root
            / f"dev_runs_vllm/r3/seed{seed}/epoch3/predictions.jsonl"
        )
    else:
        prediction = (
            artifact_root
            / "preference_lr5e6_followup/dev/"
            f"p1_field_dpo/seed_{seed}/predictions.jsonl"
        )
    return (
        probability_dir / "score_probabilities.jsonl",
        probability_dir / "report.json",
        prediction,
    )


def _score_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    nll = 0.0
    brier = 0.0
    rps = 0.0
    exact = 0
    for row in rows:
        gold = int(row["label_5"])
        probabilities = [
            float(row["canonical_score_option_probabilities"][str(score)])
            for score in SCORES
        ]
        nll -= math.log(max(probabilities[gold - 1], 1e-300))
        brier += sum(
            (probabilities[index] - float(index == gold - 1)) ** 2
            for index in range(5)
        )
        cumulative = 0.0
        for threshold in range(1, 5):
            cumulative += probabilities[threshold - 1]
            target = float(gold <= threshold)
            rps += (cumulative - target) ** 2
        exact += int(int(row["forced_choice_score"]) == gold)
    return {
        "forced_choice_exact": exact / len(rows),
        "multiclass_nll": nll / len(rows),
        "multiclass_brier": brier / len(rows),
        "ranked_probability_score": rps / len(rows),
    }


def audit(
    *,
    artifact_root: Path,
    private_root: Path,
    inventory_path: Path,
    dev_path: Path,
) -> dict[str, Any]:
    inventory = read_json(inventory_path)
    server_inventory = inventory["server_read_only_inventory"]
    dev_rows = read_jsonl(dev_path)
    if len(dev_rows) != 664:
        raise ValueError("locked dev row count differs")
    expected_dev_hash = "a18d6a27b9a524d4592a359658ae70c9348fe88e43c962971ba95f62d2b6cdf0"
    if sha256_file(dev_path) != expected_dev_hash:
        raise ValueError("locked dev hash differs")

    private_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    public: dict[str, Any] = {}
    extractor_hashes: set[str] = set()
    for arm in ARMS:
        public[arm] = {}
        prediction_group = (
            "r3_epoch3_dev_predictions"
            if arm == "R3"
            else "p1_lr5e6_dev_predictions"
        )
        adapter_group = (
            "r3_epoch3_adapter_sha256"
            if arm == "R3"
            else "p1_lr5e6_adapter_sha256"
        )
        for seed in SEEDS:
            probability_path, report_path, prediction_path = _paths(
                artifact_root, private_root, arm, seed
            )
            for path in (probability_path, report_path, prediction_path):
                if not path.is_file() or path.is_symlink():
                    raise FileNotFoundError(path)
            report = read_json(report_path)
            probability_rows = read_jsonl(probability_path)
            prediction_rows = read_jsonl(prediction_path)
            if len(probability_rows) != 664 or len(prediction_rows) != 664:
                raise ValueError(f"{arm}/{seed}: row count differs")
            expected_prediction = server_inventory[prediction_group][str(seed)]
            exact_report = {
                "status": "LABEL2_SCORE_PROBABILITY_EXTRACTION_COMPLETE",
                "arm": arm,
                "seed": seed,
                "rows": 664,
                "label_2_rows": 14,
                "score_probability_rows": 664,
                "adapter_model_sha256": server_inventory[adapter_group][str(seed)],
                "input_predictions_sha256": expected_prediction["sha256"],
                "dev_sha256": expected_dev_hash,
                "forward_only": True,
                "grad_enabled": False,
                "training_started": False,
                "test_accessed": False,
            }
            for field, expected in exact_report.items():
                if report.get(field) != expected:
                    raise ValueError(f"{arm}/{seed}: report {field} differs")
            if report.get("canonical_continuation_lengths") != {
                str(score): 1 for score in SCORES
            }:
                raise ValueError(f"{arm}/{seed}: continuation lengths differ")
            probability_hash = sha256_file(probability_path)
            if report.get("output_score_probabilities_sha256") != probability_hash:
                raise ValueError(f"{arm}/{seed}: output hash differs")
            if sha256_file(prediction_path) != expected_prediction["sha256"]:
                raise ValueError(f"{arm}/{seed}: prediction hash differs")
            extractor_hashes.add(str(report["extractor_source_sha256"]))
            for position, (row, dev_row, prediction_row) in enumerate(
                zip(probability_rows, dev_rows, prediction_rows, strict=True)
            ):
                validate_probability_row(
                    row,
                    dev_row=dev_row,
                    prediction_row=prediction_row,
                    row_position=position,
                )
            label2 = [row for row in probability_rows if row["label_5"] == 2]
            probability_means = {
                str(score): sum(
                    float(row["canonical_score_option_probabilities"][str(score)])
                    for row in label2
                )
                / len(label2)
                for score in SCORES
            }
            public[arm][str(seed)] = {
                "rows": len(probability_rows),
                "label_2_rows": len(label2),
                "output_sha256": probability_hash,
                "parsed_score_counts_label_2": dict(
                    sorted(Counter(str(row["parsed_score"]) for row in label2).items())
                ),
                "forced_choice_counts_label_2": dict(
                    sorted(
                        Counter(str(row["forced_choice_score"]) for row in label2).items()
                    )
                ),
                "decoder_recoverable_label_2_failures": sum(
                    row["parsed_score"] != 2 and row["forced_choice_score"] == 2
                    for row in label2
                ),
                "mean_score_probabilities_label_2": probability_means,
                "all_row_probability_metrics": _score_metrics(probability_rows),
                "elapsed_seconds": float(report["elapsed_seconds"]),
            }
            private_rows[arm, seed] = probability_rows

    if len(extractor_hashes) != 1:
        raise ValueError("extraction source hash differs across runs")
    mass_flow: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        r3 = private_rows["R3", seed]
        p1 = private_rows["P1_FIELD_DPO", seed]
        for left, right in zip(r3, p1, strict=True):
            if left["record_id"] != right["record_id"]:
                raise ValueError(f"seed {seed}: R3/P1 record order differs")
        selected = [
            (left, right)
            for left, right in zip(r3, p1, strict=True)
            if left["label_5"] == 2
        ]
        mass_flow[str(seed)] = {
            str(score): sum(
                float(right["canonical_score_option_probabilities"][str(score)])
                - float(left["canonical_score_option_probabilities"][str(score)])
                for left, right in selected
            )
            / len(selected)
            for score in SCORES
        }
        if abs(sum(mass_flow[str(seed)].values())) > 1e-7:
            raise ValueError(f"seed {seed}: probability mass is not conserved")

    return {
        "schema_version": "exp54-label2-score-probability-audit-v1",
        "status": "LABEL2_SCORE_PROBABILITY_AUDIT_PASS",
        "arms": public,
        "p1_minus_r3_mean_probability_mass_label_2": mass_flow,
        "extractor_source_sha256": next(iter(extractor_hashes)),
        "inventory_report_sha256": sha256_file(inventory_path),
        "dev_sha256": expected_dev_hash,
        "private_row_level_outputs_published": False,
        "gpu_used_by_auditor": False,
        "training_started": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(
        artifact_root=args.artifact_root,
        private_root=args.private_root,
        inventory_path=args.inventory,
        dev_path=args.dev,
    )
    write_json(args.output, result)
    print("LABEL2_SCORE_PROBABILITY_AUDIT_PASS")


if __name__ == "__main__":
    main()
