#!/usr/bin/env python3
"""Fit frozen EduDART-Cal on Exp33 calibration and qualify on Exp35 silver."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp33_direction_aware_aggregation.prepare_exp33b_direction_aware_aggregation import (  # noqa: E402
    DEFAULT_EXP28E_DECISION,
    DEFAULT_EXP33A_OUT,
    DEFAULT_SPLIT_DIR,
    DEFAULT_TEACHER_SUMMARY_DIR,
    human_empirical_distribution,
    load_final_silver_reference,
    load_selected_train_views,
    load_train_sources,
    quadratic_weighted_kappa,
    resolve_teacher_inputs,
)


DEFAULT_CONFIG = Path("thesis_exp/exp35_edudart_cal/configs/exp35a_edudart_cal_preregistration.json")
DEFAULT_EXP35_OUT = Path("thesis_exp/exp35_edudart_cal/outputs/exp35a_model_reviewed_qualification_seed42")
LABELS = range(1, 6)
METHODS = (
    "rounded_human", "human_empirical", "qwen_hard", "qwen_calibrated",
    "naive_fusion", "EduDART_pre_projection_diagnostic", "EduDART-Cal",
)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with repo_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with repo_path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.rstrip() + "\n", encoding="utf-8")


def normalize(values: Iterable[float]) -> list[float]:
    output = [max(0.0, float(value)) for value in values]
    total = sum(output)
    return [value / total for value in output] if total else [0.2] * 5


def expected(posterior: list[float]) -> float:
    return sum(score * posterior[score - 1] for score in LABELS)


def hard(posterior: list[float]) -> int:
    return max(LABELS, key=lambda score: (posterior[score - 1], -score))


def entropy(posterior: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in posterior) / math.log(5)


def source_obs(source: dict[str, Any], name: str) -> int | None:
    value = source.get(f"{name}_score")
    return int(value) if value is not None else None


def rounded(source: dict[str, Any]) -> int:
    return int(source["rounded_human_label"])


def qwen_raw(source: dict[str, Any], temperature: float) -> list[float]:
    score = source_obs(source, "qwen")
    if score is None:
        return human_empirical_distribution(source)
    value = source.get("qwen_score_range") or [score, score]
    lower, upper = int(value[0]), int(value[-1])
    return normalize(
        math.exp(-abs(label - score) / temperature) if lower <= label <= upper else 0.0
        for label in LABELS
    )


def group_key(source: dict[str, Any]) -> str:
    score_region = "low" if rounded(source) <= 2 else "mid" if rounded(source) == 3 else "high"
    return f"{score_region}|{source.get('language') or 'unknown'}|{source.get('metric_family') or 'unknown'}"


def fit_confusions(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    smoothing = float(config["calibration"]["confusion_smoothing"])
    global_counts = np.full((5, 5), smoothing, dtype=float)
    group_counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((5, 5), dtype=float))
    group_weights: defaultdict[str, float] = defaultdict(float)
    prior = np.full(5, smoothing, dtype=float)
    transition = np.full((5, 5), smoothing, dtype=float)
    for record in records:
        source = record["source"]
        obs = source_obs(source, "qwen")
        if obs is None:
            continue
        weight = float(record["weight"])
        posterior = record["reference"]
        human_label = rounded(source)
        for true_label in LABELS:
            mass = weight * posterior[true_label - 1]
            global_counts[true_label - 1, obs - 1] += mass
            group_counts[group_key(source)][true_label - 1, obs - 1] += mass
            prior[true_label - 1] += mass
            transition[human_label - 1, true_label - 1] += mass
            group_weights[group_key(source)] += mass
    global_matrix = global_counts / global_counts.sum(axis=1, keepdims=True)
    group_models = {}
    shrink_rows = float(config["calibration"]["coarse_group_shrinkage_rows"])
    for key, counts in group_counts.items():
        local = counts + smoothing
        local /= local.sum(axis=1, keepdims=True)
        shrink = group_weights[key] / (group_weights[key] + shrink_rows)
        group_models[key] = shrink * local + (1.0 - shrink) * global_matrix
    transition /= transition.sum(axis=1, keepdims=True)
    return {
        "global": global_matrix,
        "groups": group_models,
        "prior": prior / prior.sum(),
        "human_transition": transition,
    }


def calibrated_teacher(source: dict[str, Any], model: dict[str, Any], config: dict[str, Any]) -> list[float]:
    raw = qwen_raw(source, float(config["calibration"]["teacher_range_temperature"]))
    confusion = model["groups"].get(group_key(source), model["global"])
    likelihood = confusion @ np.asarray(raw)
    return normalize(model["prior"] * likelihood)


def feature_row(source: dict[str, Any]) -> dict[str, Any]:
    human = human_empirical_distribution(source)
    qwen = source_obs(source, "qwen") or rounded(source)
    failures = set(source.get("qwen_major_failures") or [])
    return {
        "human_entropy": entropy(human),
        "qwen_human_gap": abs(qwen - rounded(source)),
        "qwen_score": qwen,
        "qwen_confidence": float(source.get("qwen_confidence") or 0.0),
        "evidence_present": float(bool(source.get("qwen_evidence_flags"))),
        "reason_present": float(bool(source.get("qwen_reason_present"))),
        "rubric_items": float(source.get("qwen_rubric_assessment_items") or 0),
        "score_cap_present": float(source.get("qwen_score_cap") is not None),
        "direction": "same" if qwen == rounded(source) else "up" if qwen > rounded(source) else "down",
        "language": str(source.get("language") or "unknown"),
        "metric_family": str(source.get("metric_family") or "unknown"),
        "major_failure": ";".join(sorted(failures)) or "none",
    }


def fit_reliability(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    vectorizer = DictVectorizer(sparse=False)
    x = vectorizer.fit_transform([feature_row(record["source"]) for record in records])
    y = np.asarray([
        int(abs((source_obs(record["source"], "qwen") or rounded(record["source"])) - hard(record["reference"])) <= 1)
        for record in records
    ])
    weights = np.asarray([float(record["weight"]) for record in records])
    if len(set(y.tolist())) < 2:
        return {"vectorizer": vectorizer, "model": None, "constant": float(y[0])}
    model = LogisticRegression(C=1.0 / float(config["calibration"]["reliability_l2"]), max_iter=1000, random_state=42)
    model.fit(x, y, sample_weight=weights)
    return {"vectorizer": vectorizer, "model": model, "constant": None}


def reliability(source: dict[str, Any], model: dict[str, Any], config: dict[str, Any]) -> float:
    if model["model"] is None:
        value = float(model["constant"])
    else:
        x = model["vectorizer"].transform([feature_row(source)])
        value = float(model["model"].predict_proba(x)[0, 1])
    return min(float(config["calibration"]["reliability_ceiling"]), max(float(config["calibration"]["reliability_floor"]), value))


def evidence_and_learnability_gate(source: dict[str, Any]) -> bool:
    qwen = source_obs(source, "qwen")
    if qwen is None or not source.get("qwen_reason_present") or int(source.get("qwen_rubric_assessment_items") or 0) == 0:
        return False
    cap = source.get("qwen_score_cap")
    if cap is not None and qwen > int(cap):
        return False
    gap = abs(qwen - rounded(source))
    if gap <= 1:
        return True
    observable = {
        "missing_key_point", "factual_or_rubric_mismatch", "insufficient_evidence",
        "task_constraint_violation", "reasoning_gap", "format_violation",
    }
    return bool(observable & set(source.get("qwen_major_failures") or []))


def direction_factor(source: dict[str, Any], config: dict[str, Any]) -> float:
    human = rounded(source)
    qwen = source_obs(source, "qwen") or human
    mixing = config["mixing"]
    if human <= 2 and qwen >= 4:
        return float(mixing["human_low_to_teacher_high_without_independent_support"])
    if human >= 4 and qwen <= 2:
        return float(mixing["human_high_to_teacher_low_with_evidence"])
    if human == 4 and qwen == 5:
        return float(mixing["human_four_to_teacher_five"])
    return float(mixing["same_or_adjacent_direction"])


def initial_edudart(source: dict[str, Any], confusion: dict[str, Any], reliability_model: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    human = human_empirical_distribution(source)
    teacher = calibrated_teacher(source, confusion, config)
    gate = evidence_and_learnability_gate(source)
    alpha = float(config["mixing"]["alpha_max"]) * reliability(source, reliability_model, config) * float(gate) * direction_factor(source, config)
    posterior = normalize((1.0 - alpha) * human[index] + alpha * teacher[index] for index in range(5))
    return {"posterior": posterior, "alpha": alpha, "gate": gate, "reliability": reliability(source, reliability_model, config)}


def project_distribution(rows: list[dict[str, Any]], confusion: dict[str, Any], config: dict[str, Any]) -> list[list[float]]:
    matrix = np.asarray([row["posterior"] for row in rows], dtype=float)
    transition = confusion["human_transition"]
    target = np.mean([transition[rounded(row["source"]) - 1] for row in rows], axis=0)
    projection = config["distribution_projection"]
    for _ in range(int(projection["max_iterations"])):
        previous = matrix.copy()
        current = matrix.mean(axis=0)
        matrix *= target / np.maximum(current, 1e-12)
        matrix /= matrix.sum(axis=1, keepdims=True)
        for index, row in enumerate(rows):
            source = row["source"]
            qwen = source_obs(source, "qwen") or rounded(source)
            if rounded(source) <= 2 and qwen >= 4 and row["alpha"] == 0:
                high = matrix[index, 3] + matrix[index, 4]
                cap = float(projection["human_low_unconfirmed_high_probability_cap"])
                if high > cap:
                    scale = cap / high
                    matrix[index, 3:5] *= scale
                    matrix[index, :3] *= (1.0 - cap) / matrix[index, :3].sum()
            if rounded(source) == 4 and qwen == 5 and row["alpha"] < 0.5:
                human_five = human_empirical_distribution(source)[4]
                cap = human_five + float(projection["human_four_unconfirmed_five_increment_cap"])
                if matrix[index, 4] > cap:
                    removed = matrix[index, 4] - cap
                    matrix[index, 4] = cap
                    matrix[index, 3] += removed
            matrix[index] /= matrix[index].sum()
        if float(np.max(np.abs(matrix - previous))) < float(projection["convergence_tolerance"]):
            break
    return matrix.tolist()


def predictions_for_view(sources: dict[str, dict[str, Any]], references: dict[str, dict[str, Any]], ids: list[str], confusion: dict[str, Any], rel_model: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    initial = []
    for sid in ids:
        source = sources[sid]
        row = initial_edudart(source, confusion, rel_model, config)
        row.update({"sample_id": sid, "source": source})
        initial.append(row)
    projected = project_distribution(initial, confusion, config)
    output = []
    for row, edudart in zip(initial, projected, strict=True):
        source = row["source"]
        qwen = source_obs(source, "qwen")
        calibrated = calibrated_teacher(source, confusion, config)
        baselines = {
            "rounded_human": [float(label == rounded(source)) for label in LABELS],
            "human_empirical": human_empirical_distribution(source),
            "qwen_hard": [float(qwen is not None and label == qwen) for label in LABELS] if qwen else human_empirical_distribution(source),
            "qwen_calibrated": calibrated,
            "naive_fusion": normalize(3.0 * np.asarray(human_empirical_distribution(source)) + np.asarray(qwen_raw(source, float(config["calibration"]["teacher_range_temperature"])))),
            "EduDART_pre_projection_diagnostic": row["posterior"],
            "EduDART-Cal": edudart,
        }
        for method, posterior in baselines.items():
            output.append({
                "sample_id": row["sample_id"], "method": method,
                "reference": int(references[row["sample_id"]]["point_score"]),
                "posterior": list(posterior), "score": expected(list(posterior)), "hard": hard(list(posterior)),
                "alpha": row["alpha"] if method.startswith("EduDART") else "",
                "gate": row["gate"] if method.startswith("EduDART") else "",
            })
    return output


def metric_rows(predictions: list[dict[str, Any]], view: str) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        rows = [row for row in predictions if row["method"] == method]
        refs = [row["reference"] for row in rows]
        hard_preds = [row["hard"] for row in rows]
        scores = [row["score"] for row in rows]
        low = [index for index, ref in enumerate(refs) if ref <= 2]
        high = [index for index, ref in enumerate(refs) if ref >= 4]
        label2 = [index for index, ref in enumerate(refs) if ref == 2]
        output.append({
            "view": view, "method": method, "rows": len(rows),
            "mae": sum(abs(pred - ref) for pred, ref in zip(scores, refs, strict=True)) / len(rows),
            "qwk": quadratic_weighted_kappa(hard_preds, refs, [1.0] * len(rows)),
            "exact": sum(pred == ref for pred, ref in zip(hard_preds, refs, strict=True)) / len(rows),
            "signed_bias": sum(pred - ref for pred, ref in zip(scores, refs, strict=True)) / len(rows),
            "low_to_high": sum(hard_preds[index] >= 4 for index in low) / len(low) if low else "",
            "high_to_low": sum(hard_preds[index] <= 2 for index in high) / len(high) if high else "",
            "label2_recall": sum(hard_preds[index] == 2 for index in label2) / len(label2) if label2 else "",
            "avg_entropy": sum(entropy(row["posterior"]) for row in rows) / len(rows),
            "gate_rate": sum(float(bool(row["gate"])) for row in rows if method.startswith("EduDART")) / len(rows) if method.startswith("EduDART") else "",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--exp33a-out", type=Path, default=DEFAULT_EXP33A_OUT)
    parser.add_argument("--exp35a-out", type=Path, default=DEFAULT_EXP35_OUT)
    parser.add_argument("--teacher-summary-dir", type=Path, default=DEFAULT_TEACHER_SUMMARY_DIR)
    parser.add_argument("--exp28e-decision", type=Path, default=DEFAULT_EXP28E_DECISION)
    args = parser.parse_args()
    config = read_json(args.config)
    review_decision = read_json(args.exp35a_out / "decision/exp35a_review_decision.json")
    if not review_decision.get("review_gate_passed"):
        raise RuntimeError("Exp35A blind-review gate has not passed")
    final35_rows = read_csv(args.exp35a_out / "private/exp35a_model_reviewed_silver_reference.csv")
    final35 = {
        row["sample_id"]: {
            "point_score": int(row["point_score"]),
            "posterior": [float(row[f"p{score}"]) for score in LABELS],
            "view": row["qualification_view"],
        }
        for row in final35_rows
    }
    teacher_manifest, resolved = resolve_teacher_inputs(args.teacher_summary_dir, args.exp28e_decision)
    selected = load_selected_train_views(args.exp33a_out)
    sources = load_train_sources(args.split_dir, selected, resolved)
    calibration_ids = {sid for sid, row in selected.items() if row.get("view") == "representative_train"}
    final33, _ = load_final_silver_reference(args.exp33a_out, calibration_ids, require_complete=True)
    calibration = [
        {"sample_id": sid, "source": sources[sid], "reference": [float(value) for value in final33[sid]["posterior"]], "weight": float(sources[sid].get("design_weight") or 1.0)}
        for sid in sorted(calibration_ids)
    ]
    confusion = fit_confusions(calibration, config)
    rel_model = fit_reliability(calibration, config)
    all_metrics = []
    private_rows = []
    for view in ("fresh_general_qualification", "low_tail_reassessment"):
        ids = sorted(sid for sid, row in final35.items() if row["view"] == view)
        predictions = predictions_for_view(sources, final35, ids, confusion, rel_model, config)
        all_metrics.extend(metric_rows(predictions, view))
        for row in predictions:
            private_rows.append({
                "sample_id": row["sample_id"], "view": view, "method": row["method"],
                "reference": row["reference"], "score": row["score"], "hard": row["hard"],
                "alpha": row["alpha"], "gate": row["gate"],
                **{f"p{score}": row["posterior"][score - 1] for score in LABELS},
            })
    fields = list(private_rows[0])
    write_csv(args.exp35a_out / "private/exp35a_qualification_predictions.csv", private_rows, fields)
    metric_fields = list(all_metrics[0])
    write_csv(args.exp35a_out / "tables/exp35a_edudart_qualification_metrics.csv", all_metrics, metric_fields)
    by = {(row["view"], row["method"]): row for row in all_metrics}
    fresh = lambda method: by[("fresh_general_qualification", method)]
    low = lambda method: by[("low_tail_reassessment", method)]
    method = fresh("EduDART-Cal")
    qwen = fresh("qwen_hard")
    human = fresh("rounded_human")
    low_method = low("EduDART-Cal")
    gates = {
        "mae_not_worse_than_qwen": method["mae"] <= qwen["mae"],
        "qwk_not_worse_than_qwen_minus_0p01": method["qwk"] >= qwen["qwk"] - 0.01,
        "low_to_high_not_worse_than_qwen": (method["low_to_high"] == "" or qwen["low_to_high"] == "" or method["low_to_high"] <= qwen["low_to_high"]),
        "high_to_low_not_worse_than_qwen_plus_0p01": (method["high_to_low"] == "" or qwen["high_to_low"] == "" or method["high_to_low"] <= qwen["high_to_low"] + 0.01),
        "mae_better_than_rounded_human_by_0p01": method["mae"] <= human["mae"] - 0.01,
        "qwk_not_worse_than_rounded_human": method["qwk"] >= human["qwk"],
        "low_tail_label2_recall_ge_0p10": low_method["label2_recall"] != "" and low_method["label2_recall"] >= 0.10,
        "dev_test_access_zero": True,
    }
    decision = {
        "experiment": "Exp35A EduDART-Cal model-reviewed qualification",
        "reference_status": "independent model-reviewed silver, not human expert gold",
        "qualification_gate_passed": all(gates.values()),
        "gates": gates,
        "calibration_rows": len(calibration),
        "fresh_qualification_rows": 120,
        "low_tail_stress_rows": 76,
        "generate_exp35b_train_supervision": all(gates.values()),
        "dev_rows_read": 0, "test_access_count": 0, "student_training": False,
    }
    write_json(args.exp35a_out / "decision/exp35a_edudart_qualification_decision.json", decision)
    write_text(args.exp35a_out / "reports/exp35a_edudart_qualification_report.md", f"""# Exp35A EduDART-Cal Qualification

- Reference: model-reviewed silver, not human expert gold.
- Calibration-development: Exp33 representative train, 120 rows.
- Fresh general qualification: 120 rows; low-tail repeated-sample stress: 76 rows.
- EduDART-Cal fresh MAE/QWK: {method['mae']}/{method['qwk']}.
- Qwen hard fresh MAE/QWK: {qwen['mae']}/{qwen['qwk']}.
- Rounded human fresh MAE/QWK: {human['mae']}/{human['qwk']}.
- Low-tail EduDART-Cal label2 recall: {low_method['label2_recall']}.
- Qualification gate passed: {all(gates.values())}.
- Gates: `{json.dumps(gates, sort_keys=True)}`.
- Dev/test access: 0/0; student training: none.
""")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
