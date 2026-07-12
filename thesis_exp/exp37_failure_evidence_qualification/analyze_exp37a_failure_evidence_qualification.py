"""Analyze Exp37A reviews and train-only OOF utility after reviews are filled."""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import (  # noqa: E402
    FAILURE_CLASSES, OOF_PATH, QWEN_MANIFEST, ROOT, TRAIN_PATH, load_teacher_map, norm, normalize_failure,
    normalized_substring, packet_hash, read_jsonl, sample_id, sha256_text, write_csv, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--qwen-manifest", type=Path, default=QWEN_MANIFEST)
    parser.add_argument("--oof-predictions", type=Path, default=OOF_PATH)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-reviews", action="store_true")
    return parser.parse_args()


def load_packets(out_dir: Path) -> dict[str, dict[str, Any]]:
    packets = {}
    for path in sorted((out_dir / "private_packets").glob("exp37a_*_blind_packet.jsonl")):
        if "adjudication" in path.name:
            continue
        view = path.name.removeprefix("exp37a_").removesuffix("_blind_packet.jsonl")
        for row in read_jsonl(path):
            row["_view"] = view
            packets[str(row["sample_id"])] = row
    return packets


def load_optional(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def qwk(gold: list[int], pred: list[int]) -> float:
    if not gold or len(gold) != len(pred):
        return float("nan")
    n = len(gold)
    observed = [[0.0] * 5 for _ in range(5)]
    for left, right in zip(gold, pred):
        observed[left - 1][right - 1] += 1
    hist_g = [gold.count(i) for i in range(1, 6)]
    hist_p = [pred.count(i) for i in range(1, 6)]
    expected = [[hist_g[i] * hist_p[j] / n for j in range(5)] for i in range(5)]
    denom = sum(((i - j) ** 2) * expected[i][j] for i in range(5) for j in range(5))
    numer = sum(((i - j) ** 2) * observed[i][j] for i in range(5) for j in range(5))
    return 1.0 - numer / denom if denom else 1.0


def f1_scores(gold: list[set[str]], pred: list[set[str]], labels: list[str]) -> dict[str, float]:
    values = {}
    for label in labels:
        tp = sum(label in g and label in p for g, p in zip(gold, pred))
        fp = sum(label not in g and label in p for g, p in zip(gold, pred))
        fn = sum(label in g and label not in p for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values[label + "_precision"] = precision
        values[label + "_recall"] = recall
        values[label + "_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f1_values = [values[label + "_f1"] for label in labels]
    values["macro_f1"] = float(np.mean(f1_values)) if f1_values else 0.0
    values["micro_f1"] = _micro_f1(gold, pred, labels)
    return values


def _micro_f1(gold: list[set[str]], pred: list[set[str]], labels: list[str]) -> float:
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    fp = sum(len(p - g) for g, p in zip(gold, pred))
    fn = sum(len(g - p) for g, p in zip(gold, pred))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def qwen_failures(annotation: dict[str, Any]) -> set[str]:
    values = [normalize_failure(str(value)) for value in (annotation.get("major_failures") or ["unclear_or_other"])]
    return set(values) or {"unclear_or_other"}


def reviewer_agreement(a: list[dict[str, Any]], b: list[dict[str, Any]], packets: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bm = {str(row.get("sample_id")): row for row in b}
    rows = []
    score_gold: list[int] = []
    score_pred: list[int] = []
    class_gold: list[set[str]] = []
    class_pred: list[set[str]] = []
    for row in a:
        other = bm.get(str(row.get("sample_id")))
        if not other:
            continue
        score_a, score_b = int(row["most_plausible_score"]), int(other["most_plausible_score"])
        score_gold.append(score_a); score_pred.append(score_b)
        classes_a = {normalize_failure(x) for x in row["failure_classes"]}
        classes_b = {normalize_failure(x) for x in other["failure_classes"]}
        class_gold.append(classes_a); class_pred.append(classes_b)
        range_a, range_b = row["score_range"], other["score_range"]
        intersection = max(0, min(range_a[1], range_b[1]) - max(range_a[0], range_b[0]) + 1)
        union = max(range_a[1], range_b[1]) - min(range_a[0], range_b[0]) + 1
        evidence_a = {norm(x) for x in row.get("evaluator_output_evidence", [])}
        evidence_b = {norm(x) for x in other.get("evaluator_output_evidence", [])}
        rows.append({
            "sample_id_hash": packets[str(row["sample_id"])].get("anonymized_question_key_hash", sha256_text(row["sample_id"])),
            "score_exact": int(score_a == score_b), "score_within_one": int(abs(score_a - score_b) <= 1),
            "range_overlap": intersection / union if union else 0.0,
            "a_point_in_b_range": int(other["score_range"][0] <= score_a <= other["score_range"][1]),
            "b_point_in_a_range": int(row["score_range"][0] <= score_b <= row["score_range"][1]),
            "failure_bucket_agree": int(row["failure_bucket"] == other["failure_bucket"]),
            "failure_class_set_agree": int(classes_a == classes_b),
            "evidence_overlap": int(bool(evidence_a & evidence_b)),
            "needs_adjudication": int(row.get("needs_adjudication") is True or other.get("needs_adjudication") is True or score_a != score_b or classes_a != classes_b),
        })
    class_metrics = f1_scores(class_gold, class_pred, list(FAILURE_CLASSES))
    summary = [{
        "metric": "reviewer_agreement", "n": len(rows),
        "score_exact": float(np.mean([r["score_exact"] for r in rows])) if rows else None,
        "score_within_one": float(np.mean([r["score_within_one"] for r in rows])) if rows else None,
        "score_qwk": qwk(score_gold, score_pred),
        "score_range_overlap": float(np.mean([r["range_overlap"] for r in rows])) if rows else None,
        "point_within_other_range": float(np.mean([(r["a_point_in_b_range"] + r["b_point_in_a_range"]) / 2 for r in rows])) if rows else None,
        "failure_bucket_agreement": float(np.mean([r["failure_bucket_agree"] for r in rows])) if rows else None,
        "failure_class_set_agreement": float(np.mean([r["failure_class_set_agree"] for r in rows])) if rows else None,
        "evidence_overlap_rate": float(np.mean([r["evidence_overlap"] for r in rows])) if rows else None,
        "adjudication_rate": float(np.mean([r["needs_adjudication"] for r in rows])) if rows else None,
        **class_metrics,
    }]
    return summary, rows


def final_reference_metrics(reference: list[dict[str, Any]], packets: dict[str, dict[str, Any]], qwen: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    qgold: list[int] = []
    qpred: list[int] = []
    gold_sets: list[set[str]] = []
    qwen_sets: list[set[str]] = []
    rows = []
    for ref in reference:
        sid = str(ref.get("sample_id"))
        if sid not in packets or sid not in qwen:
            continue
        ann = qwen[sid]
        plausible = int(ref["most_plausible_score"])
        qscore = int(ann.get("score"))
        gold = {normalize_failure(x) for x in ref.get("failure_classes", [])}
        qfail = qwen_failures(ann)
        qgold.append(plausible); qpred.append(qscore); gold_sets.append(gold); qwen_sets.append(qfail)
        q_evidence = []
        for item in ann.get("rubric_assessment") or []:
            if isinstance(item, dict) and str(item.get("evidence") or "").strip():
                q_evidence.append(str(item["evidence"]))
        valid = all(normalized_substring(evidence, str(packets[sid].get("evaluator_output", ""))) for evidence in q_evidence)
        ref_evidence = {norm(x) for x in ref.get("evaluator_output_evidence", [])}
        support = bool(ref_evidence) and bool({norm(x) for x in q_evidence} & ref_evidence)
        rows.append({
            "sample_id": sid,
            "sample_id_hash": packets[sid].get("anonymized_question_key_hash"),
            "reference_score": plausible, "qwen_score": qscore,
            "score_abs_error": abs(qscore - plausible), "score_within_one": int(abs(qscore - plausible) <= 1),
            "reference_failure": "|".join(sorted(gold)), "qwen_failure": "|".join(sorted(qfail)),
            "failure_exact_set": int(gold == qfail), "failure_overlap": int(bool(gold & qfail)),
            "qwen_evidence_exact_substring_valid": int(valid), "qwen_evidence_supports_reference": int(support),
            "low_tail": int(plausible <= 2), "qwen_no_major_false_negative": int(gold != {"no_major_failure"} and qfail == {"no_major_failure"}),
        })
    class_metrics = f1_scores(gold_sets, qwen_sets, list(FAILURE_CLASSES))
    minority = f1_scores(
        [{x for x in values if x != "no_major_failure"} for values in gold_sets],
        [{x for x in values if x != "no_major_failure"} for values in qwen_sets],
        [x for x in FAILURE_CLASSES if x != "no_major_failure"],
    )
    metrics = [{
        "metric": "qwen_vs_final_silver", "n": len(rows),
        "score_mae": float(np.mean([r["score_abs_error"] for r in rows])) if rows else None,
        "score_within_one": float(np.mean([r["score_within_one"] for r in rows])) if rows else None,
        "score_qwk": qwk(qgold, qpred),
        "failure_micro_f1": class_metrics.get("micro_f1", 0.0), "failure_macro_f1": class_metrics.get("macro_f1", 0.0),
        "minority_failure_macro_f1": minority.get("macro_f1", 0.0),
        "low_tail_failure_recall": _low_tail_failure_recall(rows),
        "no_major_failure_false_negative_rate": float(np.mean([r["qwen_no_major_false_negative"] for r in rows])) if rows else None,
        "evidence_exact_substring_validity": float(np.mean([r["qwen_evidence_exact_substring_valid"] for r in rows])) if rows else None,
        "evidence_support_agreement": float(np.mean([r["qwen_evidence_supports_reference"] for r in rows])) if rows else None,
    }]
    per_class = [{"class": name, **{key: value for key, value in class_metrics.items() if key.startswith(name + "_")}} for name in FAILURE_CLASSES]
    return metrics, per_class, rows


def qwen_evidence_metrics(rows: list[dict[str, Any]], qwen: dict[str, dict[str, Any]], packets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows or not qwen:
        return [{"status": "MISSING_COMPLETE_REVIEWS_OR_QWEN"}]
    validity = []
    structural_gate = []
    for row in rows:
        sid = str(row.get("sample_id"))
        if sid in qwen and sid in packets:
            ann = qwen[sid]
            evidence = [str(item.get("evidence")) for item in ann.get("rubric_assessment") or [] if isinstance(item, dict) and str(item.get("evidence") or "").strip()]
            validity.append(all(normalized_substring(value, str(packets[sid].get("evaluator_output", ""))) for value in evidence))
            structural_gate.append(bool(str(ann.get("reason") or "").strip()) and bool(ann.get("rubric_assessment")) and 1 <= int(ann.get("score", 0)) <= 5)
    return [{"metric": "qwen_structural_evidence_gate", "n": len(validity), "gate_rate": float(np.mean(structural_gate)) if structural_gate else None, "exact_substring_validity": float(np.mean(validity)) if validity else None}]


def _low_tail_failure_recall(rows: list[dict[str, Any]]) -> float:
    low = [row for row in rows if row["low_tail"]]
    if not low:
        return 0.0
    return sum(row["failure_overlap"] for row in low) / len(low)


def baseline_failure_metrics(reference: list[dict[str, Any]], packets: dict[str, dict[str, Any]], qwen: dict[str, dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    pairs = []
    for ref in reference:
        sid = str(ref.get("sample_id"))
        if sid not in qwen or sid not in packets:
            continue
        pairs.append((sid, {normalize_failure(x) for x in ref.get("failure_classes", [])}, qwen_failures(qwen[sid])))
    if not pairs:
        return [{"baseline": "all", "status": "MISSING_COMPLETE_REVIEWS_OR_QWEN"}]
    gold = [item[1] for item in pairs]
    always = [{"no_major_failure"} for _ in pairs]
    majority_counts = {label: sum(label in values for values in gold) for label in FAILURE_CLASSES}
    majority = max(FAILURE_CLASSES, key=lambda label: (majority_counts[label], label))
    prior = [{majority} for _ in pairs]
    qwen_sets = [item[2] for item in pairs]
    rng = random.Random(seed)
    shuffled = list(qwen_sets); rng.shuffle(shuffled)
    rows = []
    for name, prediction in (("always_no_major_failure", always), ("failure_class_prior", prior), ("shuffled_qwen_failure", shuffled)):
        values = f1_scores(gold, prediction, list(FAILURE_CLASSES))
        rows.append({"baseline": name, "n": len(pairs), "majority_class": majority if name == "failure_class_prior" else "", "failure_micro_f1": values["micro_f1"], "failure_macro_f1": values["macro_f1"], "minority_failure_macro_f1": f1_scores([{x for x in values if x != "no_major_failure"} for values in gold], [{x for x in values if x != "no_major_failure"} for values in prediction], [x for x in FAILURE_CLASSES if x != "no_major_failure"])["macro_f1"]})
    gate_values = []
    for sid, _, _ in pairs:
        ann = qwen[sid]
        gate_values.append(bool(str(ann.get("reason") or "").strip()) and bool(ann.get("rubric_assessment")) and 1 <= int(ann.get("score", 0)) <= 5)
    rows.append({"baseline": "current_exp36_structural_evidence_gate", "n": len(gate_values), "gate_rate": float(np.mean(gate_values)), "note": "Gate structure is not a failure-class predictor."})
    rows.append({"baseline": "human_label_or_entropy_only", "status": "NOT_A_FAILURE_LABEL_PREDICTOR_IN_BLIND_REFERENCE", "note": "Human score/entropy are evaluated as OOF scalar signals, not failure-class predictions."})
    return rows


def average_precision(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    if not positives:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    hit = 0; total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            hit += 1; total += hit / rank
    return total / positives


def utility_analysis(reference: list[dict[str, Any]], packets: dict[str, dict[str, Any]], qwen: dict[str, dict[str, Any]], train: dict[str, dict[str, Any]], oof_path: Path, seed: int, resamples: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not oof_path.exists() or not reference:
        missing = [{"status": "MISSING_OOF_OR_REFERENCE", "oof_exists": oof_path.exists(), "reference_rows": len(reference), "test_access_count": 0}]
        return missing, missing, missing
    oof = {sample_id(row): row for row in read_jsonl(oof_path)}
    reviewed = []
    for ref in reference:
        sid = str(ref.get("sample_id"))
        if sid not in oof or sid not in qwen or sid not in packets:
            continue
        pred = oof[sid].get("pred_label", oof[sid].get("pred_score", oof[sid].get("prediction")))
        if pred is None:
            probs = [float(oof[sid].get("prob_" + str(label), 0.0)) for label in range(1, 6)]
            pred = int(np.argmax(probs)) + 1 if any(probs) else None
        if pred is None:
            continue
        try: pred = int(round(float(pred)))
        except (TypeError, ValueError): continue
        qfail = qwen_failures(qwen[sid])
        ref_fail = {normalize_failure(x) for x in ref.get("failure_classes", [])}
        train_row = train.get(sid, {})
        human_label = int(round(float(train_row.get("label_5", ref["most_plausible_score"]))))
        human_scores = [int(round(float(train_row.get("human_1", human_label)))), int(round(float(train_row.get("human_2", human_label)))), int(round(float(train_row.get("human_3", human_label))))]
        counts = [human_scores.count(label) / 3.0 for label in range(1, 6)]
        entropy = -sum(value * math.log(value) for value in counts if value > 0) / math.log(5)
        qwen_score = int(qwen[sid].get("score", human_label))
        reviewed.append({
            "sample_id": sid, "question_key": packets[sid].get("anonymized_question_key_hash"),
            "gold": human_label, "review_score": int(ref["most_plausible_score"]), "oof_pred": pred,
            "severe": int(abs(pred - human_label) >= 2),
            "low_to_high": int(human_label <= 2 and pred >= 4),
            "qwen_signal": int(qfail != {"no_major_failure"}),
            "verified_signal": int(ref_fail != {"no_major_failure"}),
            "human_entropy": entropy, "qwen_human_gap": abs(qwen_score - human_label),
            "language": packets[sid].get("non_label_metadata", {}).get("language"),
            "metric_group": packets[sid].get("non_label_metadata", {}).get("metric_group"),
        })
    if not reviewed:
        missing = [{"status": "NO_ALIGNED_OOF_ROWS", "oof_exists": True, "reference_rows": len(reference), "test_access_count": 0}]
        return missing, missing, missing
    rng = random.Random(seed)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in reviewed:
        groups[(row["gold"], row["language"], row["metric_group"])].append(row)
    shuffled_by_id = {}
    for values in groups.values():
        source = [row["verified_signal"] for row in values]; rng.shuffle(source)
        for row, value in zip(values, source): shuffled_by_id[row["sample_id"]] = value
    for row in reviewed: row["shuffled_verified_signal"] = shuffled_by_id[row["sample_id"]]
    signals = [("qwen_raw_failure", "qwen_signal"), ("aligned_verified_failure", "verified_signal"), ("shuffled_verified_failure", "shuffled_verified_signal"), ("human_entropy", "human_entropy"), ("qwen_human_score_gap", "qwen_human_gap")]
    frontier = []
    for name, field in signals:
        for target in ("severe", "low_to_high"):
            frontier.append({"signal": name, "target": target, "n": len(reviewed), "AUPRC": average_precision([row[field] for row in reviewed], [row[target] for row in reviewed]), "top10_recall": _top_recall(reviewed, field, target, 0.10), "top20_recall": _top_recall(reviewed, field, target, 0.20), "odds_ratio": _odds_ratio(reviewed, field, target)})
    aligned = next((row["AUPRC"] for row in frontier if row["signal"] == "aligned_verified_failure" and row["target"] == "severe"), 0.0)
    shuffled = next((row["AUPRC"] for row in frontier if row["signal"] == "shuffled_verified_failure" and row["target"] == "severe"), 0.0)
    ci = _bootstrap_signal_diff(reviewed, "verified_signal", "shuffled_verified_signal", "severe", resamples, seed)
    bootstrap_rows = [{"signal": "aligned_verified_failure_minus_shuffled_verified_failure", "target": "severe", "aligned_auprc": aligned, "shuffled_auprc": shuffled, "difference": aligned - shuffled, **ci, "cluster_unit": "question_key_hash", "resamples": resamples}]
    utility = [{"metric": "reviewed_oof_utility", **row} for row in frontier]
    return utility, bootstrap_rows, reviewed


def _top_recall(rows: list[dict[str, Any]], field: str, target: str, fraction: float) -> float:
    n = max(1, math.ceil(len(rows) * fraction)); selected = sorted(rows, key=lambda row: -row[field])[:n]
    denom = sum(row[target] for row in rows)
    return sum(row[target] for row in selected) / denom if denom else 0.0


def _odds_ratio(rows: list[dict[str, Any]], field: str, target: str) -> float:
    a = sum(row[field] and row[target] for row in rows); b = sum(row[field] and not row[target] for row in rows)
    c = sum(not row[field] and row[target] for row in rows); d = sum(not row[field] and not row[target] for row in rows)
    return (a * d) / (b * c) if b and c else (float("inf") if a and not b and c else 0.0)


def _bootstrap_signal_diff(rows: list[dict[str, Any]], left: str, right: str, target: str, resamples: int, seed: int) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[str(row["question_key"])] .append(row)
    keys = sorted(groups); rng = np.random.default_rng(seed); values = []
    for _ in range(resamples):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        selected = [row for key in sampled for row in groups[str(key)]]
        values.append(average_precision([row[left] for row in selected], [row[target] for row in selected]) - average_precision([row[right] for row in selected], [row[target] for row in selected]))
    return {"ci_lower_95": float(np.quantile(values, 0.025)), "ci_upper_95": float(np.quantile(values, 0.975)), "bootstrap_mean": float(np.mean(values))}


def range_metrics(a: list[dict[str, Any]], b: list[dict[str, Any]], c: list[dict[str, Any]], packets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    final = {str(row.get("sample_id")): row for row in c}
    am = {str(row.get("sample_id")): row for row in a}; bm = {str(row.get("sample_id")): row for row in b}
    rows = []
    for sid, ref in final.items():
        if sid not in am or sid not in bm: continue
        for view in ("low_tail_all", "boundary_view", "high_control_view"):
            # View is intentionally derived from the packet, never from a hidden label.
            if packets.get(sid, {}).get("_view") != view: continue
            ar, br = am[sid]["score_range"], bm[sid]["score_range"]
            rows.append({"view": view, "sample_id_hash": packets[sid].get("anonymized_question_key_hash"), "range_overlap": int(max(ar[0], br[0]) <= min(ar[1], br[1])), "range_iou": (max(0, min(ar[1], br[1]) - max(ar[0], br[0]) + 1) / (max(ar[1], br[1]) - min(ar[0], br[0]) + 1)), "final_non_singleton": int(ref["score_range"][0] != ref["score_range"][1]), "a_point_in_b": int(br[0] <= am[sid]["most_plausible_score"] <= br[1]), "b_point_in_a": int(ar[0] <= bm[sid]["most_plausible_score"] <= ar[1])})
    if not rows: return [{"status": "MISSING_COMPLETE_REVIEWS"}]
    summary = []
    for view in ("low_tail_all", "boundary_view", "high_control_view"):
        subset = [row for row in rows if row["view"] == view]
        widths = []
        for sid, ref in final.items():
            if sid in packets and packets[sid].get("_view") == view:
                widths.append(int(ref["score_range"][1]) - int(ref["score_range"][0]) + 1)
        summary.append({"view": view, "n": len(subset), "range_overlap_rate": float(np.mean([row["range_overlap"] for row in subset])) if subset else None, "mean_range_iou": float(np.mean([row["range_iou"] for row in subset])) if subset else None, "final_non_singleton_rate": float(np.mean([row["final_non_singleton"] for row in subset])) if subset else None, "mean_final_range_width": float(np.mean(widths)) if widths else None, "median_final_range_width": float(np.median(widths)) if widths else None, "point_within_other_range_rate": float(np.mean([(row["a_point_in_b"] + row["b_point_in_a"]) / 2 for row in subset])) if subset else None})
    return summary


def main() -> None:
    args = parse_args()
    packets = load_packets(args.out_dir)
    train = {sample_id(row): row for row in read_jsonl(args.train_jsonl)}
    a = load_optional(args.reviewer_a or args.out_dir / "private_reviews/reviewer_a_filled.jsonl")
    b = load_optional(args.reviewer_b or args.out_dir / "private_reviews/reviewer_b_filled.jsonl")
    c = load_optional(args.adjudication or args.out_dir / "private_reviews/adjudication_filled.jsonl")
    qwen, qwen_source = load_teacher_map(args.qwen_manifest)
    agreement = [{"status": "MISSING_REVIEWS", "reviewer_a_rows": len(a), "reviewer_b_rows": len(b), "adjudication_rows": len(c)}]
    agreement_rows = []
    if len(a) == len(packets) and len(b) == len(packets):
        agreement, agreement_rows = reviewer_agreement(a, b, packets)
    write_csv(args.out_dir / "tables/exp37a_reviewer_agreement.csv", agreement)
    if agreement_rows:
        write_csv(args.out_dir / "private_reference/reviewer_pair_rows.csv", agreement_rows)
    metrics, per_class, qwen_rows = ([], [], [])
    if len(c) == len(packets):
        metrics, per_class, qwen_rows = final_reference_metrics(c, packets, qwen)
    else:
        metrics = [{"status": "WAITING_FOR_COMPLETE_ADJUDICATION", "qwen_source": qwen_source, "reviewer_a_rows": len(a), "reviewer_b_rows": len(b), "adjudication_rows": len(c)}]
        per_class = [{"status": "WAITING_FOR_COMPLETE_ADJUDICATION"}]
    write_csv(args.out_dir / "tables/exp37a_qwen_failure_metrics.csv", metrics)
    evidence_metrics = qwen_evidence_metrics(qwen_rows, qwen, packets) if qwen_rows else [{"status": "WAITING_FOR_COMPLETE_ADJUDICATION"}]
    write_csv(args.out_dir / "tables/exp37a_qwen_evidence_metrics.csv", evidence_metrics)
    if qwen_rows:
        write_csv(args.out_dir / "private_reference/qwen_vs_reference_rows.csv", qwen_rows)
    baseline_rows = baseline_failure_metrics(c, packets, qwen, args.seed) if len(c) == len(packets) else [{"baseline": "all", "status": "WAITING_FOR_COMPLETE_ADJUDICATION"}]
    write_csv(args.out_dir / "tables/exp37a_failure_class_metrics.csv", per_class + baseline_rows)
    utility, bootstrap, utility_rows = utility_analysis(c, packets, qwen, train, args.oof_predictions, args.seed, args.bootstrap_resamples) if len(c) == len(packets) else ([{"status": "WAITING_FOR_COMPLETE_ADJUDICATION"}], [{"status": "WAITING_FOR_COMPLETE_ADJUDICATION"}], [])
    write_csv(args.out_dir / "tables/exp37a_oof_utility_metrics.csv", utility)
    write_csv(args.out_dir / "tables/exp37a_oof_review_frontier.csv", utility)
    write_csv(args.out_dir / "tables/exp37a_question_key_bootstrap_ci.csv", bootstrap)
    write_csv(args.out_dir / "private_reference/oof_review_rows.csv", utility_rows)
    range_rows = range_metrics(a, b, c, packets) if len(c) == len(packets) and len(a) == len(packets) and len(b) == len(packets) else [{"status": "WAITING_FOR_COMPLETE_REVIEWS"}]
    write_csv(args.out_dir / "tables/exp37a_score_range_qualification.csv", range_rows)
    # The V7 audit is a separate script and may be run independently; this
    # report records its location without pretending it has run.
    semantic = metrics[0] if metrics and metrics[0].get("metric") == "qwen_vs_final_silver" else {}
    utility_diff = bootstrap[0] if bootstrap and bootstrap[0].get("difference") is not None else {}
    complete = len(a) == len(packets) == len(b) == len(c) and len(packets) == 196
    agreement_summary = agreement[0] if agreement and agreement[0].get("metric") == "reviewer_agreement" else {}
    adjudication_complete = complete and all(
        row.get("reference_type") == "human_rationale_grounded_model_reviewed_silver"
        and str(row.get("adjudication_reason") or "").strip()
        for row in c
    )
    core_gate = complete and (
        float(agreement_summary.get("score_qwk", -1) or -1) >= 0.65
        or adjudication_complete
    ) and float(agreement_summary.get("failure_bucket_agreement", 0) or 0) >= 0.70
    aligned_severe = next((row for row in utility if row.get("signal") == "aligned_verified_failure" and row.get("target") == "severe"), {})
    semantic_gate = complete and float(semantic.get("minority_failure_macro_f1", 0)) >= 0.50 and float(semantic.get("low_tail_failure_recall", 0)) >= 0.70 and float(semantic.get("evidence_exact_substring_validity", 0)) >= 0.90 and float(semantic.get("evidence_support_agreement", 0)) >= 0.75 and float(semantic.get("no_major_failure_false_negative_rate", 1)) <= 0.20
    utility_gate = complete and float(utility_diff.get("difference", -1)) >= 0.05 and float(utility_diff.get("ci_lower_95", -1)) > 0 and float(aligned_severe.get("odds_ratio", 0) or 0) >= 2.0
    range_gate = complete and bool(range_rows) and all(
        float(row.get("range_overlap_rate", 0) or 0) >= 0.85
        and float(row.get("point_within_other_range_rate", 0) or 0) >= 0.90
        and float(row.get("mean_final_range_width", 999) or 999) <= 2.0
        and float(row.get("final_non_singleton_rate", 0) or 0) >= 0.15
        for row in range_rows if row.get("view")
    )
    decision = {
        "status": "GO" if core_gate and semantic_gate and utility_gate else ("WAITING_FOR_REVIEWS" if not complete else "NO_GO"),
        "reference_complete": complete,
        "recommend_new_reason_evidence_training": bool(core_gate and semantic_gate and utility_gate),
        "recommend_full_train_score_range_annotation": bool(range_gate),
        "recommend_student_training": False,
        "stop_reason_evidence_supervision": bool(complete and not (core_gate and semantic_gate and utility_gate)),
        "qwen_source": qwen_source, "adjudication_complete": adjudication_complete, "core_reference_gate": core_gate, "semantic_gate": semantic_gate, "utility_gate": utility_gate, "score_range_gate": range_gate,
        "reason": "No semantic decision is made before complete independent reviews and adjudication." if not complete else "Decision follows preregistered semantic and utility gates.",
        "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp37a_failure_evidence_qualification_decision.json", decision)
    report = [
        "# Exp37A failure-evidence qualification report", "",
        "## Status", f"- Reference complete: `{complete}`", f"- Reviewer A/B/C rows: `{len(a)}/{len(b)}/{len(c)}`", f"- Qwen source: `{qwen_source}`",
        "- This analysis never reads dev/test, trains a student, runs student inference, or calls an API.", "",
        "## Interpretation", "Before complete blind reviews and adjudication, semantic metrics are intentionally unavailable and GO is false.",
        "After completion, interpret Qwen evidence only through the preregistered minority-F1, low-tail-recall, evidence-support, and aligned-vs-shuffled OOF gates.", "",
        "## Decision", f"- recommend_new_reason_evidence_training: `{decision['recommend_new_reason_evidence_training']}`", f"- recommend_full_train_score_range_annotation: `{decision['recommend_full_train_score_range_annotation']}`", f"- stop_reason_evidence_supervision: `{decision['stop_reason_evidence_supervision']}`", "",
        "## Boundary", "Only paper-like train rows and train-only OOF predictions are allowed. No dev/test labels, predictions, or metrics are used.",
    ]
    (args.out_dir / "reports/exp37a_failure_evidence_qualification_report.md").parent.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reports/exp37a_failure_evidence_qualification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    main()
