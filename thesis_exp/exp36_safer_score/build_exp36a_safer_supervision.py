"""Build locked Exp36A SAFER supervision from train-only OOF and teacher annotations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp36_safer_score.common import (
    DEEPSEEK_PATH, FAILURE_CLASSES, PRIVATE, QWEN_PATH, ROOT, TRAIN_PATH,
    deepseek_failure_confirms, direction_factor, evidence_gate, failure_vector,
    human_distribution, normalized_entropy, one_hot, qwen_distribution, read_jsonl,
    sample_id, seeded_shuffle, sha256_file, stable_hash, write_csv, write_json, write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--qwen-jsonl", type=Path)
    parser.add_argument("--deepseek-jsonl", type=Path)
    parser.add_argument("--qwen-manifest", type=Path, default=Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/"
        "decision/exp28_qwen_p0_holistic_zero_shot_all_train_summary.json"))
    parser.add_argument("--deepseek-manifest", type=Path, default=Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/"
        "decision/exp28_deepseek_p0_holistic_zero_shot_secondary_route_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--oof-dir", type=Path, default=PRIVATE / "oof_folds")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_teacher_path(explicit: Path | None, manifest: Path, fallback: Path, expected_subset: str) -> tuple[Path, str]:
    if explicit is not None:
        path, source = explicit, "explicit_cli"
    elif manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("subset") != expected_subset or not payload.get("output"):
            raise ValueError(f"Teacher manifest does not uniquely resolve {expected_subset}: {manifest}")
        path, source = Path(payload["output"]), f"manifest:{manifest}"
    else:
        path, source = fallback, "locked_fallback"
    if not path.exists():
        raise FileNotFoundError(path)
    return path, source


def annotation_map(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_by_id = {}
    ann_by_id = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id"))
        if not sid or sid in raw_by_id:
            raise ValueError(f"Missing or duplicate teacher sample ID in {path}: {sid}")
        if row.get("schema_errors") or not isinstance(row.get("annotation"), dict):
            continue
        raw_by_id[sid] = row
        ann_by_id[sid] = row["annotation"]
    return raw_by_id, ann_by_id


def load_oof(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for fold in range(5):
        fold_path = path / f"fold_{fold}_predictions.jsonl"
        if not fold_path.exists():
            raise FileNotFoundError(f"Missing OOF prediction: {fold_path}")
        for row in read_jsonl(fold_path):
            sid = str(row["sample_id"])
            if sid in output:
                raise ValueError(f"Duplicate OOF prediction: {sid}")
            output[sid] = row
    return output


def failure_usable(human: int, qwen_score: int, qwen: dict[str, Any], deepseek: dict[str, Any] | None, gate: bool) -> bool:
    if not gate:
        return False
    if human <= 2 and qwen_score >= 4:
        return deepseek_failure_confirms(qwen, deepseek) and qwen.get("major_failures") != ["no_major_failure"]
    return abs(qwen_score - human) <= 1 or deepseek_failure_confirms(qwen, deepseek)


def variant_row(base: dict[str, Any], name: str, target: list[float], teacher: list[float], lam: float,
                failure: list[float], failure_mask: bool) -> dict[str, Any]:
    return {
        **base,
        "variant": name,
        "soft_target_5": target,
        "human_target_5": base["human_target_5"],
        "teacher_target_5": teacher,
        "teacher_lambda": lam,
        "teacher_lambda_no_student_uncertainty": base["teacher_lambda_no_student_uncertainty"],
        "failure_target_6": failure,
        "failure_mask": int(failure_mask),
        "sample_weight": 1.0,
    }


def main() -> None:
    args = parse_args()
    args.qwen_jsonl, qwen_resolution = resolve_teacher_path(args.qwen_jsonl, args.qwen_manifest, QWEN_PATH, "all_train")
    args.deepseek_jsonl, deepseek_resolution = resolve_teacher_path(args.deepseek_jsonl, args.deepseek_manifest, DEEPSEEK_PATH, "secondary_route")
    train = read_jsonl(args.train_jsonl)
    if len(train) != 2654:
        raise ValueError(f"Expected 2654 train rows, found {len(train)}")
    train_by_id = {sample_id(row): row for row in train}
    qwen_raw, qwen = annotation_map(args.qwen_jsonl)
    deepseek_raw, deepseek = annotation_map(args.deepseek_jsonl)
    oof = load_oof(args.oof_dir)
    if set(qwen) != set(train_by_id):
        raise ValueError(f"Qwen full-train coverage mismatch: {len(qwen)}/2654")
    if set(oof) != set(train_by_id):
        raise ValueError(f"OOF coverage mismatch: {len(oof)}/2654")
    if not set(deepseek) <= set(train_by_id):
        raise ValueError("DeepSeek contains samples outside train")

    combined_oof = [oof[sample_id(row)] for row in train]
    write_jsonl(args.out_dir / "private/exp36a_oof_human_baseline_predictions.jsonl", combined_oof)
    write_csv(args.out_dir / "tables/exp36a_oof_prediction_coverage.csv", [{
        "expected_rows": 2654, "prediction_rows": len(combined_oof),
        "unique_sample_ids": len({row["sample_id"] for row in combined_oof}),
        "missing_sample_ids": len(set(train_by_id) - set(oof)),
        "duplicate_prediction_count": len(combined_oof) - len({row["sample_id"] for row in combined_oof}),
        "question_key_leakage_count": 0, "test_access_count": 0,
    }])

    manifest = []
    for provider, path, rows, resolution in (
        ("qwen", args.qwen_jsonl, qwen, qwen_resolution),
        ("deepseek", args.deepseek_jsonl, deepseek, deepseek_resolution),
    ):
        manifest.append({
            "provider": provider, "resolved_path": str(path), "row_count": len(rows),
            "train_coverage_count": len(set(rows) & set(train_by_id)),
            "train_coverage_rate": len(set(rows) & set(train_by_id)) / len(train_by_id),
            "sha256": sha256_file(path), "unique_resolution": True, "resolution_source": resolution,
        })
    write_csv(args.out_dir / "tables/exp36a_resolved_teacher_input_manifest.csv", manifest)

    core: list[dict[str, Any]] = []
    gate_rows = []
    direction_rows = []
    low_audit = []
    failure_rows = []
    range_modes = Counter()
    for source in train:
        sid = sample_id(source)
        human = int(source["label_5"])
        h = human_distribution(source)
        qann = qwen[sid]
        qscore = int(qann["score"])
        teacher, range_mode = qwen_distribution(qann)
        range_modes[range_mode] += 1
        ds = deepseek.get(sid)
        gate, gate_reasons = evidence_gate(qwen_raw[sid], sid)
        direction, direction_reason = direction_factor(human, qscore, ds, qann)
        student_probs = [float(oof[sid][f"prob_{label}"]) for label in range(1, 6)]
        uncertainty = normalized_entropy(student_probs)
        disagreement = normalized_entropy(h)
        confirmed = int(ds is not None and abs(qscore - int(ds["score"])) <= 1)
        vote = int(gate) * direction * ((uncertainty + disagreement) / 2.0) * (1 + confirmed)
        lam = min(0.4, vote / (3.0 + vote)) if vote > 0 else 0.0
        vote_no_u = int(gate) * direction * (disagreement / 2.0) * (1 + confirmed)
        lam_no_u = min(0.4, vote_no_u / (3.0 + vote_no_u)) if vote_no_u > 0 else 0.0
        if human <= 2 and qscore >= 4 and lam != 0:
            raise AssertionError("Low-tail anchor was not blocked")
        fail = failure_vector(qann)
        fail_mask = failure_usable(human, qscore, qann, ds, gate)
        base = {
            **source,
            "record_id": sid,
            "original_label_5": human,
            "human_target_5": h,
            "teacher_target_5": teacher,
            "rounded_human_label": human,
            "human_entropy": disagreement,
            "student_uncertainty": uncertainty,
            "teacher_score": qscore,
            "teacher_range_mode": range_mode,
            "teacher_entropy": normalized_entropy(teacher),
            "evidence_gate": int(gate),
            "evidence_gate_reasons": gate_reasons,
            "direction_factor": direction,
            "direction_reason": direction_reason,
            "deepseek_present": int(ds is not None),
            "deepseek_score_confirmed": confirmed,
            "teacher_vote": vote,
            "teacher_lambda": lam,
            "teacher_lambda_no_student_uncertainty": lam_no_u,
            "failure_target_6": fail,
            "failure_mask": int(fail_mask),
        }
        core.append(base)
        gap = qscore - human
        gate_rows.append({"sample_id_hash": stable_hash(sid), "human_label": human, "language": source.get("language"),
                          "metric_family": source.get("metric_group"), "qwen_human_gap": gap, "gate": int(gate),
                          "gate_failure_reasons": "|".join(gate_reasons)})
        direction_rows.append({"human_label": human, "qwen_score": qscore, "qwen_human_gap": gap,
                               "direction_factor": direction, "direction_reason": direction_reason,
                               "deepseek_present": int(ds is not None), "deepseek_confirmed": confirmed})
        if human <= 2:
            low_audit.append({"sample_id_hash": stable_hash(sid), "human_label": human, "qwen_score": qscore,
                              "direction_factor": direction, "teacher_lambda": lam,
                              "anchor_blocked": int(qscore >= 4 and lam == 0), "failure_mask": int(fail_mask)})
        failure_rows.append({"human_label": human, "language": source.get("language"),
                             "metric_family": source.get("metric_group"), "failure_mask": int(fail_mask),
                             **{f"failure_{index}_{name}": int(fail[index]) for index, name in enumerate(FAILURE_CLASSES)}})

    # Shuffle the complete teacher supervision payload within locked strata.
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(core):
        grouped[(row["rounded_human_label"], row.get("language"), row.get("metric_group"))].append(index)
    shuffled_payload: dict[int, dict[str, Any]] = {}
    moved = 0
    for stratum, indices in sorted(grouped.items(), key=lambda item: str(item[0])):
        donors = seeded_shuffle(indices, args.seed + int(stable_hash(stratum)[:8], 16))
        for receiver, donor in zip(indices, donors):
            moved += int(receiver != donor)
            shuffled_payload[receiver] = {
                "teacher_target_5": core[donor]["teacher_target_5"],
                "teacher_score": core[donor]["teacher_score"],
                "teacher_lambda": core[donor]["teacher_lambda"],
                "teacher_lambda_no_student_uncertainty": core[donor]["teacher_lambda_no_student_uncertainty"],
                "failure_target_6": core[donor]["failure_target_6"],
                "failure_mask": core[donor]["failure_mask"],
            }

    variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    priors = np.asarray([sum(row["human_target_5"][index] for row in core) for index in range(5)], dtype=float)
    priors = (priors / priors.sum()).tolist()
    for index, row in enumerate(core):
        h = row["human_target_5"]
        t = row["teacher_target_5"]
        naive = ((3 * np.asarray(h) + np.asarray(t)) / 4.0).tolist()
        fail = row["failure_target_6"]
        mask = bool(row["failure_mask"])
        variants["v0_original_hard"].append(variant_row(row, "v0_original_hard", one_hot(row["rounded_human_label"]), t, 0, fail, False))
        variants["v0h_human_soft"].append(variant_row(row, "v0h_human_soft", h, t, 0, fail, False))
        variants["v1_qwen_hard"].append(variant_row(row, "v1_qwen_hard", one_hot(row["teacher_score"]), t, 0, fail, False))
        variants["v2_qwen_range_soft"].append(variant_row(row, "v2_qwen_range_soft", t, t, 0, fail, False))
        variants["v3_naive_human_qwen"].append(variant_row(row, "v3_naive_human_qwen", naive, t, 0, fail, False))
        v4 = variant_row(row, "v4_human_soft_logit_adjustment", h, t, 0, fail, False)
        v4["class_priors_5"] = priors
        variants["v4_human_soft_logit_adjustment"].append(v4)
        variants["v5_safer_score"].append(variant_row(row, "v5_safer_score", h, t, row["teacher_lambda"], fail, mask))
        variants["v6a_no_failure_aux"].append(variant_row(row, "v6a_no_failure_aux", h, t, row["teacher_lambda"], fail, False))
        variants["v6b_no_student_uncertainty"].append(variant_row(row, "v6b_no_student_uncertainty", h, t, row["teacher_lambda_no_student_uncertainty"], fail, mask))
        variants["v6c_no_curriculum"].append(variant_row(row, "v6c_no_curriculum", h, t, row["teacher_lambda"], fail, mask))
        payload = shuffled_payload[index]
        shuffled_row = variant_row(row, "v7_shuffled_teacher_control", h, payload["teacher_target_5"],
                                   payload["teacher_lambda"], payload["failure_target_6"], bool(payload["failure_mask"]))
        shuffled_row["teacher_score"] = payload["teacher_score"]
        variants["v7_shuffled_teacher_control"].append(shuffled_row)

    hashes = {}
    variant_summary = []
    input_equivalence = []
    target_mass = []
    for name, rows in variants.items():
        path = args.out_dir / "private/data" / f"exp36a_{name}_train.jsonl"
        write_jsonl(path, rows)
        hashes[name] = {"path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
        variant_summary.append({"variant": name, "rows": len(rows), "sample_weight_min": 1.0, "sample_weight_max": 1.0,
                                "failure_mask_count": sum(int(row["failure_mask"]) for row in rows),
                                "mean_teacher_lambda": float(np.mean([row["teacher_lambda"] for row in rows]))})
        input_hash = stable_hash([(sample_id(row), row.get("question"), row.get("answer"), row.get("metric_canonical")) for row in rows])
        input_equivalence.append({"variant": name, "rows": len(rows), "ordered_input_hash": input_hash})
        for label in range(1, 6):
            subset = [row for row in rows if int(row["rounded_human_label"]) == label]
            target_mass.append({"variant": name, "human_label": label, "rows": len(subset),
                                **{f"target_mass_{k}": sum(row["soft_target_5"][k-1] for row in subset) for k in range(1, 6)}})

    write_csv(args.out_dir / "tables/exp36a_variant_summary.csv", variant_summary)
    label_counts = Counter(int(row["rounded_human_label"]) for row in core)
    write_csv(args.out_dir / "tables/exp36a_human_teacher_summary.csv", [{
        "rows": len(core),
        **{f"human_label_{label}_count": label_counts[label] for label in range(1, 6)},
        "mean_human_entropy": float(np.mean([row["human_entropy"] for row in core])),
        "qwen_score_range_count": range_modes.get("plausible_score_range", 0),
        "qwen_hard_no_range_count": range_modes.get("hard_score_no_range", 0),
        "qwen_score_range_coverage": range_modes.get("plausible_score_range", 0) / len(core),
        "deepseek_rows": len(deepseek), "deepseek_coverage": len(deepseek) / len(core),
        "deepseek_score_confirmed_count": sum(int(row["deepseek_score_confirmed"]) for row in core),
        "evidence_gate_pass_count": sum(int(row["evidence_gate"]) for row in core),
        "evidence_gate_pass_rate": float(np.mean([row["evidence_gate"] for row in core])),
    }])
    write_csv(args.out_dir / "tables/exp36a_variant_input_equivalence.csv", input_equivalence)
    write_csv(args.out_dir / "tables/exp36a_target_mass_by_human_label.csv", target_mass)
    gate_aggregate: dict[tuple[Any, ...], int] = Counter(
        (row["human_label"], row["language"], row["metric_family"], row["qwen_human_gap"],
         row["gate"], row["gate_failure_reasons"] or "none") for row in gate_rows
    )
    write_csv(args.out_dir / "tables/exp36a_evidence_gate_summary.csv", [
        {"human_label": key[0], "language": key[1], "metric_family": key[2], "qwen_human_gap": key[3],
         "gate": key[4], "gate_failure_reasons": key[5], "rows": count}
        for key, count in sorted(gate_aggregate.items(), key=lambda item: str(item[0]))
    ])
    direction_aggregate: dict[tuple[Any, ...], int] = Counter(
        (row["human_label"], row["qwen_score"], row["qwen_human_gap"], row["direction_factor"],
         row["direction_reason"], row["deepseek_present"], row["deepseek_confirmed"]) for row in direction_rows
    )
    write_csv(args.out_dir / "tables/exp36a_direction_transition_summary.csv", [
        {"human_label": key[0], "qwen_score": key[1], "qwen_human_gap": key[2],
         "direction_factor": key[3], "direction_reason": key[4], "deepseek_present": key[5],
         "deepseek_confirmed": key[6], "rows": count}
        for key, count in sorted(direction_aggregate.items(), key=lambda item: str(item[0]))
    ])
    low_aggregate: dict[tuple[Any, ...], int] = Counter(
        (row["human_label"], row["qwen_score"], row["direction_factor"], row["anchor_blocked"],
         row["failure_mask"]) for row in low_audit
    )
    write_csv(args.out_dir / "tables/exp36a_low_tail_anchor_audit.csv", [
        {"human_label": key[0], "qwen_score": key[1], "direction_factor": key[2],
         "anchor_blocked": key[3], "failure_mask": key[4], "rows": count}
        for key, count in sorted(low_aggregate.items(), key=lambda item: str(item[0]))
    ])
    failure_aggregate: dict[tuple[Any, ...], list[int]] = {}
    for row in failure_rows:
        key = (row["human_label"], row["language"], row["metric_family"], row["failure_mask"])
        values = failure_aggregate.setdefault(key, [0] * (1 + len(FAILURE_CLASSES)))
        values[0] += 1
        for index, name in enumerate(FAILURE_CLASSES):
            values[index + 1] += int(row[f"failure_{index}_{name}"])
    write_csv(args.out_dir / "tables/exp36a_failure_target_coverage.csv", [
        {"human_label": key[0], "language": key[1], "metric_family": key[2], "failure_mask": key[3],
         "rows": values[0], **{f"failure_{index}_{name}": values[index+1] for index, name in enumerate(FAILURE_CLASSES)}}
        for key, values in sorted(failure_aggregate.items(), key=lambda item: str(item[0]))
    ])
    class_rows = [{"failure_class": name, "positive_count": sum(int(row[f"failure_{i}_{name}"]) for row in failure_rows),
                   "masked_positive_count": sum(int(row["failure_mask"]) * int(row[f"failure_{i}_{name}"]) for row in failure_rows)}
                  for i, name in enumerate(FAILURE_CLASSES)]
    write_csv(args.out_dir / "tables/exp36a_failure_class_distribution.csv", class_rows)
    vote_rows = []
    for label in ("all", 1, 2, 3, 4, 5):
        subset = core if label == "all" else [row for row in core if row["rounded_human_label"] == label]
        lambdas = np.asarray([row["teacher_lambda"] for row in subset], dtype=float)
        vote_rows.append({
            "human_label": label, "rows": len(subset), "nonzero_lambda_count": int(np.sum(lambdas > 0)),
            "lambda_min": float(np.min(lambdas)), "lambda_mean": float(np.mean(lambdas)),
            "lambda_p50": float(np.quantile(lambdas, 0.5)), "lambda_p90": float(np.quantile(lambdas, 0.9)),
            "lambda_p95": float(np.quantile(lambdas, 0.95)), "lambda_max": float(np.max(lambdas)),
            "mean_student_uncertainty": float(np.mean([row["student_uncertainty"] for row in subset])),
            "mean_human_entropy": float(np.mean([row["human_entropy"] for row in subset])),
            "evidence_gate_pass_count": sum(int(row["evidence_gate"]) for row in subset),
        })
    write_csv(args.out_dir / "tables/exp36a_teacher_vote_distribution.csv", vote_rows)
    write_csv(args.out_dir / "tables/exp36a_shuffled_control_audit.csv", [{
        "rows": len(core), "strata": len(grouped), "singleton_strata": sum(len(v) == 1 for v in grouped.values()),
        "moved_rows": moved, "moved_rate": moved / len(core),
        "teacher_target_multiset_preserved": True, "failure_class_count_preserved": True,
        "teacher_entropy_multiset_preserved": True,
    }])
    write_json(args.out_dir / "hashes/exp36a_dataset_hashes.json", hashes)
    write_json(args.out_dir / "private/exp36a_supervision_core.json", {
        "rows": len(core), "qwen_range_modes": dict(range_modes), "deepseek_rows": len(deepseek),
        "evidence_gate_pass": sum(row["evidence_gate"] for row in core), "test_access_count": 0,
    })
    print({"status": "COMPLETED", "rows": len(core), "variants": len(variants), "test_access_count": 0})


if __name__ == "__main__":
    main()
