"""Build Exp27I Codex-calibrated teacher-audited train annotations.

This script materializes the calibration after direct inspection of the
Exp27I top-conflict queue. It is intentionally conservative: high-conflict
cases are not forced into high-weight training labels. They keep a calibrated
score when evidence is usable, but carry `review_only` unless the teacher/human
agreement is clean enough for SFT/DPO.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import clean  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        score = int(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def label_region(score: int | None) -> str:
    if score is None:
        return "unknown"
    if score <= 2:
        return "low"
    if score == 3:
        return "mid"
    return "high"


def load_stage(out_dir: Path, provider: str, stage: str) -> dict[str, dict[str, Any]]:
    path = out_dir / "annotations" / "parsed" / provider / f"exp27d_{stage}_outputs.jsonl"
    key = "blind" if stage == "blind" else "audit"
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        obj = parsed.get(key) if isinstance(parsed.get(key), dict) else {}
        if row.get("sample_id") and obj:
            out[str(row["sample_id"])] = obj
    return out


def provider_view(out_dir: Path, provider: str) -> dict[str, dict[str, Any]]:
    blind = load_stage(out_dir, provider, "blind")
    audit = load_stage(out_dir, provider, "audit")
    out: dict[str, dict[str, Any]] = {}
    for sid, b in blind.items():
        a = audit.get(sid, {})
        out[sid] = {
            "provider": provider,
            "score": as_int(b.get("teacher_score")),
            "reason": clean(b.get("teacher_reason")),
            "major_failures": b.get("major_failures") if isinstance(b.get("major_failures"), list) else [],
            "score_cap": as_int(b.get("score_cap")),
            "rubric_clause": b.get("rubric_clause"),
            "failure_visibility": b.get("failure_visibility", ""),
            "overestimation_risk": b.get("overestimation_risk", ""),
            "surface_plausibility": b.get("surface_plausibility", ""),
            "target_confusion_risk": b.get("target_confusion_risk", ""),
            "audit_label_quality": a.get("label_quality", ""),
            "audit_label_noise_type": a.get("label_noise_type", ""),
            "audit_recommended_training_use": a.get("recommended_training_use", ""),
            "audit_sample_weight_suggestion": a.get("sample_weight_suggestion"),
            "audit_hard_conflict": bool(a.get("hard_conflict")),
            "audit_target_confusion_detected": bool(a.get("target_confusion_detected")),
            "audit_reason": clean(a.get("audit_reason")),
        }
    return out


def teacher_reliability(view: dict[str, Any]) -> float:
    score = 1.0
    if view.get("target_confusion_risk") in {"possible", "high"}:
        score -= 1.5
    if view.get("audit_target_confusion_detected"):
        score -= 3.0
    quality = view.get("audit_label_quality")
    if quality == "reliable":
        score += 2.0
    elif quality == "plausible_adjacent":
        score += 0.75
    elif quality == "suspected_conflict":
        score -= 0.75
    elif quality == "unclear":
        score -= 1.0
    use = view.get("audit_recommended_training_use")
    if use == "high_weight":
        score += 1.0
    elif use == "low_weight":
        score += 0.25
    elif use == "review_only":
        score -= 0.75
    elif use == "exclude":
        score -= 2.0
    if view.get("audit_hard_conflict"):
        score -= 0.5
    return score


def select_reason(score: int, q: dict[str, Any], d: dict[str, Any]) -> tuple[str, list[str], int | None, str]:
    candidates = []
    for view in [q, d]:
        if view.get("score") == score and view.get("reason"):
            candidates.append((teacher_reliability(view), view))
    if candidates:
        _rel, view = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
        failures = view.get("major_failures") or (["no_major_failure"] if score >= 4 else ["unclear"])
        return view.get("reason", ""), failures, view.get("score_cap"), view.get("provider", "")
    failures = ["no_major_failure"] if score >= 4 else ["unclear"]
    return (
        "Codex calibration retained the score under unresolved teacher-human disagreement; use this row conservatively.",
        failures,
        None if score >= 4 else score,
        "codex_conservative",
    )


def median_score(scores: list[int]) -> int:
    return max(1, min(5, int(round(float(median(scores))))))


def choose_calibration(
    sid: str,
    original: int,
    q: dict[str, Any],
    d: dict[str, Any],
    top80: bool,
) -> dict[str, Any]:
    q_score = q.get("score")
    d_score = d.get("score")
    q_rel = teacher_reliability(q)
    d_rel = teacher_reliability(d)
    score_gap = abs(q_score - d_score) if q_score is not None and d_score is not None else 0
    target_issue = any(
        [
            q.get("target_confusion_risk") in {"possible", "high"},
            d.get("target_confusion_risk") in {"possible", "high"},
            q.get("audit_target_confusion_detected"),
            d.get("audit_target_confusion_detected"),
        ]
    )

    source = "agreement_based"
    note = "Low-conflict row calibrated by teacher/human agreement."
    confidence = "medium"
    training_use = "low_weight"
    sample_weight = 1.0

    if q_score is None and d_score is None:
        final_score = original
        source = "human_fallback_missing_teachers"
        note = "Both teacher scores missing; retain original human score and keep review-only."
        confidence = "low"
        training_use = "review_only"
        sample_weight = 0.0
    elif q_score == d_score and q_score is not None:
        final_score = q_score
        source = "dual_teacher_consensus"
        if abs(final_score - original) <= 1 and not target_issue:
            confidence = "high"
            training_use = "high_weight"
            sample_weight = 2.0
            note = "Both teachers agree and are within one point of the original human score."
        else:
            confidence = "medium"
            training_use = "review_only" if top80 else "low_weight"
            sample_weight = 0.5 if top80 else 1.0
            note = "Both teachers agree but conflict with the original human score or target-scope flags; keep conservative."
    elif q_score == original and d_score is not None and (q_rel >= d_rel or abs(d_score - original) >= 2):
        final_score = original
        source = "human_qwen_agreement"
        confidence = "high" if abs(d_score - original) <= 1 and not target_issue else "medium"
        training_use = "high_weight" if confidence == "high" else "low_weight"
        sample_weight = 2.0 if confidence == "high" else 1.0
        note = "Original human score agrees with Qwen; DeepSeek is de-weighted by conflict or lower reliability."
    elif d_score == original and q_score is not None and (d_rel >= q_rel or abs(q_score - original) >= 2):
        final_score = original
        source = "human_deepseek_agreement"
        confidence = "high" if abs(q_score - original) <= 1 and not target_issue else "medium"
        training_use = "high_weight" if confidence == "high" else "low_weight"
        sample_weight = 2.0 if confidence == "high" else 1.0
        note = "Original human score agrees with DeepSeek; Qwen is de-weighted by conflict or lower reliability."
    elif q_score is not None and d_score is not None and score_gap <= 1 and abs(median_score([q_score, d_score]) - original) <= 1:
        final_score = median_score([original, q_score, d_score])
        source = "adjacent_three_way_consensus"
        confidence = "medium"
        training_use = "low_weight"
        sample_weight = 1.0
        note = "Human and both teachers are within a small adjacent band; use median score with low weight."
    else:
        all_scores = [score for score in [original, q_score, d_score] if score is not None]
        final_score = median_score(all_scores)
        source = "codex_top80_conservative_median" if top80 else "conservative_median"
        confidence = "low" if target_issue or score_gap >= 2 else "medium"
        training_use = "review_only" if top80 or score_gap >= 2 or target_issue else "low_weight"
        sample_weight = 0.25 if training_use == "review_only" else 0.75
        note = (
            "High-conflict sample inspected in the Codex top-conflict pass; "
            "final score uses conservative median and should not be high-weight training data."
            if top80
            else "Teacher-human disagreement unresolved; use conservative median with low confidence."
        )

    reason, failures, score_cap, reason_source = select_reason(final_score, q, d)
    if training_use == "review_only":
        sample_weight = min(sample_weight, 0.25)
    if training_use == "high_weight":
        sample_weight = max(sample_weight, 2.0)

    return {
        "sample_id": sid,
        "original_human_score": original,
        "qwen_score": q_score,
        "deepseek_score": d_score,
        "calibrated_score": final_score,
        "calibrated_score_region": label_region(final_score),
        "calibrated_reason": reason,
        "calibrated_major_failures": failures,
        "calibrated_score_cap": score_cap,
        "reason_source": reason_source,
        "calibration_source": source,
        "codex_top80_direct_review": top80,
        "calibration_confidence": confidence,
        "recommended_training_use": training_use,
        "sample_weight": sample_weight,
        "calibration_note": note,
        "qwen_reliability": round(q_rel, 3),
        "deepseek_reliability": round(d_rel, 3),
        "target_issue_flag": target_issue,
        "score_gap": score_gap,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    packets = {
        str(row["sample_id"]): row
        for row in read_jsonl(out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl")
    }
    refs = {
        str(row["sample_id"]): row
        for row in read_jsonl(out_dir / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl")
    }
    qwen = provider_view(out_dir, "qwen")
    deepseek = provider_view(out_dir, "deepseek")
    queue = read_csv_rows(out_dir / "annotation" / "exp27i_human_qwen_deepseek_conflict_queue.csv")
    top80 = {str(row["sample_id"]) for row in queue[:80]}

    calibrated_rows: list[dict[str, Any]] = []
    top80_rows: list[dict[str, Any]] = []
    for sid, ref in refs.items():
        original = as_int(ref.get("original_score"))
        if original is None:
            continue
        q = qwen.get(sid, {"provider": "qwen", "score": None})
        d = deepseek.get(sid, {"provider": "deepseek", "score": None})
        cal = choose_calibration(sid, original, q, d, sid in top80)
        packet = packets.get(sid, {})
        teacher_input = packet.get("teacher_input", {}) if isinstance(packet.get("teacher_input"), dict) else {}
        source_meta = packet.get("source_meta", {}) if isinstance(packet.get("source_meta"), dict) else {}
        row = {
            **cal,
            "split": "train",
            "pilot_group": ref.get("pilot_group", ""),
            "question_key": source_meta.get("question_key", ""),
            "language": source_meta.get("language", ""),
            "subject": source_meta.get("subject", ""),
            "metric": source_meta.get("metric", ""),
            "question": teacher_input.get("question", ""),
            "answer": teacher_input.get("answer", ""),
            "rubric": teacher_input.get("rubric", ""),
            "metadata": teacher_input.get("metadata", ""),
            "qwen_reason": q.get("reason", ""),
            "deepseek_reason": d.get("reason", ""),
            "qwen_major_failures": q.get("major_failures", []),
            "deepseek_major_failures": d.get("major_failures", []),
            "qwen_audit_reason": q.get("audit_reason", ""),
            "deepseek_audit_reason": d.get("audit_reason", ""),
        }
        calibrated_rows.append(row)
        if sid in top80:
            top80_rows.append(
                {
                    "sample_id": sid,
                    "original_human_score": cal["original_human_score"],
                    "qwen_score": cal["qwen_score"],
                    "deepseek_score": cal["deepseek_score"],
                    "calibrated_score": cal["calibrated_score"],
                    "recommended_training_use": cal["recommended_training_use"],
                    "calibration_source": cal["calibration_source"],
                    "calibration_confidence": cal["calibration_confidence"],
                    "calibration_note": cal["calibration_note"],
                    "metric": source_meta.get("metric", ""),
                    "qwen_reason_preview": clean(q.get("reason", ""))[:220],
                    "deepseek_reason_preview": clean(d.get("reason", ""))[:220],
                }
            )

    data_dir = out_dir / "data"
    write_jsonl(data_dir / "exp27i_teacher_audited_361_calibrated_train.jsonl", calibrated_rows)
    write_csv(out_dir / "annotation" / "exp27i_codex_top80_direct_review.csv", top80_rows)

    review_manifest = [
        {
            "sample_id": row["sample_id"],
            "pilot_group": row["pilot_group"],
            "metric": row["metric"],
            "language": row["language"],
            "subject": row["subject"],
            "original_human_score": row["original_human_score"],
            "qwen_score": row["qwen_score"],
            "deepseek_score": row["deepseek_score"],
            "calibrated_score": row["calibrated_score"],
            "recommended_training_use": row["recommended_training_use"],
            "sample_weight": row["sample_weight"],
            "calibration_source": row["calibration_source"],
            "calibration_confidence": row["calibration_confidence"],
            "codex_top80_direct_review": row["codex_top80_direct_review"],
            "target_issue_flag": row["target_issue_flag"],
            "calibration_note": row["calibration_note"],
            "calibrated_reason_preview": clean(row["calibrated_reason"])[:260],
        }
        for row in calibrated_rows
    ]
    write_csv(out_dir / "review" / "exp27i_calibrated_361_review_manifest.csv", review_manifest)

    sft_rows = []
    for row in calibrated_rows:
        if row["recommended_training_use"] not in {"high_weight", "low_weight"}:
            continue
        assistant = {
            "reason": row["calibrated_reason"],
            "major_failures": row["calibrated_major_failures"],
            "score_cap": row["calibrated_score_cap"],
            "risk_flag": "hidden_low_failure" if row["calibrated_score"] <= 2 else "no_hidden_low_failure",
            "score": row["calibrated_score"],
        }
        user = "\n\n".join(
            [
                f"Question:\n{row['question']}",
                f"Answer:\n{row['answer']}",
                f"Metric:\n{row['metric']}",
                f"Rubric:\n{row['rubric']}",
                f"Metadata:\n{row['metadata']}",
            ]
        )
        sft_rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an educational assessment evaluator. Return only valid JSON.",
                    },
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, sort_keys=True)},
                ],
                "sample_id": row["sample_id"],
                "calibrated_score": row["calibrated_score"],
                "recommended_training_use": row["recommended_training_use"],
                "sample_weight": row["sample_weight"],
                "calibration_source": row["calibration_source"],
            }
        )
    write_jsonl(data_dir / "exp27i_teacher_audited_sft_train_high_low_weight.jsonl", sft_rows)

    label_counts = Counter(str(row["calibrated_score"]) for row in calibrated_rows)
    use_counts = Counter(str(row["recommended_training_use"]) for row in calibrated_rows)
    source_counts = Counter(str(row["calibration_source"]) for row in calibrated_rows)
    confidence_counts = Counter(str(row["calibration_confidence"]) for row in calibrated_rows)
    group_use_counts = Counter((row["pilot_group"], row["recommended_training_use"]) for row in calibrated_rows)

    write_csv(
        out_dir / "tables" / "exp27i_calibrated_label_distribution.csv",
        [{"calibrated_score": key, "count": value} for key, value in sorted(label_counts.items())],
    )
    write_csv(
        out_dir / "tables" / "exp27i_calibration_use_counts.csv",
        [{"recommended_training_use": key, "count": value} for key, value in sorted(use_counts.items())],
    )
    write_csv(
        out_dir / "tables" / "exp27i_calibration_source_counts.csv",
        [{"calibration_source": key, "count": value} for key, value in sorted(source_counts.items())],
    )
    write_csv(
        out_dir / "tables" / "exp27i_calibration_confidence_counts.csv",
        [{"calibration_confidence": key, "count": value} for key, value in sorted(confidence_counts.items())],
    )
    write_csv(
        out_dir / "tables" / "exp27i_group_use_counts.csv",
        [
            {"pilot_group": key[0], "recommended_training_use": key[1], "count": value}
            for key, value in sorted(group_use_counts.items())
        ],
    )

    high_weight = use_counts.get("high_weight", 0)
    low_weight = use_counts.get("low_weight", 0)
    review_only = use_counts.get("review_only", 0)
    train_ready = high_weight + low_weight
    decision = {
        "calibrated_rows": len(calibrated_rows),
        "codex_top80_direct_review_rows": len(top80_rows),
        "high_weight_rows": high_weight,
        "low_weight_rows": low_weight,
        "review_only_rows": review_only,
        "train_ready_rows_high_or_low_weight": train_ready,
        "sft_ready_rows": len(sft_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "use_counts": dict(sorted(use_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "recommendation": "use_high_and_low_weight_rows_for_data_quality_experiments_keep_review_only_out_of_training",
        "test_label_read": False,
        "raw_api_outputs_committed": False,
    }
    write_json(out_dir / "decision" / "exp27i_codex_calibration_decision.json", decision)

    report = [
        "# Exp27I Codex-Calibrated Teacher-Audited Data",
        "",
        "This step builds the final train-only calibrated 361-row annotation set from Exp27I Qwen/DeepSeek outputs.",
        "",
        "## What Codex Reviewed",
        "",
        "- The top 80 conflict cases were inspected in batches using the evaluator output, metric, Qwen reason, DeepSeek reason, original human score, and audit flags.",
        "- The recurring conflict modes were: answer-key/rubric ambiguity, scenario-integration rubric mismatch, evaluator-output internal contradictions, and high-score-protection disagreements.",
        "- High-conflict rows are intentionally not forced into high-weight gold labels. They are marked `review_only` unless there is clean teacher/human agreement.",
        "",
        "## Outputs",
        "",
        "- `data/exp27i_teacher_audited_361_calibrated_train.jsonl`: final calibrated train-only annotations.",
        "- `data/exp27i_teacher_audited_sft_train_high_low_weight.jsonl`: SFT-ready subset using only high/low-weight calibrated rows.",
        "- `annotation/exp27i_codex_top80_direct_review.csv`: top-conflict review decisions.",
        "- `review/exp27i_calibrated_361_review_manifest.csv`: compact all-row review manifest.",
        "- `tables/exp27i_calibration_use_counts.csv`: train-use counts.",
        "",
        "## Counts",
        "",
        f"- calibrated_rows: {len(calibrated_rows)}",
        f"- top80_direct_review_rows: {len(top80_rows)}",
        f"- high_weight_rows: {high_weight}",
        f"- low_weight_rows: {low_weight}",
        f"- review_only_rows: {review_only}",
        f"- train_ready_rows_high_or_low_weight: {train_ready}",
        f"- sft_ready_rows: {len(sft_rows)}",
        f"- calibrated_label_counts: `{dict(sorted(label_counts.items()))}`",
        "",
        "## Training Recommendation",
        "",
        "Use only `high_weight` and `low_weight` rows for the first teacher-audited SFT/DPO data-quality experiment. Keep `review_only` rows for human/GPT adjudication or qualitative analysis.",
        "",
        "This step does not fabricate DPO rejected responses. For DPO, use this calibrated dataset as the corrected/chosen source and build rejected responses from real model mistakes in a separate pair-construction step.",
        "",
        "## Guardrails",
        "",
        "- Only train split samples are included.",
        "- Dev/test were used only as leakage guards in packet preparation.",
        "- No test label was read.",
        "- Raw API responses are not part of this calibrated output.",
    ]
    write_text(out_dir / "reports" / "exp27i_codex_calibrated_dataset_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Exp27I Codex-calibrated train annotations.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
