"""Build the preregistered Exp47A diagnosis and aggregate reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp47_label2_identifiability.common import ROOT, ensure_dirs, read_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def fmt(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    human = {row["group"]: row for row in read_csv(args.out_dir / "tables/exp47_label2_stable_vs_ambiguous.csv")}
    concentration = {row["label2_subset"]: row for row in read_csv(args.out_dir / "tables/exp47_label2_question_concentration.csv")}
    subtype = {
        (row["model"], row["role"], row["label2_subtype"]): row
        for row in read_csv(args.out_dir / "tables/exp47_label2_subtype_metrics.csv")
    }
    ranks = {
        (row["model"], row["role"], row["label2_subtype"]): row
        for row in read_csv(args.out_dir / "tables/exp47_label2_logit_rank_summary.csv")
    }
    metrics = {
        (row["model"], row["role"], row["fold"]): row
        for row in read_csv(args.out_dir / "tables/exp47_train_vs_heldout_metrics.csv")
    }

    total = int(human["all_hard_label2"]["count"])
    stable = int(human["stable_label2_total"]["count"])
    strict = int(human["strict_label2"]["count"])
    ambiguous = int(human["ambiguous_label2"]["count"])
    ambiguous_rate = ambiguous / total
    stable_rate = stable / total
    median_range = as_float(next(row["score_range_median"] for row in read_csv(args.out_dir / "tables/exp47_label2_entropy_summary.csv") if row["label2_subtype"] == "all_hard_label2"))

    teacher_train = subtype[("M1_4B_teacher", "outer_train", "stable_label2")]
    teacher_heldout = subtype[("M1_4B_teacher", "heldout", "stable_label2")]
    baseline_heldout = subtype[("M0_0.6B_E4", "heldout", "stable_label2")]
    teacher_train_recall = as_float(teacher_train.get("label2_recall"))
    teacher_heldout_recall = as_float(teacher_heldout.get("label2_recall"))
    baseline_heldout_recall = as_float(baseline_heldout.get("label2_recall"))

    ambiguity_flag = stable_rate < 0.50 or ambiguous_rate >= 0.50 or (median_range is not None and median_range >= 2)
    generalization_flag = teacher_train_recall is not None and teacher_train_recall >= 0.40 and teacher_heldout_recall is not None and teacher_heldout_recall <= 0.05
    adaptation_4b_flag = teacher_train_recall is not None and teacher_train_recall < 0.20
    baseline_outer_train_available = subtype[("M0_0.6B_E4", "outer_train", "stable_label2")]["availability"] == "AVAILABLE"
    adaptation_both_flag = adaptation_4b_flag and baseline_outer_train_available and as_float(subtype[("M0_0.6B_E4", "outer_train", "stable_label2")].get("label2_recall")) < 0.20

    if ambiguity_flag:
        category = "LABEL2_TARGET_AMBIGUOUS"
    elif generalization_flag:
        category = "QUESTION_GENERALIZATION_LIMIT"
    elif adaptation_both_flag:
        category = "OBJECTIVE_OR_ADAPTATION_LIMIT"
    else:
        category = "MIXED_DATA_AND_ADAPTATION_LIMIT"
    flags = {
        "label2_target_ambiguous": ambiguity_flag,
        "question_generalization_limit": generalization_flag,
        "4b_outer_train_adaptation_limit": adaptation_4b_flag,
        "objective_or_adaptation_limit_both_models": adaptation_both_flag,
        "0.6b_outer_train_unavailable": not baseline_outer_train_available,
    }
    recommend_new_data = category in {"QUESTION_GENERALIZATION_LIMIT", "MIXED_DATA_AND_ADAPTATION_LIMIT"}
    recommend_selective = category == "LABEL2_TARGET_AMBIGUOUS"
    decision = {
        "status": "EXP47A_AUDIT_COMPLETE",
        "primary_diagnosis": category,
        "secondary_diagnosis_flags": flags,
        "counts": {"hard_label2": total, "stable_label2": stable, "strict_label2": strict, "ambiguous_label2": ambiguous},
        "rates": {"stable_label2": stable_rate, "ambiguous_label2": ambiguous_rate, "median_human_score_range": median_range},
        "recall": {
            "0.6b_outer_train_stable_label2": None,
            "0.6b_heldout_stable_label2": baseline_heldout_recall,
            "4b_outer_train_stable_label2": teacher_train_recall,
            "4b_heldout_stable_label2": teacher_heldout_recall,
        },
        "recommend_new_human_low_tail_data": recommend_new_data,
        "recommend_selective_or_set_valued_scoring": recommend_selective,
        "recommend_student_kd": False,
        "recommend_test": False,
        "no_new_training_authorized": True,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp47_label2_identifiability_decision.json", decision)

    hard_concentration = concentration["hard_label2"]
    teacher_rank_train = ranks[("M1_4B_teacher", "outer_train", "stable_label2")]
    teacher_rank_heldout = ranks[("M1_4B_teacher", "heldout", "stable_label2")]
    base_overall = metrics[("M0_0.6B_E4", "heldout", "pooled")]
    teacher_overall = metrics[("M1_4B_teacher", "heldout", "pooled")]
    report = [
        "# Exp47A Label-2 Identifiability and Generalization Audit",
        "",
        f"Primary diagnosis: **{category}**",
        "",
        "## Human label structure",
        "",
        f"- Hard label-2 rows: {total}.",
        f"- Stable label-2 rows: {stable} ({stable_rate:.2%}); strict [2,2,2]: {strict}.",
        f"- Ambiguous label-2 rows: {ambiguous} ({ambiguous_rate:.2%}).",
        f"- Median human score range: {fmt(median_range)}.",
        "",
        "## Concentration",
        "",
        f"- Unique question keys: {hard_concentration['unique_question_keys']}.",
        f"- Effective question keys: {float(hard_concentration['effective_question_keys']):.2f}.",
        f"- Maximum single-question share: {float(hard_concentration['max_question_key_rate']):.2%}.",
        "",
        "## Train versus unseen-question behavior",
        "",
        f"- 4B stable-label2 outer-train recall: {fmt(teacher_train_recall)}.",
        f"- 4B stable-label2 heldout recall: {fmt(teacher_heldout_recall)}.",
        f"- 4B correctly predicts label 2 on {teacher_train['label2_correct']}/{teacher_train['n']} outer-train fold-sample predictions, versus {teacher_heldout['label2_correct']}/{teacher_heldout['n']} OOF heldout rows.",
        f"- 0.6B stable-label2 heldout recall: {fmt(baseline_heldout_recall)}.",
        "- 0.6B outer-train recall is unavailable because its fold checkpoints were removed; it was not recomputed or substituted.",
        f"- 4B class-2 top-2 rate on stable train rows: {float(teacher_rank_train['class2_top2_rate']):.4f}; heldout: {float(teacher_rank_heldout['class2_top2_rate']):.4f}.",
        f"- 4B mean class-2 probability on stable train rows: {float(teacher_rank_train['p2_mean']):.4f}; heldout: {float(teacher_rank_heldout['p2_mean']):.4f}.",
        f"- On heldout label-2 rows, 4B predicts class 4 for {teacher_heldout['pred_count_4']} and class 5 for {teacher_heldout['pred_count_5']} cases.",
        "",
        "## Existing OOF overall metrics",
        "",
        f"- 0.6B heldout MAE/QWK/Exact: {float(base_overall['MAE']):.4f} / {float(base_overall['QWK']):.4f} / {float(base_overall['Exact_Match']):.4f}.",
        f"- 4B heldout MAE/QWK/Exact: {float(teacher_overall['MAE']):.4f} / {float(teacher_overall['QWK']):.4f} / {float(teacher_overall['Exact_Match']):.4f}.",
        "",
        "## Decision",
        "",
        f"- New independent low-tail human data required: **{recommend_new_data}**.",
        f"- Selective/set-valued scoring recommended: **{recommend_selective}**.",
        "- Student KD: **false**.",
        "- Test access: **false**.",
        "- No new training is authorized by this audit.",
        "",
        "## Integrity",
        "",
        "No model was trained, no API was called, dev/test were not opened, and no row-level prediction, logit, sample-ID, checkpoint, or log artifact is public.",
    ]
    report_path = args.out_dir / "reports/exp47_label2_identifiability_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    causal = [
        "# Exp47A Optimization-versus-Data Diagnosis",
        "",
        f"Primary diagnosis: **{category}**",
        "",
        "The audit separates human-label ambiguity, question-key generalization, and train-side adaptation. The 0.6B outer-train result is deliberately marked unavailable rather than reconstructed after checkpoint deletion.",
        "",
        "Secondary flags:",
        "",
        *[f"- {name}: **{value}**" for name, value in flags.items()],
        "",
        "This diagnosis does not establish that full-fine-tuned 4B models are incapable. It only characterizes the existing locked 4B LoRA runs and existing 0.6B OOF runs.",
        "",
        "No post-hoc hyperparameter search, student distillation, or test evaluation is authorized.",
    ]
    causal_path = args.out_dir / "reports/exp47_optimization_vs_data_diagnosis.md"
    causal_path.write_text("\n".join(causal) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DECISION_BUILT", "primary_diagnosis": category, "recommend_student_kd": False, "recommend_test": False, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
