"""Construct Exp48B answers mechanically and validate local-edit invariants."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import char_ngrams, jaccard, normalize_text, read_jsonl, tokens, write_csv, write_json, write_jsonl
from .exp48b_common import OUT, PRIVATE, TRAIN, construct_answers, validate_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, default=PRIVATE / "generated_plans/exp48b_metric_specific_edit_plans.jsonl")
    parser.add_argument("--packets", type=Path, default=PRIVATE / "source_packets/exp48b_metric_rubric_blueprints_12.jsonl")
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    plans = read_jsonl(args.plans)
    packets = {row["metric"]: row for row in read_jsonl(args.packets)}
    train_questions = [str(row["question"]) for row in read_jsonl(args.train)]
    train_features = [(normalize_text(text), char_ngrams(text), tokens(text)) for text in train_questions]
    validation_rows, novelty_rows, families = [], [], []
    family_ids, question_keys = set(), set()
    for plan in plans:
        errors = validate_plan(plan)
        source = packets.get(plan.get("metric"))
        if source is None:
            errors.append("metric_not_in_locked_source_packets")
        else:
            if plan.get("rubric_levels") != source["rubric_levels"]:
                errors.append("rubric_levels_not_verbatim_locked_source")
            if plan.get("language") != source["language"]:
                errors.append("language_not_locked_source_language")
            plan["source_blueprint_id"] = source["blueprint_id"]
        family_id = str(plan.get("family_id", ""))
        question_key = str(plan.get("synthetic_question_key", ""))
        if family_id in family_ids:
            errors.append("duplicate_family_id")
        if question_key in question_keys:
            errors.append("duplicate_synthetic_question_key")
        family_ids.add(family_id)
        question_keys.add(question_key)
        question = str(plan.get("synthetic_question", ""))
        normalized, chars, toks = normalize_text(question), char_ngrams(question), tokens(question)
        exact = any(normalized == row[0] for row in train_features)
        max_char = max((jaccard(chars, row[1]) for row in train_features), default=0.0)
        max_token = max((jaccard(toks, row[2]) for row in train_features), default=0.0)
        novelty_pass = not exact and max_char < 0.80 and max_token < 0.80
        if not novelty_pass:
            errors.append("question_novelty_failed")
        novelty_rows.append({"family_id": family_id, "metric": plan.get("metric"), "exact_train_match": int(exact), "max_char5_jaccard": max_char, "max_token_jaccard": max_token, "pass": novelty_pass})
        answers = construct_answers(plan) if not any("source_span" in error for error in errors) else []
        outside_identity = len(answers) == 3
        if outside_identity:
            base = str(plan["base_answer"])
            for score in (2, 3):
                edit = plan[f"score{score}_edit"]
                reconstructed = base.replace(str(edit["source_span"]), str(edit["replacement_span"]), 1)
                actual = next(row["text"] for row in answers if row["intended_score"] == score)
                outside_identity = outside_identity and reconstructed == actual
        if not outside_identity:
            errors.append("outside_span_identity_failed")
        valid = not errors
        validation_rows.append({"family_id": family_id, "metric": plan.get("metric"), "language": plan.get("language"), "valid": valid, "outside_span_identity": outside_identity, "errors": "|".join(sorted(set(errors)))})
        if valid:
            family = dict(plan)
            family["answers"] = answers
            families.append(family)
    write_jsonl(args.out_dir / "private/generated_families/exp48b_constructed_families.jsonl", families)
    write_csv(args.out_dir / "tables/exp48b_generation_validation.csv", validation_rows)
    write_csv(args.out_dir / "tables/exp48b_question_novelty.csv", novelty_rows)
    metrics = {row.get("metric") for row in families}
    decision = {
        "status": "EXP48B_GENERATION_VALID" if len(families) == 12 and len(metrics) == 12 else "EXP48B_GENERATION_INVALID",
        "plans": len(plans), "valid_families": len(families), "unique_metrics": len(metrics),
        "language_distribution": dict(sorted(Counter(str(row.get("language", "unknown")) for row in families).items())),
        "outside_span_identity_families": sum(row["outside_span_identity"] for row in validation_rows),
        "novelty_pass_families": sum(bool(row["pass"]) for row in novelty_rows),
        "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp48b_generation_decision.json", decision)
    report = ["# Exp48B generation validation", "", f"- Status: **{decision['status']}**", f"- Plans / valid families: {len(plans)} / {len(families)}", f"- Unique metrics: {len(metrics)}", f"- Outside-span identity: {decision['outside_span_identity_families']}/12", f"- Question novelty: {decision['novelty_pass_families']}/12", "- Final answers were mechanically constructed from one base answer; the generator did not freely write three answers.", "- Dev/test access: 0/0."]
    (args.out_dir / "reports/exp48b_generation_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    if decision["status"] != "EXP48B_GENERATION_VALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
