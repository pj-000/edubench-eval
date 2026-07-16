"""Lock 12 new train-only sources and their original metric-specific rubrics."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .common import ensure_output_layout, read_jsonl, sha256_path, stable_id, write_csv, write_json, write_jsonl
from .exp48b_common import OUT, PRIVATE, TRAIN, exhaustive_monotonicity_audit, parse_rubric_levels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--exp48a-packets", type=Path, default=Path(__file__).resolve().parent / "outputs/exp48a_qualification_pilot/private/source_packets/exp48a_generator_blueprints_60.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=4802)
    args = parser.parse_args()
    ensure_output_layout(args.out_dir)
    rows = read_jsonl(args.train)
    if len(rows) != 2654:
        raise ValueError(f"Locked paper-like train must contain 2654 rows, got {len(rows)}")
    excluded = set()
    if args.exp48a_packets.exists():
        excluded = {str(row["private_source_question_key"]) for row in read_jsonl(args.exp48a_packets)}
    representatives: dict[tuple[str, str], dict] = {}
    for row in rows:
        metric = str(row.get("metric_canonical", ""))
        question_key = str(row.get("question_key", ""))
        if not metric or not question_key or question_key in excluded:
            continue
        try:
            parse_rubric_levels(row.get("rubric"))
        except ValueError:
            continue
        representatives.setdefault((metric, question_key), row)
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for row in representatives.values():
        by_metric[str(row["metric_canonical"])].append(row)
    if len(by_metric) != 12 or any(not pool for pool in by_metric.values()):
        raise ValueError("Expected at least one unused source for each of 12 metrics")
    rng = random.Random(args.seed)
    selected = []
    used_subjects = Counter()
    used_scenarios = Counter()
    used_question_keys: set[str] = set()
    for metric in sorted(by_metric):
        pool = [row for row in by_metric[metric] if str(row["question_key"]) not in used_question_keys]
        if not pool:
            raise ValueError(f"Cannot select a globally unique source question key for {metric}")
        rng.shuffle(pool)
        row = max(
            pool,
            key=lambda item: (
                used_subjects[str(item.get("subject_canonical", ""))] == 0,
                used_scenarios[str(item.get("scenario_canonical", ""))] == 0,
                -used_subjects[str(item.get("subject_canonical", ""))],
                str(item["question_key"]),
            ),
        )
        selected.append(row)
        used_question_keys.add(str(row["question_key"]))
        used_subjects[str(row.get("subject_canonical", ""))] += 1
        used_scenarios[str(row.get("scenario_canonical", ""))] += 1
    packets = []
    for index, row in enumerate(selected, 1):
        levels = parse_rubric_levels(row["rubric"])
        packets.append({
            "blueprint_id": f"b48_{index:02d}",
            "metric": row["metric_canonical"],
            "language": row["language"],
            "subject": row.get("subject_canonical", ""),
            "scenario": row.get("scenario_canonical", ""),
            "education_level": row.get("education_level_canonical", ""),
            "source_question": row["question"],
            "metadata": row["metadata_raw"],
            "rubric_levels": {str(level): text for level, text in sorted(levels.items())},
            "private_source_question_key": row["question_key"],
        })
    packet_path = args.out_dir / "private/source_packets/exp48b_metric_rubric_blueprints_12.jsonl"
    write_jsonl(packet_path, packets)
    monotonicity = exhaustive_monotonicity_audit()
    if not monotonicity["pass"]:
        raise AssertionError("Exp48B v2 score program failed exhaustive monotonicity audit")
    write_json(args.out_dir / "configs/exp48b_score_program_audit.json", monotonicity)
    write_json(args.out_dir / "configs/exp48b_protocol_lock.json", {
        "experiment": "Exp48B metric-specific rubric-grounded local-edit pilot",
        "families": 12,
        "metrics": 12,
        "scores": [2, 3, 4],
        "source_overlap_with_exp48a": 0,
        "generation": "one base answer plus independent metric-specific score2 and score3 span replacements",
        "rubric_source": "original per-metric 1-5 benchmark rubric",
        "score_program_version": "eduq_tail_v2_metric_specific",
        "replacement_length_ratio": [0.8, 1.2],
        "answer_length_ratio_max": 1.2,
        "verifiers_required": 2,
        "cross_model_family_required": True,
        "dev_access_count": 0,
        "test_access_count": 0,
    })
    gates = {
        "generation_valid_families": 12,
        "outside_span_identity_families": 12,
        "verifier_completeness": 1.0,
        "evidence_substring_validity": 1.0,
        "criterion_agreement_min": 0.90,
        "decisive_agreement_min": 0.90,
        "ab_programmatic_exact_min": 0.90,
        "intended_exact_min_per_verifier_rows": 33,
        "qwk_min_per_verifier": 0.85,
        "fully_confirmed_families_min": 9,
        "score2_confirmed_by_both_min": 10,
        "score2_to_4_max": 0,
        "ordered_families_min_per_verifier": 10,
        "metrics_with_accepted_family_min": 9,
        "style_macro_f1_max": 0.45,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(args.out_dir / "configs/exp48b_qualification_gates.json", gates)
    write_csv(args.out_dir / "tables/exp48b_source_distribution.csv", [{
        "metric": row["metric"], "language": row["language"], "subject": row["subject"],
        "scenario": row["scenario"], "education_level": row["education_level"],
    } for row in packets])
    decision = {
        "status": "EXP48B_PREPARED",
        "selected_sources": 12,
        "unique_metrics": len({row["metric"] for row in packets}),
        "source_overlap_with_exp48a": len({row["private_source_question_key"] for row in packets} & excluded),
        "dev_access_count": 0,
        "test_access_count": 0,
        "source_signature": stable_id("exp48b", *sorted(row["private_source_question_key"] for row in packets)),
        "train_sha256": sha256_path(args.train),
        "private_packet_sha256": sha256_path(packet_path),
    }
    write_json(args.out_dir / "decision/exp48b_prepare_decision.json", decision)
    report = [
        "# Exp48B preparation report", "", "- Status: **EXP48B_PREPARED**",
        "- Locked sources: 12 new train-only question keys, one per canonical metric.",
        "- Exp48A source overlap: 0.",
        "- Every packet preserves its original metric-specific 1-5 rubric verbatim.",
        "- Shared code validates local edits only; it does not replace rubric semantics.",
        "- Score program exhaustive monotonicity audit: PASS.",
        "- Dev access: 0; test access: 0; no training; no GPU.",
    ]
    (args.out_dir / "reports/exp48b_prepare_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
