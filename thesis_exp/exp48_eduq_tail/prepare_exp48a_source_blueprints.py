"""Select 60 train-only source blueprints without consulting outcomes."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .common import MODULE, OUT, SEED, TRAIN, counts, ensure_output_layout, read_jsonl, sha256_path, stable_id, write_csv, write_json, write_jsonl


def complete(row: dict) -> bool:
    return all(str(row.get(key, "")).strip() for key in ("question_key", "question", "metric_canonical", "rubric", "metadata_raw"))


def choose(rows: list[dict], seed: int) -> list[dict]:
    representatives: dict[tuple[str, str], dict] = {}
    for row in rows:
        if complete(row):
            representatives.setdefault((str(row["metric_canonical"]), str(row["question_key"])), row)
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for row in representatives.values():
        by_metric[str(row["metric_canonical"])].append(row)
    if len(by_metric) != 12 or any(len(group) < 5 for group in by_metric.values()):
        raise ValueError(f"Expected 12 metrics with at least five complete question keys, got {dict((k,len(v)) for k,v in by_metric.items())}")

    rng = random.Random(seed)
    selected: list[dict] = []
    global_seen = {"subject": Counter(), "scenario": Counter(), "education": Counter(), "language": Counter()}
    used_question_keys: set[str] = set()
    for metric in sorted(by_metric, key=lambda value: (len(by_metric[value]), value)):
        pool = list(by_metric[metric])
        rng.shuffle(pool)
        local: list[dict] = []
        while len(local) < 5:
            def score(row: dict) -> tuple:
                subject = str(row.get("subject_canonical", ""))
                scenario = str(row.get("scenario_canonical", ""))
                education = str(row.get("education_level_canonical", ""))
                language = str(row.get("language", ""))
                novelty = (
                    4 * (global_seen["subject"][subject] == 0)
                    + 3 * (global_seen["scenario"][scenario] == 0)
                    + 2 * (global_seen["education"][education] == 0)
                    + 2 * (global_seen["language"][language] < len(selected) / 2)
                )
                scarcity = -(global_seen["subject"][subject] + global_seen["scenario"][scenario] + global_seen["education"][education])
                return novelty, scarcity, str(row["question_key"])
            available = [row for row in pool if row not in local and str(row["question_key"]) not in used_question_keys]
            if not available:
                raise ValueError(f"Cannot allocate five unique question keys for metric {metric}")
            best = max(available, key=score)
            local.append(best)
            used_question_keys.add(str(best["question_key"]))
            global_seen["subject"][str(best.get("subject_canonical", ""))] += 1
            global_seen["scenario"][str(best.get("scenario_canonical", ""))] += 1
            global_seen["education"][str(best.get("education_level_canonical", ""))] += 1
            global_seen["language"][str(best.get("language", ""))] += 1
        selected.extend(local)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.train = args.train.resolve()
    args.out_dir = args.out_dir.resolve()
    ensure_output_layout(args.out_dir)
    rows = read_jsonl(args.train)
    if len(rows) != 2654:
        raise ValueError(f"Locked paper-like train must have 2654 rows, got {len(rows)}")
    selected = choose(rows, args.seed)
    if len(selected) != 60 or len({row["question_key"] for row in selected}) != 60:
        raise AssertionError("Source lock must contain exactly 60 unique question keys")

    packets = []
    for index, row in enumerate(selected, 1):
        packets.append({
            "blueprint_id": f"bp_{index:03d}",
            "metric": row["metric_canonical"],
            "language": row["language"],
            "subject": row.get("subject_canonical", ""),
            "scenario": row.get("scenario_canonical", ""),
            "education_level": row.get("education_level_canonical", ""),
            "question": row["question"],
            "rubric": row["rubric"],
            "metadata": row["metadata_raw"],
            "private_source_question_key": row["question_key"],
        })
    packet_path = args.out_dir / "private/source_packets/exp48a_generator_blueprints_60.jsonl"
    write_jsonl(packet_path, packets)

    distribution = []
    for dimension, field in (("metric", "metric"), ("language", "language"), ("subject", "subject"), ("scenario", "scenario"), ("education_level", "education_level")):
        for value, count in sorted(Counter(str(row[field]) for row in packets).items()):
            distribution.append({"dimension": dimension, "value": value, "selected_question_keys": count})
    write_csv(args.out_dir / "tables/exp48a_source_distribution.csv", distribution)

    protocol = {
        "experiment": "Exp48A EduQ-TAIL Qualification Pilot", "seed": args.seed,
        "train_rows_read": len(rows), "selected_question_keys": 60,
        "target_per_metric": 5, "selection_uses_outcomes": False,
        "dev_access_count": 0, "test_access_count": 0,
        "private_generator_packet": str(packet_path.relative_to(MODULE)),
    }
    write_json(args.out_dir / "configs/exp48a_source_lock.json", protocol)
    write_json(args.out_dir / "configs/exp48a_generation_protocol_lock.json", {
        "families": 60, "answers_per_family": 3, "scores": [2, 3, 5], "criteria_min": 4,
        "criteria_max": 6, "score_program_version": "eduq_tail_v1", "max_answer_length_ratio": 1.5,
        "generator_packet_must_not_include_labels": True,
    })
    write_json(args.out_dir / "configs/exp48a_verification_protocol_lock.json", {
        "verifiers_required": 2, "verifier_direct_score_forbidden": True,
        "packet_excludes_intended_scores": True, "cross_family_required_for_scale_go": True,
    })
    gates = {
        "generated_families": 60, "valid_generated_families_min": 54,
        "accepted_families_min": 45, "accepted_per_score_min": 45,
        "criterion_agreement_min": 0.80, "exact_score_agreement_min": 0.85,
        "within_one_min": 0.98, "qwk_min": 0.75, "score2_to_high_max": 0,
        "question_char5_jaccard_max_exclusive": 0.80, "question_token_jaccard_max_exclusive": 0.80,
        "style_macro_f1_max": 0.45, "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "configs/exp48a_qualification_gates.json", gates)
    write_json(args.out_dir / "decision/exp48a_prepare_decision.json", {"status": "PREPARED", **protocol})
    write_json(args.out_dir / "hashes/exp48a_source_hashes.json", {
        "train_sha256": sha256_path(args.train), "private_packet_sha256": sha256_path(packet_path),
        "selected_question_key_signature": stable_id("source", *sorted(row["private_source_question_key"] for row in packets)),
    })
    prompt_schema_paths = sorted((MODULE / "prompts").glob("*")) + sorted((MODULE / "schemas").glob("*"))
    for path in prompt_schema_paths:
        if path.is_file():
            destination = args.out_dir / path.parent.name / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    write_json(args.out_dir / "hashes/exp48a_prompt_schema_hashes.json", {
        str(path.relative_to(MODULE)): sha256_path(path) for path in prompt_schema_paths if path.is_file()
    })
    report = [
        "# Exp48A preparation report", "", "- Status: **PREPARED**", f"- Locked train rows read: {len(rows)}",
        "- Selected source question keys: 60", "- Canonical metrics: 12 (5 keys each)",
        f"- Language distribution: `{json.dumps(counts(row['language'] for row in packets), ensure_ascii=False)}`",
        f"- Subject/scenario/education coverage: {len(set(row['subject'] for row in packets))}/{len(set(row['scenario'] for row in packets))}/{len(set(row['education_level'] for row in packets))}",
        "- Selection used no human labels, model predictions, or low-tail outcomes.",
        "- Dev access: 0; test access: 0.", "- Generator packet is private and gitignored.",
    ]
    (args.out_dir / "reports/exp48a_prepare_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PREPARED", "selected": 60, "packet": str(packet_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
