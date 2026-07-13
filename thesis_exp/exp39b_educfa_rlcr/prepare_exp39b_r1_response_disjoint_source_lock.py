"""Prepare the seed-44 response-disjoint Exp39B-R1 source lock before API use."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import (  # noqa: E402
    BAND_COUNTS, BANDS, EXP39A_LOCK, EXP39A_PACKETS, PROMPT_DIR, R1_ROOT, ROOT,
    SCHEMA_DIR, TRAIN_PATH, character_ngrams, ensure_layout, jaccard,
    normalize_answer, read_json, read_jsonl, sample_id, score_value, sha256_file,
    stable_hash, text_tokens, token_count, write_csv, write_json, write_jsonl,
)
from thesis_exp.exp39b_educfa_rlcr.prepare_exp39b_fresh_source_lock import assign_bands  # noqa: E402

NEAR_DUPLICATE_THRESHOLD = 0.90
BASE_COMMIT = "b4fcc26"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--exp39a-lock", type=Path, default=EXP39A_LOCK)
    parser.add_argument("--exp39a-packets", type=Path, default=EXP39A_PACKETS)
    parser.add_argument("--out-dir", type=Path, default=R1_ROOT)
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def eligible(row: dict[str, Any]) -> bool:
    human = [score_value(row, key) for key in ("human_1", "human_2", "human_3")]
    return (
        sum(value >= 4 for value in human) >= 2
        and all(value > 2 for value in human)
        and int(row["label_5"]) in {4, 5}
        and bool(str(row.get("answer") or "").strip())
        and bool(str(row.get("question_key") or "").strip())
        and bool(row.get("rubric"))
        and token_count(str(row.get("answer") or "")) >= 20
    )


def generator(row: dict[str, Any]) -> str:
    return str(row.get("generator_model") or row.get("answer_model") or "unknown")


def metric(row: dict[str, Any]) -> str:
    return str(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_id") or "unknown")


def answer_hash(row: dict[str, Any]) -> str:
    return stable_hash(normalize_answer(str(row.get("answer") or row.get("original_answer") or "")))


def assessment_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["question_key"]),
        normalize_answer(str(row.get("answer") or row.get("original_answer") or "")),
        metric(row),
        generator(row),
    )


def novelty(row: dict[str, Any], exp39a_same_qkey: list[dict[str, Any]]) -> tuple[float, float]:
    candidate_grams = character_ngrams(str(row["answer"]))
    candidate_tokens = text_tokens(str(row["answer"]))
    similarities = [
        (
            jaccard(candidate_grams, character_ngrams(str(other["answer"]))),
            jaccard(candidate_tokens, text_tokens(str(other["answer"]))),
        )
        for other in exp39a_same_qkey
    ]
    return (
        max((value[0] for value in similarities), default=0.0),
        max((value[1] for value in similarities), default=0.0),
    )


def dimensions(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("language") or "unknown"),
        str(row.get("metric_group") or "unknown"),
        str(row["label_5"]),
        generator(row),
    )


def select_balanced(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    available = [Counter() for _ in range(4)]
    selected_counts = [Counter() for _ in range(4)]
    for row in rows:
        for index, value in enumerate(dimensions(row)):
            available[index][value] += 1
    selected: list[dict[str, Any]] = []
    used_qkeys: set[str] = set()
    remaining = {sample_id(row): row for row in rows}
    while len(selected) < count:
        candidates = [row for row in remaining.values() if str(row["question_key"]) not in used_qkeys]
        if not candidates:
            raise RuntimeError("Could not select one response from each of 60 question keys")

        def rank(row: dict[str, Any]) -> tuple[float, float, str]:
            ratios = [
                selected_counts[index][value] / max(available[index][value], 1)
                for index, value in enumerate(dimensions(row))
            ]
            return sum(ratios), max(ratios), stable_hash({"seed": seed, "sample_id": sample_id(row)})

        chosen = min(candidates, key=rank)
        selected.append(chosen)
        used_qkeys.add(str(chosen["question_key"]))
        for index, value in enumerate(dimensions(chosen)):
            selected_counts[index][value] += 1
        del remaining[sample_id(chosen)]
    return selected


def packet(row: dict[str, Any], band_name: str, seed: int) -> dict[str, Any]:
    source_id = sample_id(row)
    return {
        "sample_id": "exp39b_r1_" + stable_hash({"source": source_id, "band": band_name, "seed": seed})[:21],
        "source_sample_id": source_id,
        "question_key": str(row["question_key"]),
        "question": str(row.get("question") or ""),
        "original_answer": str(row["answer"]),
        "metric": metric(row),
        "metric_group": str(row.get("metric_group") or "unknown"),
        "rubric": row["rubric"],
        "metadata": {
            "language": row.get("language"),
            "subject": row.get("subject_canonical") or row.get("subject_raw"),
            "education_level": row.get("education_level_canonical") or row.get("education_level_raw"),
            "scenario": row.get("scenario_canonical") or row.get("scenario_raw"),
            "metric_group": row.get("metric_group"),
            "generator_model": generator(row),
        },
        "language": str(row.get("language") or "unknown"),
        "subject": str(row.get("subject_canonical") or row.get("subject_raw") or "unknown"),
        "generator_model": generator(row),
        "source_label_5": int(row["label_5"]),
        "target_band_name": band_name,
        "target_band": BANDS[band_name],
    }


def freeze(path: Path, value: dict[str, Any]) -> None:
    if path.exists() and read_json(path) != value:
        raise RuntimeError(f"Frozen Exp39B-R1 artifact changed: {path}")
    write_json(path, value)


def verify_frozen_materials() -> tuple[dict[str, Any], str]:
    frozen_hashes = read_json(ROOT / "hashes/exp39b_prompt_schema_hashes.json")
    current = {
        "prompts": {path.name: sha256_file(path) for path in sorted(PROMPT_DIR.glob("exp39b_*.md"))},
        "schemas": {path.name: sha256_file(path) for path in sorted(SCHEMA_DIR.glob("exp39b_*.json"))},
    }
    if current != frozen_hashes:
        raise RuntimeError("Exp39B-R1 frozen prompt/schema hashes differ from b4fcc26")
    base_gate_hash = sha256_file(ROOT / "configs/exp39b_acceptance_gate.json")
    return current, base_gate_hash


def main() -> None:
    args = parse_args()
    ensure_layout(args.out_dir)
    (args.out_dir / "protocols").mkdir(parents=True, exist_ok=True)
    frozen_hashes, base_gate_hash = verify_frozen_materials()
    train = read_jsonl(args.train_jsonl)
    exp39a_packets = read_jsonl(args.exp39a_packets)
    exp39a_lock = read_json(args.exp39a_lock)
    if len(train) != 2654 or len({sample_id(row) for row in train}) != 2654:
        raise ValueError("Paper-like train must contain 2654 unique rows")
    if int(exp39a_lock.get("source_rows", -1)) != len(exp39a_packets):
        raise ValueError("Exp39A source lock and packets disagree")
    train_by_id = {sample_id(row): row for row in train}
    exp39a_ids = {str(row["source_sample_id"]) for row in exp39a_packets}
    missing = exp39a_ids - set(train_by_id)
    if missing:
        raise ValueError(f"Exp39A sources missing from train: {len(missing)}")
    exp39a_rows = [train_by_id[source_id] for source_id in sorted(exp39a_ids)]
    exp39a_qkeys = {str(row["question_key"]) for row in exp39a_rows}
    exp39a_answer_hashes = {answer_hash(row) for row in exp39a_rows}
    exp39a_assessment_keys = {assessment_key(row) for row in exp39a_rows}
    exp39a_by_qkey: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exp39a_rows:
        exp39a_by_qkey[str(row["question_key"])].append(row)

    stage_eligible = [row for row in train if eligible(row)]
    stage_id = [row for row in stage_eligible if sample_id(row) not in exp39a_ids]
    stage_hash = [row for row in stage_id if answer_hash(row) not in exp39a_answer_hashes]
    stage_assessment = [row for row in stage_hash if assessment_key(row) not in exp39a_assessment_keys]
    novelty_rows = []
    stage_novel = []
    for row in stage_assessment:
        char_similarity, token_similarity = novelty(row, exp39a_by_qkey[str(row["question_key"])])
        accepted = char_similarity < NEAR_DUPLICATE_THRESHOLD and token_similarity < NEAR_DUPLICATE_THRESHOLD
        novelty_rows.append({
            "sample_id_hash": stable_hash(sample_id(row)),
            "question_key_hash": stable_hash(str(row["question_key"])),
            "max_char_5gram_jaccard": char_similarity,
            "max_token_jaccard": token_similarity,
            "near_duplicate_pass": accepted,
        })
        if accepted:
            stage_novel.append(row)
    write_jsonl(args.out_dir / "private/source_packets/exp39b_r1_row_novelty_audit.jsonl", novelty_rows)
    novel_qkeys = {str(row["question_key"]) for row in stage_novel}
    can_prepare = len(stage_novel) >= args.rows and len(novel_qkeys) >= args.rows
    selected = select_balanced(stage_novel, args.rows, args.seed) if can_prepare else []
    band_assignment = assign_bands(selected, args.seed) if selected else {}
    packets = [packet(row, band_assignment[sample_id(row)], args.seed) for row in selected]
    packet_path = args.out_dir / "private/source_packets/exp39b_source_anchor_packets.jsonl"
    write_jsonl(packet_path, packets)

    selected_sims = [novelty(row, exp39a_by_qkey[str(row["question_key"])]) for row in selected]
    selected_answer_hashes = {answer_hash(row) for row in selected}
    selected_assessment_keys = {assessment_key(row) for row in selected}
    selected_qkeys = {str(row["question_key"]) for row in selected}
    source_overlap = sum(sample_id(row) in exp39a_ids for row in selected)
    answer_overlap = len(selected_answer_hashes & exp39a_answer_hashes)
    assessment_overlap = len(selected_assessment_keys & exp39a_assessment_keys)
    qkey_overlap = len(selected_qkeys & exp39a_qkeys)
    same_generator = sum(
        any(generator(row) == generator(other) for other in exp39a_by_qkey[str(row["question_key"])]) for row in selected
    )
    same_metric = sum(
        any(metric(row) == metric(other) for other in exp39a_by_qkey[str(row["question_key"])]) for row in selected
    )
    stage_counts = {
        "eligible_initial": len(stage_eligible), "after_source_id_exclusion": len(stage_id),
        "after_answer_hash_exclusion": len(stage_hash), "after_assessment_key_exclusion": len(stage_assessment),
        "after_near_duplicate_exclusion": len(stage_novel),
    }
    audit = {
        "train_rows": len(train), "train_question_keys": len({str(row["question_key"]) for row in train}),
        "exp39a_source_rows": len(exp39a_ids), "exp39a_question_keys": len(exp39a_qkeys),
        **stage_counts, "eligible_qkeys_after_all_exclusions": len(novel_qkeys),
        "selected_rows": len(selected), "selected_question_keys": len(selected_qkeys),
        "source_id_overlap": source_overlap, "exact_answer_hash_overlap": answer_overlap,
        "assessment_key_overlap": assessment_overlap, "question_key_overlap_count": qkey_overlap,
        "question_key_overlap_rate": qkey_overlap / len(selected_qkeys) if selected_qkeys else 0.0,
        "max_char_5gram_similarity": max((value[0] for value in selected_sims), default=0.0),
        "mean_char_5gram_similarity": sum(value[0] for value in selected_sims) / len(selected_sims) if selected_sims else 0.0,
        "max_token_jaccard": max((value[1] for value in selected_sims), default=0.0),
        "mean_token_jaccard": sum(value[1] for value in selected_sims) / len(selected_sims) if selected_sims else 0.0,
        "same_generator_count": same_generator, "same_metric_count": same_metric,
        "inference_scope": "response_disjoint_within_seen_question_clusters",
        "dev_access_count": 0, "test_access_count": 0,
    }
    write_csv(args.out_dir / "tables/exp39b_r1_source_novelty_audit.csv", [audit])
    write_csv(args.out_dir / "tables/exp39b_r1_exp39a_overlap_audit.csv", [audit])
    write_csv(args.out_dir / "tables/exp39b_exp39a_overlap_audit.csv", [{
        "train_rows": len(train), "train_question_keys": audit["train_question_keys"],
        "exp39a_source_rows": len(exp39a_ids), "exp39a_question_keys": len(exp39a_qkeys),
        "eligible_before_exclusion": len(stage_eligible), "eligible_after_source_id_exclusion": len(stage_id),
        "eligible_after_source_and_qkey_exclusion": len(stage_novel), "remaining_question_keys": len(novel_qkeys),
        "selected_source_overlap": source_overlap, "selected_qkey_overlap": qkey_overlap,
        "dev_access_count": 0, "test_access_count": 0,
    }])
    distribution = []
    dimensions_map = {
        "language": lambda row: row["language"], "metric_group": lambda row: row["metric_group"],
        "source_label": lambda row: row["source_label_5"], "generator_model": lambda row: row["generator_model"],
        "target_band": lambda row: row["target_band_name"],
    }
    for name, getter in dimensions_map.items():
        for value, count in sorted(Counter(str(getter(row)) for row in packets).items()):
            distribution.append({"dimension": name, "value": value, "count": count, "rate": count / len(packets)})
    write_csv(args.out_dir / "tables/exp39b_r1_source_distribution.csv", distribution)
    write_csv(args.out_dir / "tables/exp39b_source_distribution.csv", distribution)
    qkey_distribution = [{"sources_per_question_key": 1, "question_key_count": len(selected_qkeys)}] if selected else []
    write_csv(
        args.out_dir / "tables/exp39b_r1_source_qkey_distribution.csv", qkey_distribution,
        fieldnames=["sources_per_question_key", "question_key_count"],
    )
    write_csv(
        args.out_dir / "tables/exp39b_source_qkey_distribution.csv", qkey_distribution,
        fieldnames=["sources_per_question_key", "question_key_count"],
    )

    for path in sorted(PROMPT_DIR.glob("exp39b_*.md")):
        shutil.copy2(path, args.out_dir / "prompts" / path.name)
    for path in sorted(SCHEMA_DIR.glob("exp39b_*.json")):
        shutil.copy2(path, args.out_dir / "schemas" / path.name)
    amendment_source = Path("thesis_exp/exp39b_educfa_rlcr/protocols/exp39b_r1_sampling_amendment.md")
    shutil.copy2(amendment_source, args.out_dir / "protocols" / amendment_source.name)

    base_gate = read_json(ROOT / "configs/exp39b_acceptance_gate.json")
    r1_gate = {**base_gate, "source_qkeys_min": 60}
    protocol = {
        "experiment": "Exp39B-R1 EduCFA-RLCR Response-Disjoint Protocol Pilot",
        "base_commit": BASE_COMMIT, "seed": args.seed,
        "qwen_model": "qwen3.7-max", "deepseek_model": "deepseek-v4-pro",
        "temperature": 0, "thinking": "disabled", "target_band_counts": BAND_COUNTS,
        "target_bands": BANDS, "prompts_schemas_frozen": True, "acceptance_rules_frozen": True,
        "dev_access_count": 0, "test_access_count": 0,
    }
    source_lock = {
        "experiment": protocol["experiment"], "status": "SOURCE_LOCKED" if can_prepare else "SOURCE_POOL_NO_GO_R1",
        "source_independence_unit": "response", "question_cluster_reuse": True,
        "inference_scope": "response_disjoint_within_seen_question_clusters",
        "train_path": str(args.train_jsonl), "train_sha256": sha256_file(args.train_jsonl),
        "train_rows": len(train), "train_question_keys": audit["train_question_keys"],
        "exp39a_source_rows_excluded": len(exp39a_ids), "exp39a_question_keys_reused": qkey_overlap,
        "eligible_after_exclusion": len(stage_novel), "eligible_question_keys_after_exclusion": len(novel_qkeys),
        "source_rows": len(packets), "source_question_keys": len(selected_qkeys),
        "source_id_overlap": source_overlap, "answer_hash_overlap": answer_overlap,
        "assessment_key_overlap": assessment_overlap, "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "selection_seed": args.seed, "prompt_schema_hashes": frozen_hashes,
        "base_acceptance_gate_sha256": base_gate_hash,
        "selection_uses_teacher_scores": False, "selection_uses_model_predictions": False,
        "dev_access_count": 0, "test_access_count": 0,
    }
    freeze(args.out_dir / "configs/exp39b_r1_source_lock.json", source_lock)
    freeze(args.out_dir / "configs/exp39b_source_lock.json", source_lock)
    freeze(args.out_dir / "configs/exp39b_r1_protocol_lock.json", protocol)
    freeze(args.out_dir / "configs/exp39b_protocol_lock.json", protocol)
    freeze(args.out_dir / "configs/exp39b_r1_acceptance_gate.json", r1_gate)
    freeze(args.out_dir / "configs/exp39b_acceptance_gate.json", r1_gate)
    decision = {
        "status": "PROTOCOL_PREPARE_GO" if can_prepare else "SOURCE_POOL_NO_GO_R1",
        "fresh_source_count": len(packets), "fresh_question_key_count": len(selected_qkeys),
        "eligible_after_exclusion": len(stage_novel), "eligible_question_keys_after_exclusion": len(novel_qkeys),
        "recommend_api_calls": can_prepare, "inference_scope": source_lock["inference_scope"],
        "dev_access_count": 0, "test_access_count": 0,
    }
    freeze(args.out_dir / "decision/exp39b_r1_protocol_prepare_decision.json", decision)
    freeze(args.out_dir / "decision/exp39b_protocol_prepare_decision.json", decision)
    write_json(args.out_dir / "hashes/exp39b_r1_source_hashes.json", {
        "selected_source_ids_sha256": stable_hash(sorted(row["source_sample_id"] for row in packets)),
        "selected_question_keys_sha256": stable_hash(sorted(row["question_key"] for row in packets)),
        "selected_answer_hashes_sha256": stable_hash(sorted(selected_answer_hashes)),
        "private_packet_sha256": sha256_file(packet_path),
    })
    write_json(args.out_dir / "hashes/exp39b_r1_prompt_schema_hashes.json", frozen_hashes)
    report = [
        "# Exp39B-R1 response-disjoint source preparation", "",
        f"- Status: **{decision['status']}**", f"- Eligible after all exclusions: `{len(stage_novel)}` rows / `{len(novel_qkeys)}` qkeys",
        f"- Selected: `{len(packets)}` rows / `{len(selected_qkeys)}` qkeys",
        f"- Source/answer/assessment overlap: `{source_overlap}/{answer_overlap}/{assessment_overlap}`",
        f"- Expected question-key overlap with Exp39A: `{qkey_overlap}`",
        f"- Maximum character/token Jaccard: `{audit['max_char_5gram_similarity']:.4f}/{audit['max_token_jaccard']:.4f}`",
        "- Inference scope: `response_disjoint_within_seen_question_clusters`.",
        "- Frozen prompts, schemas, method gates, and target bands match b4fcc26.",
        "- No API, dev, or test data was used during preparation.",
    ]
    (args.out_dir / "reports/exp39b_r1_protocol_prepare_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    if not can_prepare:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
