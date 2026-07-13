"""Independently validate the frozen Exp39B-R1 response-disjoint source lock."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp39b_educfa_rlcr.common import (  # noqa: E402
    EXP39A_PACKETS, R1_ROOT, TRAIN_PATH, character_ngrams, jaccard, normalize_answer,
    read_json, read_jsonl, sample_id, sha256_file, stable_hash, text_tokens, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=R1_ROOT)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--exp39a-packets", type=Path, default=EXP39A_PACKETS)
    return parser.parse_args()


def generator(row: dict) -> str:
    return str(row.get("generator_model") or row.get("answer_model") or "unknown")


def metric(row: dict) -> str:
    return str(row.get("metric_canonical") or row.get("metric_raw") or row.get("metric_id") or "unknown")


def assessment_key(row: dict) -> tuple[str, str, str, str]:
    return str(row["question_key"]), normalize_answer(str(row["answer"])), metric(row), generator(row)


def main() -> None:
    args = parse_args()
    source_lock = read_json(args.out_dir / "configs/exp39b_r1_source_lock.json")
    packets = read_jsonl(args.out_dir / "private/source_packets/exp39b_source_anchor_packets.jsonl")
    train = read_jsonl(args.train_jsonl)
    train_by_id = {sample_id(row): row for row in train}
    exp_packets = read_jsonl(args.exp39a_packets)
    exp_ids = {str(row["source_sample_id"]) for row in exp_packets}
    exp_rows = [train_by_id[source_id] for source_id in exp_ids]
    exp_hashes = {stable_hash(normalize_answer(str(row["answer"]))) for row in exp_rows}
    exp_keys = {assessment_key(row) for row in exp_rows}
    exp_by_qkey: dict[str, list[dict]] = defaultdict(list)
    for row in exp_rows:
        exp_by_qkey[str(row["question_key"])].append(row)
    selected_rows = [train_by_id[str(packet["source_sample_id"])] for packet in packets]
    similarities = []
    for row in selected_rows:
        candidate_grams = character_ngrams(str(row["answer"]))
        candidate_tokens = text_tokens(str(row["answer"]))
        for other in exp_by_qkey[str(row["question_key"])]:
            similarities.append((
                jaccard(candidate_grams, character_ngrams(str(other["answer"]))),
                jaccard(candidate_tokens, text_tokens(str(other["answer"]))),
            ))
    checks = {
        "source_rows_60": len(packets) == 60,
        "unique_question_keys_60": len({str(row["question_key"]) for row in packets}) == 60,
        "one_row_per_question_key": len(packets) == len({str(row["question_key"]) for row in packets}),
        "source_id_overlap_zero": not ({str(row["source_sample_id"]) for row in packets} & exp_ids),
        "answer_hash_overlap_zero": not ({stable_hash(normalize_answer(str(row["answer"]))) for row in selected_rows} & exp_hashes),
        "assessment_key_overlap_zero": not ({assessment_key(row) for row in selected_rows} & exp_keys),
        "near_duplicate_below_0p90": all(char < 0.90 and token < 0.90 for char, token in similarities),
        "train_hash_matches": source_lock["train_sha256"] == sha256_file(args.train_jsonl),
        "response_scope_declared": source_lock["inference_scope"] == "response_disjoint_within_seen_question_clusters",
        "question_cluster_reuse_declared": source_lock["question_cluster_reuse"] is True,
        "dev_test_zero": source_lock["dev_access_count"] == source_lock["test_access_count"] == 0,
    }
    decision = {
        "status": "SOURCE_LOCK_VALID" if all(checks.values()) else "SOURCE_LOCK_INVALID",
        "checks": checks, "source_rows": len(packets),
        "source_question_keys": len({str(row["question_key"]) for row in packets}),
        "max_char_5gram_jaccard": max((value[0] for value in similarities), default=0.0),
        "max_token_jaccard": max((value[1] for value in similarities), default=0.0),
        "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp39b_r1_source_lock_validation.json", decision)
    print(json.dumps(decision, sort_keys=True))
    if decision["status"] != "SOURCE_LOCK_VALID":
        raise SystemExit(4)


if __name__ == "__main__":
    main()
