"""Prepare frozen, label-blind Qwen score-range packets for Exp38A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp38_hails_score.common import (
    EXPECTED_VIEWS,
    PACKET_PATH,
    ROOT,
    TRAIN_PATH,
    frozen_view_map,
    read_jsonl,
    resolve_final_reference,
    sample_id,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)

PROMPT_PATH = Path("thesis_exp/exp38_hails_score/prompts/exp38a_qwen_score_range_prompt.md")
SCHEMA_PATH = Path("thesis_exp/exp38_hails_score/schemas/exp38a_qwen_score_range_schema.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("qualification", "all_train"), default="qualification")
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--frozen-packets", type=Path, default=PACKET_PATH)
    parser.add_argument("--final-reference", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--model", default="qwen3.7-max")
    return parser.parse_args()


def packet_from_train(row: dict[str, Any], view: str) -> dict[str, Any]:
    sid = sample_id(row)
    metadata = {
        "language": row.get("language"),
        "subject": row.get("subject_canonical") or row.get("subject_raw"),
        "education_level": row.get("education_level_canonical") or row.get("education_level_raw"),
        "scenario": row.get("scenario_canonical") or row.get("scenario_raw"),
        "metric_group": row.get("metric_group"),
    }
    review_text = (
        "<CONTEXT_ONLY_ORIGINAL_TASK>\n"
        f"{row.get('question', '')}\n"
        f"Evaluation dimension: {row.get('metric_canonical') or row.get('metric_raw') or row.get('metric_id', '')}\n"
        f"Canonical rubric: {json.dumps(row.get('rubric', ''), ensure_ascii=False)}\n"
        f"Non-label metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n"
        "</CONTEXT_ONLY_ORIGINAL_TASK>\n"
        "<EVALUATOR_OUTPUT_TO_SCORE>\n"
        f"{row.get('answer', '')}\n"
        "</EVALUATOR_OUTPUT_TO_SCORE>"
    )
    packet = {"sample_id": sid, "view": view, "review_text": review_text}
    packet["packet_hash"] = stable_hash(packet)
    return packet


def ensure_protocol_dirs(out_dir: Path) -> None:
    for name in ("configs", "tables", "reports", "decision", "hashes", "raw_api", "parsed_ranges_private", "private_reference", "private/data", "private/groupcv_predictions", "logs_private", "checkpoints_private"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_protocol_dirs(args.out_dir)
    train_rows = read_jsonl(args.train_jsonl)
    if len(train_rows) != 2654 or len({sample_id(row) for row in train_rows}) != 2654:
        raise ValueError("Paper-like train must contain 2654 unique rows")
    train = {sample_id(row): row for row in train_rows}
    view_map = frozen_view_map()

    prompt_hash = sha256_file(PROMPT_PATH)
    schema_hash = sha256_file(SCHEMA_PATH)
    lock_path = args.out_dir / "configs/exp38a_range_protocol_lock.json"
    lock = {
        "experiment": "Exp38A HAILS-Score",
        "prompt_path": str(PROMPT_PATH),
        "prompt_sha256": prompt_hash,
        "schema_path": str(SCHEMA_PATH),
        "schema_sha256": schema_hash,
        "model": args.model,
        "temperature": 0,
        "enable_thinking": False,
        "max_tokens": 512,
        "response_format": "json_object",
        "qualification_rows": 196,
        "qualification_view_counts": EXPECTED_VIEWS,
        "train_rows": 2654,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        for key in ("prompt_sha256", "schema_sha256", "model", "temperature", "enable_thinking", "max_tokens"):
            if existing.get(key) != lock.get(key):
                raise RuntimeError(f"Frozen range protocol changed at {key}; refusing reuse of the 196 qualification set")
    else:
        write_json(lock_path, lock)

    if args.scope == "qualification":
        frozen = read_jsonl(args.frozen_packets)
        frozen_ids = {sample_id(row) for row in frozen}
        if len(frozen) != 196 or frozen_ids != set(view_map):
            raise ValueError("Qualification IDs must exactly equal the frozen Exp37A-R1 196 IDs")
        reference_path, reference_rows = resolve_final_reference(explicit=args.final_reference)
        reference_type = sorted({str(row.get("reference_type")) for row in reference_rows})
        if len(reference_type) != 1:
            raise ValueError("Final reference must have one reference_type")
        packets = [packet_from_train(train[sid], view_map[sid]) for sid in sorted(frozen_ids)]
        output_path = args.out_dir / "private_reference/exp38a_qwen_range_qualification_packets.jsonl"
        write_jsonl(output_path, packets)
        manifest = [
            {
                "input_type": "paper_like_train",
                "resolved_path": str(args.train_jsonl),
                "row_count": len(train_rows),
                "unique_sample_ids": len(train),
                "sha256": sha256_file(args.train_jsonl),
                "reference_type": "training_source",
            },
            {
                "input_type": "frozen_exp37_packets",
                "resolved_path": str(args.frozen_packets),
                "row_count": len(frozen),
                "unique_sample_ids": len(frozen_ids),
                "sha256": sha256_file(args.frozen_packets),
                "reference_type": "frozen_qualification_ids",
            },
            {
                "input_type": "final_reference",
                "resolved_path": str(reference_path),
                "row_count": len(reference_rows),
                "unique_sample_ids": len({sample_id(row) for row in reference_rows}),
                "sha256": sha256_file(reference_path),
                "reference_type": reference_type[0],
            },
        ]
        write_csv(args.out_dir / "tables/exp38a_resolved_input_manifest.csv", manifest)
        write_json(args.out_dir / "hashes/exp38a_qualification_hashes.json", {
            "sample_ids_sha256": stable_hash(sorted(frozen_ids)),
            "packet_sha256": sha256_file(output_path),
            "prompt_sha256": prompt_hash,
            "schema_sha256": schema_hash,
        })
        report = [
            "# Exp38A range qualification preparation", "",
            "- Frozen qualification rows: `196`",
            "- Unique sample IDs: `196`",
            f"- View counts: `{json.dumps(EXPECTED_VIEWS, sort_keys=True)}`",
            "- Exp37A-R1 frozen IDs exactly preserved: `true`",
            f"- Final reference type: `{reference_type[0]}`",
            f"- Prompt SHA-256: `{prompt_hash}`",
            f"- Schema SHA-256: `{schema_hash}`",
            "- Labels, previous teacher outputs, OOF predictions, and reviewer decisions are absent from API packets.",
            "- Dev/test access count: `0`",
        ]
        (args.out_dir / "reports/exp38a_range_qualification_prepare_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    else:
        decision_path = args.out_dir / "decision/exp38a_range_qualification_decision.json"
        if not decision_path.exists() or not json.loads(decision_path.read_text(encoding="utf-8")).get("recommend_full_train_range_annotation"):
            raise RuntimeError("Full-train packet preparation is blocked until every qualification gate passes")
        packets = [packet_from_train(row, "all_train") for row in train_rows]
        output_path = args.out_dir / "private_reference/exp38a_qwen_range_all_train_packets.jsonl"
        write_jsonl(output_path, packets)
        write_json(args.out_dir / "hashes/exp38a_all_train_packet_hashes.json", {
            "rows": len(packets), "unique_ids": len({sample_id(row) for row in packets}), "packet_sha256": sha256_file(output_path)
        })
    print(json.dumps({"status": "PREPARED", "scope": args.scope, "rows": len(packets), "output": str(output_path), "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
