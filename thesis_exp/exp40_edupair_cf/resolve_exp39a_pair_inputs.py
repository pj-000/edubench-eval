"""Resolve the frozen Exp39A source/counterfactual pairs without evaluation data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp40_edupair_cf.common import (  # noqa: E402
    EXP39A_ROOT,
    ROOT,
    ensure_output_dirs,
    read_jsonl,
    sha256_file,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp39a-root", type=Path, default=EXP39A_ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def fail_missing(out_dir: Path, missing: list[Path]) -> None:
    decision = {
        "status": "MISSING_EXP39A_PRIVATE_PAIRS",
        "missing_paths": [str(path) for path in missing],
        "recommend_pairwise_verification": False,
        "recommend_groupcv_training": False,
        "dev_access_count": 0,
        "test_access_count": 0,
    }
    write_json(out_dir / "decision/exp40a_pairwise_qualification_decision.json", decision)
    raise SystemExit("Missing Exp39A private pair inputs; Exp39B plans may not be substituted: " + ", ".join(map(str, missing)))


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.out_dir)
    paths = {
        "source_lock": args.exp39a_root / "configs/exp39a_source_lock.json",
        "protocol_lock": args.exp39a_root / "configs/exp39a_generation_protocol_lock.json",
        "source_hashes": args.exp39a_root / "hashes/exp39a_source_anchor_hashes.json",
        "source_packets": args.exp39a_root / "private/source_packets/exp39a_source_anchor_packets.jsonl",
        "generated_candidates": args.exp39a_root / "private/generated_candidates/exp39a_qwen_generated_candidates.jsonl",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        fail_missing(args.out_dir, missing)

    source_lock = json.loads(paths["source_lock"].read_text(encoding="utf-8"))
    protocol_lock = json.loads(paths["protocol_lock"].read_text(encoding="utf-8"))
    packets = read_jsonl(paths["source_packets"])
    generated = read_jsonl(paths["generated_candidates"])
    if source_lock.get("source_rows") != 240 or len(packets) != 240 or len(generated) != 240:
        raise ValueError("Exp40A requires exactly 240 frozen Exp39A source rows and 240 generated candidates")
    if protocol_lock.get("acceptance_rules_frozen") is not True:
        raise ValueError("Exp39A generation protocol is not frozen")

    packet_by_id = {str(row["sample_id"]): row for row in packets}
    generated_by_id = {str(row["sample_id"]): row for row in generated}
    if len(packet_by_id) != 240 or len(generated_by_id) != 240 or set(packet_by_id) != set(generated_by_id):
        raise ValueError("Exp39A packet/candidate IDs do not form a one-to-one 240-pair join")

    resolved: list[dict[str, Any]] = []
    for exp39a_id in sorted(packet_by_id):
        packet, candidate = packet_by_id[exp39a_id], generated_by_id[exp39a_id]
        if str(packet["source_sample_id"]) != str(candidate["source_sample_id"]):
            raise ValueError(f"Source ID mismatch for Exp39A pair {exp39a_id}")
        original = str(packet.get("original_answer") or "")
        counterfactual = str(candidate.get("counterfactual_answer") or "")
        if not original.strip() or not counterfactual.strip() or original.strip() == counterfactual.strip():
            raise ValueError(f"Empty or unchanged Exp39A pair: {exp39a_id}")
        pair_id = "exp40a_" + stable_hash(
            {"exp39a_id": exp39a_id, "source_sample_id": packet["source_sample_id"]}
        )[:24]
        resolved.append(
            {
                "pair_id": pair_id,
                "exp39a_pair_id": exp39a_id,
                "sample_id": str(packet["source_sample_id"]),
                "question_key": str(packet["question_key"]),
                "question": packet.get("question", ""),
                "original_answer": original,
                "counterfactual_answer": counterfactual,
                "metric": packet.get("metric", ""),
                "metric_group": packet.get("metric_group", "unknown"),
                "rubric": packet.get("rubric", []),
                "metadata": packet.get("metadata", {}),
                "language": packet.get("language", "unknown"),
                "subject": packet.get("subject", "unknown"),
                "operator": candidate.get("operator", packet.get("assigned_operator")),
                "targeted_rubric_clause": candidate.get("targeted_rubric_clause", ""),
            }
        )
    if len({row["pair_id"] for row in resolved}) != 240:
        raise RuntimeError("Resolved Exp40A pair IDs are not unique")

    resolved_path = args.out_dir / "private/resolved_pairs/exp40a_resolved_exp39a_pairs.jsonl"
    write_jsonl(resolved_path, resolved)
    manifest = []
    for role, path in paths.items():
        rows = 240 if role in {"source_packets", "generated_candidates"} else 1
        manifest.append({"role": role, "path": str(path), "rows": rows, "sha256": sha256_file(path)})
    manifest.append(
        {
            "role": "resolved_private_pairs",
            "path": str(resolved_path),
            "rows": len(resolved),
            "sha256": sha256_file(resolved_path),
        }
    )
    write_csv(args.out_dir / "tables/exp40a_resolved_pair_input_manifest.csv", manifest)
    write_json(
        args.out_dir / "hashes/exp40a_pair_input_hashes.json",
        {
            "artifacts": {row["role"]: {"path": row["path"], "rows": row["rows"], "sha256": row["sha256"]} for row in manifest},
            "pair_join_sha256": stable_hash(
                [[row["pair_id"], row["exp39a_pair_id"], row["sample_id"], row["question_key"]] for row in resolved]
            ),
            "pair_count": 240,
            "dev_access_count": 0,
            "test_access_count": 0,
        },
    )
    print(json.dumps({"status": "RESOLVED", "pairs": 240, "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
