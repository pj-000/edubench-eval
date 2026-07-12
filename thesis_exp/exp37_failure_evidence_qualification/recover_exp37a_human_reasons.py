"""Strictly recover train-only human reasons; never use fuzzy matching."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp37_failure_evidence_qualification.common import (  # noqa: E402
    REASON_PATHS, ROOT, TRAIN_PATH, load_reason_rows, read_jsonl, sample_id,
    strict_reason_recovery, write_csv, write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = read_jsonl(args.train_jsonl)
    all_reason_rows = load_reason_rows()
    aggregate_summary, recovered = strict_reason_recovery(train, all_reason_rows)
    # A recovered row is usable only if the exact key was unique across all
    # supplied files and its source score agrees with the split label.
    view_by_id = {}
    packet_dir = args.out_dir / "private_packets"
    for path in sorted(packet_dir.glob("exp37a_*_blind_packet.jsonl")):
        if "adjudication" in path.name:
            continue
        view = path.name.removeprefix("exp37a_").removesuffix("_blind_packet.jsonl")
        for packet in read_jsonl(path):
            view_by_id[str(packet["sample_id"])] = view
    if len(view_by_id) != 196:
        raise ValueError(f"Expected prepared view packets for 196 samples, found {len(view_by_id)}")
    by_source = []
    for path in REASON_PATHS:
        rows = [row for row in all_reason_rows if row.get("_source_file") == path.name]
        summary, _ = strict_reason_recovery(train, rows)
        for view in ("low_tail_all", "boundary_view", "high_control_view"):
            subset = [row for row in summary if view_by_id.get(row["sample_id"]) == view]
            by_source.append({
                "view": view, "annotator": path.name,
                "recovered_count": sum(int(row["recovered"]) for row in subset),
                "unique_match_count": sum(int(row["unique_match_count"]) for row in subset),
                "ambiguous_count": sum(int(row["ambiguous_count"]) for row in subset),
                "unavailable_count": sum(int(row["unavailable_count"]) for row in subset),
                "score_consistent_count": sum(int(row["score_consistent"]) for row in subset),
                "score_inconsistent_count": sum(int(row["score_inconsistent"]) for row in subset),
                "matching_rule": "exact sample_id or exact normalized question+answer+metric+generator/model only",
            })
    for view in ("low_tail_all", "boundary_view", "high_control_view"):
        subset = [row for row in aggregate_summary if view_by_id.get(row["sample_id"]) == view]
        by_source.append({
            "view": view, "annotator": "all_sources_strict_union",
            "recovered_count": sum(int(row["recovered"]) for row in subset),
            "unique_match_count": sum(int(row["unique_match_count"]) for row in subset),
            "ambiguous_count": sum(int(row["ambiguous_count"]) for row in subset),
            "unavailable_count": sum(int(row["unavailable_count"]) for row in subset),
            "score_consistent_count": sum(int(row["score_consistent"]) for row in subset),
            "score_inconsistent_count": sum(int(row["score_inconsistent"]) for row in subset),
            "matching_rule": "exact sample_id or exact normalized question+answer+metric+generator/model only",
        })
    write_csv(args.out_dir / "tables/exp37a_human_reason_recovery.csv", by_source)
    write_jsonl(args.out_dir / "private_reference/recovered_human_reasons.jsonl", recovered)
    print({"train_rows": len(train), "recovered_unique_rows": len(recovered), "test_access_count": 0})


if __name__ == "__main__":
    main()
