"""Prepare Exp24 score-channel ORC-DPO data.

Exp24 turns the matched R7D/R7E pair pool into R7G:

- DPO main channel compares score-only responses, not full reason+score text.
- recovered human rationale is kept as an auxiliary target, never in the user
  prompt;
- risk metadata and ordinal distance are explicit per pair for ORC-DPO weights
  and margins.

Dev/test files are used only for sample_id/question_key leakage checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import clean, clamp_score  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_R7D = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42/"
    "data/edubench_r7d_strict_label_consistent_reason_real_dpo_train.json"
)
DEFAULT_R7E = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp22_r7_matched_controls_seed42/"
    "data/edubench_r7e_matched_score_only_strict_real_dpo_train.json"
)
DEFAULT_DEV_JSONL = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST_JSONL = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp24_orc_dpo_seed42")
R7G_NAME = "edubench_r7g_orc_score_channel_reason_aux_train"
R7G_FILE = "data/edubench_r7g_orc_score_channel_reason_aux_train.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def sha1(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def parse_payload(message: dict[str, Any]) -> dict[str, Any]:
    try:
        data = json.loads(clean(message.get("content")))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def assistant_message(payload: dict[str, Any]) -> dict[str, str]:
    return {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}


def pair_key(pair: dict[str, Any]) -> tuple[str, str, int, int, str, str]:
    return (
        clean(pair.get("source_sample_id")),
        clean(pair.get("source_question_key")),
        clamp_score(pair.get("gold_label")),
        clamp_score(pair.get("rejected_score")),
        clean(pair.get("risk_type")),
        clean(pair.get("reason_hash")),
    )


def stable_pair_id(pair: dict[str, Any]) -> str:
    return sha1("|".join(str(item) for item in pair_key(pair)))


def split_ids(path: Path) -> tuple[set[str], set[str], int]:
    sample_ids: set[str] = set()
    question_keys: set[str] = set()
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            sid = clean(row.get("sample_id") or row.get("record_id") or row.get("id"))
            qkey = clean(row.get("question_key") or row.get("question_id"))
            if sid:
                sample_ids.add(sid)
            if qkey:
                question_keys.add(qkey)
    return sample_ids, question_keys, n


def risk_flags(gold: int, rejected: int) -> dict[str, int]:
    return {
        "LH": int(gold <= 2 and rejected >= 4),
        "LM": int(gold <= 2 and rejected == 3),
        "HL": int(gold >= 4 and rejected <= 2),
        "HM": int(gold >= 4 and rejected == 3),
    }


def risk_type_from_scores(gold: int, rejected: int) -> str:
    if gold <= 2 and rejected >= 4:
        return "low_to_high_real_model_error"
    if gold <= 2 and rejected == 3:
        return "low_to_mid_real_model_error"
    if gold >= 4 and rejected <= 2:
        return "high_to_low_real_model_error"
    if gold >= 4 and rejected == 3:
        return "high_to_mid_real_model_error"
    if rejected > gold:
        return "upward_real_model_error"
    if rejected < gold:
        return "downward_real_model_error"
    return "same_score_non_error"


def make_r7g_pair(r7d_pair: dict[str, Any], r7e_pair: dict[str, Any], idx: int) -> dict[str, Any]:
    d_key = pair_key(r7d_pair)
    e_key = pair_key(r7e_pair)
    if d_key != e_key:
        raise ValueError(f"R7D/R7E pair key mismatch at index={idx}: {d_key} != {e_key}")

    gold = clamp_score(r7d_pair.get("gold_label"))
    rejected = clamp_score(r7d_pair.get("rejected_score"))
    chosen_payload = parse_payload(r7d_pair["chosen"])
    reason = clean(chosen_payload.get("reason"))
    if not reason:
        raise ValueError(f"missing human reason for R7D pair index={idx}")
    if clamp_score(chosen_payload.get("score")) != gold:
        raise ValueError(f"chosen score mismatch for R7D pair index={idx}")
    if parse_payload(r7e_pair["chosen"]) != {"score": gold}:
        raise ValueError(f"R7E chosen is not score-only gold at index={idx}")
    if parse_payload(r7e_pair["rejected"]) != {"score": rejected}:
        raise ValueError(f"R7E rejected is not score-only rejected at index={idx}")

    flags = risk_flags(gold, rejected)
    ordinal_distance = abs(gold - rejected)
    pair_id = stable_pair_id(r7d_pair)
    source_risk = clean(r7d_pair.get("risk_type"))
    expected_risk = risk_type_from_scores(gold, rejected)
    return {
        "messages": r7d_pair["messages"],
        "chosen_score_response": assistant_message({"score": gold}),
        "rejected_score_response": assistant_message({"score": rejected}),
        "auxiliary_reason_target": reason,
        "pair_id": pair_id,
        "source_sample_id": clean(r7d_pair.get("source_sample_id")),
        "question_key": clean(r7d_pair.get("source_question_key")),
        "gold_label": gold,
        "rejected_score": rejected,
        "risk_type": source_risk,
        "score_derived_risk_type": expected_risk,
        "risk_type_matches_score_direction": source_risk == expected_risk,
        "ordinal_distance": ordinal_distance,
        "has_human_reason": bool(reason),
        "reason_hash": clean(r7d_pair.get("reason_hash")) or sha1(reason),
        "reason_char_len": len(reason),
        "reason_token_proxy_len": len(reason.split()),
        **flags,
        "matched_from": "r7d_r7e",
        "matched_pair_index": idx,
        "r7d_dataset_variant": clean(r7d_pair.get("dataset_variant")),
        "r7e_dataset_variant": clean(r7e_pair.get("dataset_variant")),
        "rejected_source": clean(r7d_pair.get("rejected_source")),
    }


def dataset_info() -> dict[str, Any]:
    return {
        R7G_NAME: {
            "file_name": R7G_FILE,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "messages",
                "chosen": "chosen_score_response",
                "rejected": "rejected_score_response",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }


def summarize_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": "", "p50": "", "mean": "", "p90": "", "max": ""}
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "p90": ordered[min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)],
        "max": ordered[-1],
    }


def build_tables(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    alignment_rows = [
        {
            "matched_pair_index": row["matched_pair_index"],
            "pair_id": row["pair_id"],
            "source_sample_id": row["source_sample_id"],
            "question_key": row["question_key"],
            "gold_label": row["gold_label"],
            "rejected_score": row["rejected_score"],
            "risk_type": row["risk_type"],
            "score_derived_risk_type": row["score_derived_risk_type"],
            "risk_type_matches_score_direction": row["risk_type_matches_score_direction"],
            "ordinal_distance": row["ordinal_distance"],
            "has_human_reason": row["has_human_reason"],
            "reason_hash": row["reason_hash"],
        }
        for row in rows
    ]
    write_csv(args.out_dir / "tables" / "exp24_r7g_alignment.csv", alignment_rows)

    risk_counts = Counter(row["risk_type"] for row in rows)
    distance_counts = Counter(str(row["ordinal_distance"]) for row in rows)
    flag_counts = Counter()
    for row in rows:
        for name in ("LH", "LM", "HL", "HM"):
            flag_counts[name] += int(row[name])
    metadata_rows = [
        {"metric": "pair_count", "value": len(rows)},
        {"metric": "unique_pair_id_count", "value": len({row["pair_id"] for row in rows})},
        {"metric": "has_human_reason_count", "value": sum(1 for row in rows if row["has_human_reason"])},
        {
            "metric": "risk_type_mismatch_count",
            "value": sum(1 for row in rows if not row["risk_type_matches_score_direction"]),
        },
    ]
    metadata_rows.extend({"metric": f"risk_type::{key}", "value": value} for key, value in sorted(risk_counts.items()))
    metadata_rows.extend({"metric": f"ordinal_distance::{key}", "value": value} for key, value in sorted(distance_counts.items()))
    metadata_rows.extend({"metric": f"risk_flag::{key}", "value": value} for key, value in sorted(flag_counts.items()))
    write_csv(args.out_dir / "tables" / "exp24_orc_metadata_summary.csv", metadata_rows)

    reason_stats = summarize_numbers([int(row["reason_char_len"]) for row in rows])
    token_stats = summarize_numbers([int(row["reason_token_proxy_len"]) for row in rows])
    write_csv(
        args.out_dir / "tables" / "exp24_reason_aux_length_stats.csv",
        [
            {"length_type": "chars", **reason_stats},
            {"length_type": "whitespace_token_proxy", **token_stats},
        ],
    )

    dev_ids, dev_qkeys, dev_n = split_ids(args.dev_jsonl)
    test_ids, test_qkeys, test_n = split_ids(args.test_jsonl)
    source_ids = {row["source_sample_id"] for row in rows}
    qkeys = {row["question_key"] for row in rows}
    reason_in_prompt = 0
    for row in rows:
        prompt_text = json.dumps(row.get("messages") or [], ensure_ascii=False)
        if clean(row.get("auxiliary_reason_target")) and clean(row.get("auxiliary_reason_target")) in prompt_text:
            reason_in_prompt += 1
    leakage_rows = [
        {
            "dataset_name": R7G_NAME,
            "pairs": len(rows),
            "dev_rows_read_for_id_guard_only": dev_n,
            "test_rows_read_for_id_guard_only": test_n,
            "dev_sample_id_overlap": len(source_ids & dev_ids),
            "dev_question_key_overlap": len(qkeys & dev_qkeys),
            "test_sample_id_overlap": len(source_ids & test_ids),
            "test_question_key_overlap": len(qkeys & test_qkeys),
            "human_reason_in_prompt_count": reason_in_prompt,
            "dev_test_label_read": False,
            "leakage_pass": len(source_ids & dev_ids) == 0
            and len(qkeys & dev_qkeys) == 0
            and len(source_ids & test_ids) == 0
            and len(qkeys & test_qkeys) == 0
            and reason_in_prompt == 0,
        }
    ]
    write_csv(args.out_dir / "tables" / "exp24_leakage_audit.csv", leakage_rows)


def write_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    risk_counts = Counter(row["risk_type"] for row in rows)
    distance_counts = Counter(row["ordinal_distance"] for row in rows)
    lines = [
        "# Exp24 ORC-DPO Score-Channel Data Preparation",
        "",
        "R7G is derived from the exact R7D/R7E matched 429-pair pool.",
        "",
        "## Key Design",
        "",
        "- DPO main channel is score-only: `{\"score\": gold}` vs `{\"score\": rejected}`.",
        "- Human rationale is stored as `auxiliary_reason_target` and is not included in the prompt.",
        "- `risk_type` and `ordinal_distance` are explicit metadata for ORC-DPO weights and margins.",
        "",
        "## Counts",
        "",
        f"- pairs: {len(rows)}",
        f"- unique pair ids: {len({row['pair_id'] for row in rows})}",
        f"- all have human reason: {all(row['has_human_reason'] for row in rows)}",
        "",
        "## Risk Type Distribution",
        "",
        "| risk_type | count |",
        "|---|---:|",
    ]
    for key, value in sorted(risk_counts.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Ordinal Distance Distribution", "", "| distance | count |", "|---:|---:|"])
    for key, value in sorted(distance_counts.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Dev/test are used only for ID and question-key leakage checks.",
            "- Test labels are not read.",
            "- Human rationale is not in the user prompt.",
        ]
    )
    write_text(args.out_dir / "reports" / "exp24_orc_data_report.md", "\n".join(lines))


def build(args: argparse.Namespace) -> dict[str, Any]:
    r7d = load_json(args.r7d_json)
    r7e = load_json(args.r7e_json)
    if len(r7d) != len(r7e):
        raise SystemExit(f"R7D/R7E length mismatch: {len(r7d)} != {len(r7e)}")
    rows = [make_r7g_pair(d, e, idx) for idx, (d, e) in enumerate(zip(r7d, r7e))]
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise SystemExit("R7G pair_id is not unique; refusing to continue.")

    data_path = args.out_dir / R7G_FILE
    write_json_array(data_path, rows)
    write_json(args.out_dir / "dataset_info.json", dataset_info())
    write_json(args.out_dir / "dataset_info_exp24_snippet.json", dataset_info())
    build_tables(rows, args)
    write_report(rows, args)
    decision = {
        "status": "READY_FOR_ORC_DPO_TRAINING",
        "dataset_name": R7G_NAME,
        "pair_count": len(rows),
        "r7d_r7e_aligned": True,
        "test_read": False,
        "dev_test_label_read": False,
        "human_reason_in_prompt": False,
    }
    write_json(args.out_dir / "decision" / "exp24_orc_data_decision.json", decision)
    return {"out_dir": str(args.out_dir), "data": str(data_path), **decision}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp24 R7G score-channel ORC-DPO data.")
    parser.add_argument("--r7d-json", type=Path, default=DEFAULT_R7D)
    parser.add_argument("--r7e-json", type=Path, default=DEFAULT_R7E)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_JSONL)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
