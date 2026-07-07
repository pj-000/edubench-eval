"""Prepare Exp25 Structured SRC-DPO v2 data.

Exp25 upgrades the previous score-channel DPO data into same-schema
reason/score consistency preferences. It is a train-data construction step only:

- no training;
- no dev/test labels are read;
- human rationales never enter the user prompt;
- counterfactual rejected rationales are explicitly marked as counterfactuals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
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
DEFAULT_R7F = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp22_r7_matched_controls_seed42/"
    "data/edubench_r7f_score_reason_consistency_dpo_train.json"
)
DEFAULT_R7G = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp24_orc_dpo_seed42/"
    "data/edubench_r7g_orc_score_channel_reason_aux_train.json"
)
DEFAULT_DEV_JSONL = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST_JSONL = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42")

R7H_SCORE_NAME = "edubench_r7h_score_mismatch_only_train"
R7H_MIXED_NAME = "edubench_r7h_structured_src_dpo_train"
R7H_SCORE_FILE = "data/edubench_r7h_score_mismatch_only_train.json"
R7H_MIXED_FILE = "data/edubench_r7h_structured_src_dpo_train.json"


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


def parse_assistant_payload(message: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(clean(message.get("content")))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def assistant_message(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    }


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


def metric_from_messages(pair: dict[str, Any]) -> str:
    for message in pair.get("messages") or []:
        if message.get("role") != "user":
            continue
        content = clean(message.get("content"))
        marker = "Evaluation metric:\n"
        if marker in content:
            return clean(content.split(marker, 1)[1].split("\n\n", 1)[0])
    return ""


def prompt_has_reason(messages: list[dict[str, Any]], reason: str) -> bool:
    if not reason:
        return False
    reason_short = reason[:80]
    return any(reason_short and reason_short in clean(message.get("content")) for message in messages)


def score_cap_for(label: int) -> int | None:
    return label if label <= 2 else None


def score_mismatch_payload(reason: str, score_cap: int | None, score: int) -> dict[str, Any]:
    return {"reason": reason, "score_cap": score_cap, "score": score}


def low_failure_payload(reason: str, score_cap: int | None, score: int) -> dict[str, Any]:
    return {
        "reason": reason,
        "major_failures": ["recovered_human_failure"],
        "score_cap": score_cap,
        "score": score,
    }


def low_failure_erasure_payload(score: int) -> dict[str, Any]:
    return {
        "reason": "No major failure is identified.",
        "major_failures": [],
        "score_cap": None,
        "score": score,
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


def base_metadata(pair: dict[str, Any], negative_type: str, pair_weight: float) -> dict[str, Any]:
    chosen_payload = parse_assistant_payload(pair["chosen"])
    reason = clean(chosen_payload.get("reason"))
    gold = clamp_score(pair.get("gold_label"))
    rejected = clamp_score(pair.get("rejected_score"))
    source_sample_id = clean(pair.get("source_sample_id"))
    question_key = clean(pair.get("source_question_key") or pair.get("question_key"))
    reason_hash = clean(pair.get("reason_hash")) or sha1(reason)
    risk_type = clean(pair.get("risk_type")) or risk_type_from_scores(gold, rejected)
    return {
        "source_sample_id": source_sample_id,
        "question_key": question_key,
        "gold_label": gold,
        "rejected_score": rejected,
        "risk_type": risk_type,
        "negative_type": negative_type,
        "pair_weight": pair_weight,
        "ordinal_distance": abs(gold - rejected),
        "has_human_reason": bool(reason),
        "reason_hash": reason_hash,
        "reason_in_prompt": prompt_has_reason(pair.get("messages") or [], reason),
        "train_only": True,
        "metric": metric_from_messages(pair),
        "rejected_source": clean(pair.get("rejected_source")),
        "supporting_rejected_sources": clean(pair.get("supporting_rejected_sources")),
        "source_dataset_variant": clean(pair.get("dataset_variant")),
    }


def pair_id(meta: dict[str, Any], suffix: str) -> str:
    key = {
        "source_sample_id": meta["source_sample_id"],
        "question_key": meta["question_key"],
        "gold_label": meta["gold_label"],
        "rejected_score": meta["rejected_score"],
        "negative_type": meta["negative_type"],
        "reason_hash": meta["reason_hash"],
        "suffix": suffix,
    }
    return sha1(key)


def make_score_mismatch_pair(pair: dict[str, Any], negative_type: str, pair_weight: float, suffix: str) -> dict[str, Any]:
    chosen_payload = parse_assistant_payload(pair["chosen"])
    reason = clean(chosen_payload.get("reason"))
    meta = base_metadata(pair, negative_type, pair_weight)
    gold = int(meta["gold_label"])
    rejected = int(meta["rejected_score"])
    score_cap = score_cap_for(gold)
    out = {
        "messages": pair["messages"],
        "chosen": assistant_message(score_mismatch_payload(reason, score_cap, gold)),
        "rejected": assistant_message(score_mismatch_payload(reason, score_cap, rejected)),
        **meta,
        "negative_source": "real_model_score_output_same_human_reason",
    }
    out["pair_id"] = pair_id(meta, suffix)
    return out


def choose_reason_mismatch_source(pair: dict[str, Any], candidates_by_metric: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    metric = metric_from_messages(pair)
    gold = clamp_score(pair.get("gold_label"))
    sid = clean(pair.get("source_sample_id"))
    candidates = [
        item
        for item in candidates_by_metric.get(metric, [])
        if clean(item.get("source_sample_id")) != sid and clamp_score(item.get("gold_label")) != gold
    ]
    if not candidates:
        candidates = [
            item
            for items in candidates_by_metric.values()
            for item in items
            if clean(item.get("source_sample_id")) != sid and clamp_score(item.get("gold_label")) != gold
        ]
    if not candidates:
        return None
    digest = int(sha1(clean(pair.get("source_sample_id")) + clean(pair.get("reason_hash")))[:8], 16)
    return candidates[digest % len(candidates)]


def make_reason_mismatch_pair(pair: dict[str, Any], mismatch_source: dict[str, Any]) -> dict[str, Any]:
    chosen_payload = parse_assistant_payload(pair["chosen"])
    source_payload = parse_assistant_payload(mismatch_source["chosen"])
    reason = clean(chosen_payload.get("reason"))
    mismatch_reason = clean(source_payload.get("reason"))
    meta = base_metadata(pair, "reason_mismatch_same_score", 0.3)
    gold = int(meta["gold_label"])
    cap_i = score_cap_for(gold)
    cap_j = score_cap_for(clamp_score(mismatch_source.get("gold_label")))
    out = {
        "messages": pair["messages"],
        "chosen": assistant_message(score_mismatch_payload(reason, cap_i, gold)),
        "rejected": assistant_message(score_mismatch_payload(mismatch_reason, cap_j, gold)),
        **meta,
        "negative_source": "shuffled_train_reason_same_or_fallback_metric",
        "counterfactual_reason_source_sample_id": clean(mismatch_source.get("source_sample_id")),
        "counterfactual_reason_source_question_key": clean(
            mismatch_source.get("source_question_key") or mismatch_source.get("question_key")
        ),
        "counterfactual_reason_source_gold_label": clamp_score(mismatch_source.get("gold_label")),
        "counterfactual_reason_same_metric": metric_from_messages(pair) == metric_from_messages(mismatch_source),
        "counterfactual_reason_hash": sha1(mismatch_reason),
    }
    out["pair_id"] = pair_id(meta, "reason_mismatch")
    return out


def make_low_failure_erasure_pair(pair: dict[str, Any]) -> dict[str, Any]:
    chosen_payload = parse_assistant_payload(pair["chosen"])
    reason = clean(chosen_payload.get("reason"))
    meta = base_metadata(pair, "low_failure_erasure_counterfactual", 1.5)
    gold = int(meta["gold_label"])
    rejected = int(meta["rejected_score"])
    out = {
        "messages": pair["messages"],
        "chosen": assistant_message(low_failure_payload(reason, gold, gold)),
        "rejected": assistant_message(low_failure_erasure_payload(rejected)),
        **meta,
        "negative_source": "counterfactual_failure_erasure_train_only_not_human",
    }
    out["pair_id"] = pair_id(meta, "low_failure_erasure")
    return out


def build_datasets(r7d: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reason_candidates_by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in r7d:
        if clean(parse_assistant_payload(pair["chosen"]).get("reason")):
            reason_candidates_by_metric[metric_from_messages(pair)].append(pair)

    score_only: list[dict[str, Any]] = []
    mixed: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for pair in r7d:
        reason = clean(parse_assistant_payload(pair["chosen"]).get("reason"))
        if not reason:
            dropped.append(
                {
                    "source_sample_id": pair.get("source_sample_id"),
                    "negative_type": "all",
                    "drop_reason": "missing_human_reason",
                }
            )
            continue
        gold = clamp_score(pair.get("gold_label"))
        rejected = clamp_score(pair.get("rejected_score"))
        score_pair = make_score_mismatch_pair(pair, "score_mismatch_same_reason", 1.0, "score_mismatch")
        score_only.append(score_pair)
        mixed.append(score_pair)

        mismatch_source = choose_reason_mismatch_source(pair, reason_candidates_by_metric)
        if mismatch_source is None:
            dropped.append(
                {
                    "source_sample_id": pair.get("source_sample_id"),
                    "negative_type": "reason_mismatch_same_score",
                    "drop_reason": "no_train_reason_with_different_label",
                }
            )
        else:
            mixed.append(make_reason_mismatch_pair(pair, mismatch_source))

        if gold <= 2 and rejected >= 4:
            mixed.append(make_low_failure_erasure_pair(pair))
        if gold >= 4 and rejected <= 2:
            mixed.append(make_score_mismatch_pair(pair, "high_protection_score_mismatch", 1.0, "high_protection"))
    return score_only, mixed, dropped


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


def response_text(row: dict[str, Any], key: str) -> str:
    return clean((row.get(key) or {}).get("content"))


def build_tables(
    score_rows: list[dict[str, Any]],
    mixed_rows: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    r7d: list[dict[str, Any]],
    r7f: list[dict[str, Any]],
    r7g: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_outputs = [(R7H_SCORE_NAME, score_rows), (R7H_MIXED_NAME, mixed_rows)]
    pair_count_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    for dataset_name, rows in all_outputs:
        pair_count_rows.append(
            {
                "dataset": dataset_name,
                "pair_count": len(rows),
                "unique_pair_id_count": len({row["pair_id"] for row in rows}),
                "unique_source_sample_count": len({row["source_sample_id"] for row in rows}),
                "low_to_high_pair_count": sum(
                    1 for row in rows if int(row["gold_label"]) <= 2 and int(row["rejected_score"]) >= 4
                ),
                "high_to_low_pair_count": sum(
                    1 for row in rows if int(row["gold_label"]) >= 4 and int(row["rejected_score"]) <= 2
                ),
            }
        )
        for (negative_type, weight), count in sorted(Counter((row["negative_type"], row["pair_weight"]) for row in rows).items()):
            negative_rows.append(
                {
                    "dataset": dataset_name,
                    "negative_type": negative_type,
                    "pair_weight": weight,
                    "count": count,
                    "rate": count / max(len(rows), 1),
                }
            )
        for target_key in ("chosen", "rejected"):
            lengths = [len(response_text(row, target_key).split()) for row in rows]
            stats = summarize_numbers(lengths)
            length_rows.append({"dataset": dataset_name, "target": target_key, **stats})

    dev_ids, dev_qkeys, dev_n = split_ids(args.dev_jsonl)
    test_ids, test_qkeys, test_n = split_ids(args.test_jsonl)
    leakage_rows: list[dict[str, Any]] = []
    for dataset_name, rows in all_outputs:
        sample_ids = {row["source_sample_id"] for row in rows}
        question_keys = {row["question_key"] for row in rows}
        leakage_rows.append(
            {
                "dataset": dataset_name,
                "dev_rows_read_for_id_guard": dev_n,
                "test_rows_read_for_id_guard": test_n,
                "dev_sample_overlap": len(sample_ids & dev_ids),
                "dev_question_overlap": len(question_keys & dev_qkeys),
                "test_sample_overlap": len(sample_ids & test_ids),
                "test_question_overlap": len(question_keys & test_qkeys),
                "test_label_read": False,
                "reason_in_prompt_count": sum(1 for row in rows if row.get("reason_in_prompt")),
                "counterfactual_negative_count": sum(
                    1 for row in rows if clean(row.get("negative_source")).startswith("counterfactual")
                ),
            }
        )
    write_csv(args.out_dir / "tables" / "exp25_src_pair_counts.csv", pair_count_rows)
    write_csv(args.out_dir / "tables" / "exp25_src_negative_type_distribution.csv", negative_rows)
    write_csv(args.out_dir / "tables" / "exp25_src_length_stats.csv", length_rows)
    write_csv(args.out_dir / "tables" / "exp25_src_leakage_audit.csv", leakage_rows)
    write_csv(args.out_dir / "tables" / "exp25_src_dropped_pairs.csv", dropped)

    decision = {
        "recommendation": "ready_for_metadata_validation",
        "r7d_source_pairs": len(r7d),
        "r7f_source_pairs": len(r7f),
        "r7g_source_pairs": len(r7g),
        "score_mismatch_only_pairs": len(score_rows),
        "structured_mixed_pairs": len(mixed_rows),
        "negative_type_counts": dict(Counter(row["negative_type"] for row in mixed_rows)),
        "dropped_pair_count": len(dropped),
        "test_label_read": False,
        "human_reason_in_prompt": False,
    }
    write_json(args.out_dir / "decision" / "exp25_src_data_decision.json", decision)
    lines = [
        "# Exp25 Structured SRC-DPO Data Report",
        "",
        "Exp25 builds same-schema reason/score consistency DPO pairs from train-only recovered human reasons.",
        "",
        "## Dataset Counts",
        "",
        f"- R7H score_mismatch_only pairs: {len(score_rows)}",
        f"- R7H structured mixed pairs: {len(mixed_rows)}",
        f"- dropped pairs: {len(dropped)}",
        "",
        "## Negative Types",
        "",
    ]
    for key, count in sorted(Counter(row["negative_type"] for row in mixed_rows).items()):
        lines.append(f"- {key}: {count}")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Human reason is never placed in the user prompt.",
            "- Dev/test are used only for sample_id and question_key leakage guards.",
            "- Test labels are not read.",
            "- Counterfactual rejected reasons are marked as counterfactual train-only negatives.",
        ]
    )
    write_text(args.out_dir / "reports" / "exp25_structured_src_data_report.md", "\n".join(lines))
    return decision


def dataset_info() -> dict[str, Any]:
    base = {
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {"messages": "messages", "chosen": "chosen", "rejected": "rejected"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    return {
        R7H_SCORE_NAME: {**base, "file_name": R7H_SCORE_FILE},
        R7H_MIXED_NAME: {**base, "file_name": R7H_MIXED_FILE},
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    r7d = load_json(args.r7d_json)
    r7f = load_json(args.r7f_json) if args.r7f_json.exists() else []
    r7g = load_json(args.r7g_json) if args.r7g_json.exists() else []
    score_rows, mixed_rows, dropped = build_datasets(r7d)
    write_json_array(args.out_dir / R7H_SCORE_FILE, score_rows)
    write_json_array(args.out_dir / R7H_MIXED_FILE, mixed_rows)
    write_json(args.out_dir / "dataset_info_exp25_src_snippet.json", dataset_info())
    decision = build_tables(score_rows, mixed_rows, dropped, r7d, r7f, r7g, args)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp25 Structured SRC-DPO v2 data.")
    parser.add_argument("--r7d-json", type=Path, default=DEFAULT_R7D)
    parser.add_argument("--r7f-json", type=Path, default=DEFAULT_R7F)
    parser.add_argument("--r7g-json", type=Path, default=DEFAULT_R7G)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_JSONL)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
