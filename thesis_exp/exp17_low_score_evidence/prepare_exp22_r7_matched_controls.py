"""Prepare Exp22 matched R7 controls and score-reason consistency negatives.

Exp22 is a train-data preparation step only:

- R7E is a strict score-only matched control for R7D. It uses the exact same
  prompts, source samples, rejected scores, and risk types as R7D, but removes
  the human rationale from the chosen response.
- R7F is a score-reason consistency DPO dataset. It keeps the R7D chosen
  response and constructs train-only counterfactual negatives where either the
  score conflicts with the same human rationale or a different train rationale
  is paired with the gold score.

No model is trained here. Dev/test files are used only for sample_id and
question_key leakage checks.
"""

from __future__ import annotations

import argparse
import csv
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

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    clean,
    clamp_score,
    dpo_dataset_entry,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_R7D = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42/"
    "data/edubench_r7d_strict_label_consistent_reason_real_dpo_train.json"
)
DEFAULT_R7A = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42/"
    "data/edubench_r7a_score_only_reason_covered_real_dpo_train.json"
)
DEFAULT_MANIFEST = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42/"
    "review/r7_reason_recovered_dpo_review_manifest.csv"
)
DEFAULT_PAIR_COUNTS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42/"
    "tables/r7_pair_counts.csv"
)
DEFAULT_DEV_JSONL = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST_JSONL = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp22_r7_matched_controls_seed42")

R7E_NAME = "edubench_r7e_matched_score_only_strict_real_dpo_train"
R7F_NAME = "edubench_r7f_score_reason_consistency_dpo_train"
R7E_FILE = "data/edubench_r7e_matched_score_only_strict_real_dpo_train.json"
R7F_FILE = "data/edubench_r7f_score_reason_consistency_dpo_train.json"


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


def assistant_message(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    }


def parse_target(message: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(clean(message.get("content")))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def score_payload(score: Any) -> dict[str, int]:
    return {"score": clamp_score(score)}


def pair_key(pair: dict[str, Any]) -> tuple[str, str, str, int, int, str]:
    return (
        clean(pair.get("source_sample_id")),
        clean(pair.get("source_question_key")),
        clean(pair.get("risk_type")),
        clamp_score(pair.get("gold_label")),
        clamp_score(pair.get("rejected_score")),
        clean(pair.get("reason_hash")),
    )


def prompt_text(pair: dict[str, Any]) -> str:
    return json.dumps(pair.get("messages") or [], ensure_ascii=False, sort_keys=True)


def metric_from_messages(pair: dict[str, Any]) -> str:
    for message in pair.get("messages") or []:
        if message.get("role") != "user":
            continue
        content = clean(message.get("content"))
        marker = "Evaluation metric:\n"
        if marker in content:
            after = content.split(marker, 1)[1]
            return clean(after.split("\n\n", 1)[0])
    return ""


def read_split_ids(path: Path) -> tuple[set[str], set[str], int]:
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


def make_r7e_pair(pair: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(pair)
    out["dataset_variant"] = "r7e_matched_score_only_strict"
    out["chosen_schema"] = "score_only"
    out["matched_from_dataset"] = "r7d_strict_label_consistent_reason"
    out["matched_pair_index"] = index
    out["chosen"] = assistant_message(score_payload(pair.get("gold_label")))
    out["rejected"] = assistant_message(score_payload(pair.get("rejected_score")))
    return out


def base_consistency_pair(pair: dict[str, Any], negative_type: str, rejected_payload: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(pair)
    out["dataset_variant"] = "r7f_score_reason_consistency"
    out["chosen_schema"] = "human_reason_score"
    out["negative_type"] = negative_type
    out["reason_source"] = "human_recovered_train" if negative_type == "score_mismatch" else "counterfactual_train_only"
    out["matched_from_dataset"] = "r7d_strict_label_consistent_reason"
    out["matched_pair_index"] = index
    out["chosen"] = pair["chosen"]
    out["rejected"] = assistant_message(rejected_payload)
    return out


def choose_reason_mismatch_source(
    pair: dict[str, Any],
    candidates_by_metric: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    metric = metric_from_messages(pair)
    gold = clamp_score(pair.get("gold_label"))
    sid = clean(pair.get("source_sample_id"))
    candidates = [
        item
        for item in candidates_by_metric.get(metric, [])
        if clean(item.get("source_sample_id")) != sid and clamp_score(item.get("gold_label")) != gold
    ]
    if not candidates:
        return None
    digest = int(sha1(pair_key(pair))[:8], 16)
    return candidates[digest % len(candidates)]


def make_r7f_pairs(r7d: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates_by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in r7d:
        chosen = parse_target(pair["chosen"])
        if clean(chosen.get("reason")):
            candidates_by_metric[metric_from_messages(pair)].append(pair)
    out: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for idx, pair in enumerate(r7d):
        chosen = parse_target(pair["chosen"])
        reason = clean(chosen.get("reason"))
        if not reason:
            dropped.append(
                {
                    "source_sample_id": pair.get("source_sample_id"),
                    "negative_type": "score_mismatch",
                    "drop_reason": "missing_chosen_reason",
                }
            )
            continue
        score_mismatch_rejected = {
            "reason": reason,
            "score": clamp_score(pair.get("rejected_score")),
        }
        out.append(base_consistency_pair(pair, "score_mismatch", score_mismatch_rejected, idx))

        mismatch_source = choose_reason_mismatch_source(pair, candidates_by_metric)
        if mismatch_source is None:
            dropped.append(
                {
                    "source_sample_id": pair.get("source_sample_id"),
                    "negative_type": "reason_mismatch",
                    "drop_reason": "no_same_metric_train_reason_with_different_label",
                }
            )
            continue
        source_chosen = parse_target(mismatch_source["chosen"])
        mismatch_reason = clean(source_chosen.get("reason"))
        if not mismatch_reason:
            dropped.append(
                {
                    "source_sample_id": pair.get("source_sample_id"),
                    "negative_type": "reason_mismatch",
                    "drop_reason": "selected_reason_empty",
                }
            )
            continue
        reason_mismatch_rejected = {
            "reason": mismatch_reason,
            "score": clamp_score(pair.get("gold_label")),
        }
        record = base_consistency_pair(pair, "reason_mismatch", reason_mismatch_rejected, idx)
        record["counterfactual_reason_source_sample_id"] = clean(mismatch_source.get("source_sample_id"))
        record["counterfactual_reason_source_question_key"] = clean(mismatch_source.get("source_question_key"))
        record["counterfactual_reason_source_gold_label"] = clamp_score(mismatch_source.get("gold_label"))
        record["counterfactual_reason_source_metric"] = metric_from_messages(mismatch_source)
        record["counterfactual_reason_same_metric"] = metric_from_messages(pair) == metric_from_messages(mismatch_source)
        record["counterfactual_reason_hash"] = sha1(mismatch_reason)
        out.append(record)
    return out, dropped


def alignment_rows(r7d: list[dict[str, Any]], r7a: list[dict[str, Any]], r7e: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r7a_by_key = {pair_key(pair): pair for pair in r7a}
    rows: list[dict[str, Any]] = []
    for idx, (source, matched) in enumerate(zip(r7d, r7e)):
        key = pair_key(source)
        r7a_pair = r7a_by_key.get(key)
        source_chosen = parse_target(source["chosen"])
        matched_chosen = parse_target(matched["chosen"])
        rows.append(
            {
                "pair_index": idx,
                "sample_id": source.get("source_sample_id"),
                "question_key": source.get("source_question_key"),
                "risk_type": source.get("risk_type"),
                "gold_label": source.get("gold_label"),
                "rejected_score": source.get("rejected_score"),
                "reason_hash": source.get("reason_hash"),
                "r7a_match_found": r7a_pair is not None,
                "r7e_same_sample": pair_key(source) == pair_key(matched),
                "r7e_messages_match_r7d": sha1(source.get("messages")) == sha1(matched.get("messages")),
                "r7e_rejected_match_r7d": parse_target(source["rejected"]) == parse_target(matched["rejected"]),
                "r7e_chosen_score_only": matched_chosen == score_payload(source.get("gold_label")),
                "r7d_chosen_has_reason": bool(clean(source_chosen.get("reason"))),
                "r7a_messages_match_r7d": bool(r7a_pair and sha1(source.get("messages")) == sha1(r7a_pair.get("messages"))),
                "r7a_rejected_match_r7d": bool(r7a_pair and parse_target(source["rejected"]) == parse_target(r7a_pair["rejected"])),
            }
        )
    return rows


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    return sorted_values[lower] * (upper - pos) + sorted_values[upper] * (pos - lower)


def length_stats(dataset_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clean(row.get("negative_type") or row.get("risk_type") or "all")].append(row)
    out: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items()):
        chosen_lengths = [len(clean(item["chosen"].get("content"))) for item in items]
        rejected_lengths = [len(clean(item["rejected"].get("content"))) for item in items]
        gaps = [c - r for c, r in zip(chosen_lengths, rejected_lengths)]
        out.append(
            {
                "dataset_name": dataset_name,
                "group": group,
                "n": len(items),
                "chosen_chars_mean": f"{statistics.mean(chosen_lengths):.2f}",
                "chosen_chars_p50": f"{quantile([float(x) for x in chosen_lengths], 0.50):.2f}",
                "chosen_chars_p95": f"{quantile([float(x) for x in chosen_lengths], 0.95):.2f}",
                "rejected_chars_mean": f"{statistics.mean(rejected_lengths):.2f}",
                "rejected_chars_p50": f"{quantile([float(x) for x in rejected_lengths], 0.50):.2f}",
                "rejected_chars_p95": f"{quantile([float(x) for x in rejected_lengths], 0.95):.2f}",
                "chosen_minus_rejected_chars_mean": f"{statistics.mean(gaps):.2f}",
            }
        )
    return out


def leakage_audit(
    datasets: dict[str, list[dict[str, Any]]],
    dev_ids: set[str],
    dev_qkeys: set[str],
    test_ids: set[str],
    test_qkeys: set[str],
    dev_n: int,
    test_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, pairs in datasets.items():
        sample_ids = {clean(pair.get("source_sample_id")) for pair in pairs}
        question_keys = {clean(pair.get("source_question_key")) for pair in pairs}
        reason_prompt_hits = 0
        for pair in pairs:
            prompt = prompt_text(pair)
            for side in ["chosen", "rejected"]:
                reason = clean(parse_target(pair[side]).get("reason"))
                if reason and reason in prompt:
                    reason_prompt_hits += 1
        rows.append(
            {
                "dataset_name": name,
                "pairs": len(pairs),
                "dev_rows_read_for_id_guard_only": dev_n,
                "test_rows_read_for_id_guard_only": test_n,
                "dev_sample_id_overlap": len(sample_ids & dev_ids),
                "dev_question_key_overlap": len(question_keys & dev_qkeys),
                "test_sample_id_overlap": len(sample_ids & test_ids),
                "test_question_key_overlap": len(question_keys & test_qkeys),
                "human_reason_in_prompt_count": reason_prompt_hits,
                "dev_test_label_read": False,
                "leakage_pass": not any(
                    [
                        sample_ids & dev_ids,
                        question_keys & dev_qkeys,
                        sample_ids & test_ids,
                        question_keys & test_qkeys,
                        reason_prompt_hits,
                    ]
                ),
            }
        )
    return rows


def pair_counts(dataset_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: Counter[tuple[str, str]] = Counter()
    for row in rows:
        by_key[(clean(row.get("risk_type")), clean(row.get("negative_type") or "real_score_rejected"))] += 1
    return [
        {
            "dataset_name": dataset_name,
            "risk_type": risk_type,
            "negative_type": negative_type,
            "pairs": count,
        }
        for (risk_type, negative_type), count in sorted(by_key.items())
    ]


def dataset_info() -> dict[str, Any]:
    return {
        R7E_NAME: dpo_dataset_entry(R7E_FILE),
        R7F_NAME: dpo_dataset_entry(R7F_FILE),
    }


def write_report(
    out_dir: Path,
    r7d: list[dict[str, Any]],
    r7e: list[dict[str, Any]],
    r7f: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
) -> None:
    r7e_aligned = all(
        row["r7e_same_sample"]
        and row["r7e_messages_match_r7d"]
        and row["r7e_rejected_match_r7d"]
        and row["r7e_chosen_score_only"]
        for row in alignment
    )
    r7a_missing = sum(1 for row in alignment if not row["r7a_match_found"])
    negative_counts = Counter(clean(row.get("negative_type") or "real_score_rejected") for row in r7f)
    lines = [
        "# Exp22 R7 Matched Controls",
        "",
        "Exp22 prepares train-only data for the next DPO experiments. It does not train a model.",
        "",
        "## Outputs",
        "",
        f"- R7D source pairs: `{len(r7d)}`",
        f"- R7E matched score-only pairs: `{len(r7e)}`",
        f"- R7F consistency pairs: `{len(r7f)}`",
        f"- R7F dropped candidates: `{len(dropped)}`",
        "",
        "## Alignment",
        "",
        f"- R7E exactly aligned with R7D: `{r7e_aligned}`",
        f"- R7D pairs not found in broad R7A pool: `{r7a_missing}`",
        "",
        "## R7F Negative Types",
        "",
    ]
    for key, value in sorted(negative_counts.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Leakage Audit", "", "| dataset | pairs | pass | reason in prompt | dev overlap | test overlap |", "|---|---:|---|---:|---:|---:|"])
    for row in leakage_rows:
        lines.append(
            f"| `{row['dataset_name']}` | {row['pairs']} | `{row['leakage_pass']}` | "
            f"{row['human_reason_in_prompt_count']} | {row['dev_sample_id_overlap'] + row['dev_question_key_overlap']} | "
            f"{row['test_sample_id_overlap'] + row['test_question_key_overlap']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- R7E is the strict score-only control required before comparing R7D against score-only DPO.",
            "- R7F provides score-mismatch and reason-mismatch counterfactuals for SRC-DPO.",
            "- Reason text is only present in assistant targets, never in user prompts.",
            "- Dev/test splits are used only for ID and question-key leakage checks.",
        ]
    )
    write_text(out_dir / "reports" / "exp22_r7_matched_controls_report.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp22 R7 matched controls.")
    parser.add_argument("--r7d", type=Path, default=DEFAULT_R7D)
    parser.add_argument("--r7a", type=Path, default=DEFAULT_R7A)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pair-counts", type=Path, default=DEFAULT_PAIR_COUNTS)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_JSONL)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    r7d = load_json(args.r7d)
    r7a = load_json(args.r7a)
    if not isinstance(r7d, list) or not isinstance(r7a, list):
        raise TypeError("R7D and R7A must be JSON arrays.")
    if not args.manifest.exists():
        raise FileNotFoundError(f"Missing R7 manifest: {args.manifest}")
    if not args.pair_counts.exists():
        raise FileNotFoundError(f"Missing R7 pair count table: {args.pair_counts}")

    r7e = [make_r7e_pair(pair, idx) for idx, pair in enumerate(r7d)]
    r7f, dropped = make_r7f_pairs(r7d)
    alignment = alignment_rows(r7d, r7a, r7e)

    dev_ids, dev_qkeys, dev_n = read_split_ids(args.dev_jsonl)
    test_ids, test_qkeys, test_n = read_split_ids(args.test_jsonl)
    datasets = {R7E_NAME: r7e, R7F_NAME: r7f}
    leakage_rows = leakage_audit(datasets, dev_ids, dev_qkeys, test_ids, test_qkeys, dev_n, test_n)
    if any(not row["leakage_pass"] for row in leakage_rows):
        raise RuntimeError(f"Exp22 leakage audit failed: {leakage_rows}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_array(args.out_dir / R7E_FILE, r7e)
    write_json_array(args.out_dir / R7F_FILE, r7f)
    write_json(args.out_dir / "dataset_info.json", dataset_info())
    write_json(args.out_dir / "dataset_info_exp22_snippet.json", dataset_info())

    write_csv(args.out_dir / "tables" / "exp22_pair_alignment.csv", alignment)
    length_rows = length_stats(R7E_NAME, r7e) + length_stats(R7F_NAME, r7f)
    write_csv(args.out_dir / "tables" / "exp22_length_stats.csv", length_rows)
    write_csv(args.out_dir / "tables" / "exp22_leakage_audit.csv", leakage_rows)
    write_csv(args.out_dir / "tables" / "exp22_pair_counts.csv", pair_counts(R7E_NAME, r7e) + pair_counts(R7F_NAME, r7f))
    write_csv(args.out_dir / "tables" / "exp22_dropped_candidates.csv", dropped)
    write_report(args.out_dir, r7d, r7e, r7f, dropped, leakage_rows, alignment)

    decision = {
        "safe_for_dpo_data_review": True,
        "r7e_pair_count": len(r7e),
        "r7f_pair_count": len(r7f),
        "r7f_negative_type_counts": dict(Counter(clean(row.get("negative_type")) for row in r7f)),
        "r7e_exactly_aligned_with_r7d": all(
            row["r7e_same_sample"]
            and row["r7e_messages_match_r7d"]
            and row["r7e_rejected_match_r7d"]
            and row["r7e_chosen_score_only"]
            for row in alignment
        ),
        "r7a_missing_alignment_count": sum(1 for row in alignment if not row["r7a_match_found"]),
        "leakage_audit_pass": all(bool(row["leakage_pass"]) for row in leakage_rows),
        "human_reason_in_prompt_count": sum(int(row["human_reason_in_prompt_count"]) for row in leakage_rows),
        "recommended_next_experiment": "Exp23 R7D vs R7E ordinary DPO scout",
        "guardrails": {
            "no_training": True,
            "gpu_required": False,
            "dev_test_ids_used_for_leakage_guard_only": True,
            "dev_test_labels_used": False,
            "human_reason_not_in_user_prompt": True,
        },
    }
    write_json(args.out_dir / "decision" / "exp22_r7_matched_controls_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
