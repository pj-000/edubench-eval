"""RAR-0A: deterministic, train-only human-reason alignment audit.

This module has no model, API, GPU, training, dev, or test dependency.  It
uses only exact normalized fields and deliberately provides no fuzzy fallback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import DEFAULT_OUTPUT, DEFAULT_TRAIN, REPO_ROOT


REASON_SOURCES = (
    REPO_ROOT / "5-grades/5_human_1.jsonl",
    REPO_ROOT / "5-grades/5_human_2.jsonl",
    REPO_ROOT / "5-grades/5_human_3.jsonl",
)

METRIC_ALIASES = {
    "BFA": {"Basic Factual Accuracy", "基础事实准确性"},
    "CRSC": {"Content Relevance & Scope Control", "内容相关性与范围控制"},
    "CSI": {"Clarity, Simplicity & Inspiration", "清晰易懂与表达启发"},
    "DKA": {"Domain Knowledge Accuracy", "领域知识准确性", "领域知识专业性"},
    "EICP": {"Error Identification & Correction Precision", "错误识别与纠正精确性"},
    "HOTSD": {"Higher-Order Thinking & Skill Development", "促进高阶思维与能力发展"},
    "IFTC": {"Instruction Following & Task Completion", "指令遵循与任务完成"},
    "MGPF": {
        "Motivation, Guidance & Positive Feedback",
        "鼓励支持与正向反馈",
        "动机引导与正向反馈",
        "激励引导与积极反馈",
    },
    "PALS": {
        "Personalization, Adaptation & Learning Support",
        "个性化适配与学习支持",
        "个性化适应与学习支持",
    },
    "RPR": {"Reasoning Process Rigor", "推理过程严谨性"},
    "RTC": {"Role & Tone Consistency", "角色与语气一致性", "角色与口吻一致性"},
    "SEI": {"Scenario Element Integration", "场景要素融合度", "场景要素整合"},
}

ALIAS_TO_ID = {alias: metric_id for metric_id, aliases in METRIC_ALIASES.items() for alias in aliases}


def normalize(value: Any) -> str:
    """NFKC + whitespace normalization only; no approximate text matching."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    value = unicodedata.normalize("NFKC", value).replace("\\n", "\n")
    return " ".join(value.split()).strip()


def metric_id(value: Any) -> str:
    normalized = normalize(value)
    return ALIAS_TO_ID.get(normalized, normalized)


def reject_eval_path(path: Path) -> None:
    name = path.name.lower()
    normalized = "/" + str(path).replace("\\", "/").lower().strip("/") + "/"
    if name in {"dev.json", "dev.jsonl", "test.json", "test.jsonl"}:
        raise PermissionError(f"RAR-0 is train-only and refuses {path}")
    if "/dev/" in normalized or "/test/" in normalized:
        raise PermissionError(f"RAR-0 is train-only and refuses {path}")


def read_jsonl(path: Path, *, protect_split: bool = False) -> list[dict[str, Any]]:
    if protect_split:
        reject_eval_path(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: Any) -> str:
    return hashlib.sha256(normalize(value).encode("utf-8")).hexdigest()


def alignment_key_from_train(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        normalize(row.get("question")),
        normalize(row.get("answer")),
        metric_id(row.get("metric_id") or row.get("metric_canonical") or row.get("metric_raw")),
        normalize(row.get("generator_model") or row.get("answer_model")),
    )


def alignment_key_from_reason(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        normalize(row.get("question")),
        normalize(row.get("response") or row.get("answer")),
        metric_id(row.get("principle") or row.get("metric_id") or row.get("metric")),
        normalize(row.get("model") or row.get("generator_model") or row.get("answer_model")),
    )


def build_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        index[alignment_key_from_reason(row)].append(row)
    return index


def integer_score(value: Any) -> int | None:
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return None
    return score if score in range(1, 6) else None


def audit_one_source(
    train_row: dict[str, Any], source_rows: list[dict[str, Any]], rater_index: int
) -> dict[str, Any]:
    expected_rater_score = integer_score(train_row.get(f"human_{rater_index}_5"))
    aggregate_score = integer_score(train_row.get("label_5"))
    if not source_rows:
        return {
            "status": "unmatched",
            "candidate_count": 0,
            "aligned_eligible": False,
            "aggregate_score_consistent": False,
            "semantically_qualified": False,
        }
    if len(source_rows) != 1:
        return {
            "status": "ambiguous",
            "candidate_count": len(source_rows),
            "aligned_eligible": False,
            "aggregate_score_consistent": False,
            "semantically_qualified": False,
        }
    source = source_rows[0]
    reason = normalize(source.get("reason"))
    source_score = integer_score(source.get("score"))
    rater_score_consistent = source_score is not None and source_score == expected_rater_score
    aligned = bool(reason) and rater_score_consistent
    aggregate_consistent = aligned and source_score == aggregate_score
    if not reason:
        status = "empty_reason"
    elif not rater_score_consistent:
        status = "rater_score_mismatch"
    elif not aggregate_consistent:
        status = "aligned_rater_only"
    else:
        status = "aligned_aggregate_consistent"
    return {
        "status": status,
        "candidate_count": 1,
        "aligned_eligible": aligned,
        "aggregate_score_consistent": aggregate_consistent,
        "semantically_qualified": False,
        "source_score": source_score,
        "expected_rater_score": expected_rater_score,
        "reason": reason,
        "reason_sha256": text_sha256(reason) if reason else "",
    }


def audit_rows(
    train_rows: list[dict[str, Any]], reason_rows_by_rater: dict[int, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexes = {rater: build_index(rows) for rater, rows in reason_rows_by_rater.items()}
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for row in train_rows:
        key = alignment_key_from_train(row)
        results = {rater: audit_one_source(row, indexes.get(rater, {}).get(key, []), rater) for rater in (1, 2, 3)}
        aligned_count = sum(bool(result["aligned_eligible"]) for result in results.values())
        aggregate_count = sum(bool(result["aggregate_score_consistent"]) for result in results.values())
        record_id = str(row.get("record_id") or row.get("sample_id") or row.get("id") or "")
        public = {
            "record_id": record_id,
            "question_key": str(row.get("question_key") or row.get("question_id") or ""),
            "score": int(row["label_5"]),
            "metric_id": metric_id(row.get("metric_id") or row.get("metric_canonical")),
            "language": normalize(row.get("language")),
            "aligned_reason_count": aligned_count,
            "aggregate_consistent_reason_count": aggregate_count,
            "any_aligned": aligned_count > 0,
            "all_three_aligned": aligned_count == 3,
            "any_direct_rationale_candidate": aggregate_count > 0,
            "semantically_qualified": False,
        }
        for rater, result in results.items():
            public[f"human_{rater}_status"] = result["status"]
            public[f"human_{rater}_candidate_count"] = result["candidate_count"]
            public[f"human_{rater}_reason_sha256"] = result.get("reason_sha256", "")
        public_rows.append(public)
        private_rows.append(
            {
                **public,
                "question": row.get("question"),
                "answer": row.get("answer"),
                "rubric": row.get("rubric"),
                "metric_canonical": row.get("metric_canonical"),
                "generator_model": row.get("generator_model") or row.get("answer_model"),
                "human_reasons": {
                    f"human_{rater}": {
                        key: value
                        for key, value in result.items()
                        if key not in {"semantically_qualified"}
                    }
                    for rater, result in results.items()
                    if result.get("aligned_eligible")
                },
                "field_gates": {
                    "score": 1,
                    "rubric_checks": 0,
                    "evidence": 0,
                    "rationale": 0,
                },
            }
        )
    return public_rows, private_rows


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def coverage_record(rows: list[dict[str, Any]], stratum_type: str, **keys: Any) -> dict[str, Any]:
    n = len(rows)
    any_aligned = sum(bool(row["any_aligned"]) for row in rows)
    all_three = sum(bool(row["all_three_aligned"]) for row in rows)
    direct = sum(bool(row["any_direct_rationale_candidate"]) for row in rows)
    return {
        "stratum_type": stratum_type,
        "score": keys.get("score", ""),
        "metric_id": keys.get("metric_id", ""),
        "language": keys.get("language", ""),
        "n_rows": n,
        "any_aligned_rows": any_aligned,
        "any_aligned_rate": rate(any_aligned, n),
        "all_three_aligned_rows": all_three,
        "all_three_aligned_rate": rate(all_three, n),
        "direct_rationale_candidate_rows": direct,
        "direct_rationale_candidate_rate": rate(direct, n),
        "semantically_qualified_rows": 0,
    }


def coverage_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [coverage_record(rows, "overall")]
    dimensions = [
        ("score", ("score",)),
        ("metric", ("metric_id",)),
        ("language", ("language",)),
        ("score_metric_language", ("score", "metric_id", "language")),
    ]
    for stratum_type, fields in dimensions:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row[field] for field in fields)].append(row)
        for values, items in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
            output.append(coverage_record(items, stratum_type, **dict(zip(fields, values))))
    return output


def source_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rater in (1, 2, 3):
        statuses = Counter(str(row[f"human_{rater}_status"]) for row in rows)
        aligned = sum(status in {"aligned_rater_only", "aligned_aggregate_consistent"} for status in (row[f"human_{rater}_status"] for row in rows))
        aggregate = sum(row[f"human_{rater}_status"] == "aligned_aggregate_consistent" for row in rows)
        output.append(
            {
                "source": f"5_human_{rater}.jsonl",
                "train_rows": len(rows),
                "aligned_eligible_rows": aligned,
                "aligned_eligible_rate": rate(aligned, len(rows)),
                "aggregate_score_consistent_rows": aggregate,
                "aggregate_score_consistent_rate": rate(aggregate, len(rows)),
                "status_counts": json.dumps(dict(sorted(statuses.items())), ensure_ascii=False, sort_keys=True),
                "matching_rule": "exact NFKC-whitespace-normalized question+answer+canonical_metric+generator_model; unique per source",
            }
        )
    return output


def write_report(out_dir: Path, rows: list[dict[str, Any]], hashes: dict[str, str]) -> None:
    overall = coverage_record(rows, "overall")
    statuses = Counter(
        status
        for row in rows
        for status in (row["human_1_status"], row["human_2_status"], row["human_3_status"])
    )
    report = f"""# RAR-0A deterministic alignment audit

## Boundary

- Train rows: `{len(rows)}`
- Dev/test accessed: `false`
- API/model/GPU/training used: `false`
- Matching: exact normalized question + answer + canonical metric + generator model; no fuzzy fallback.
- Semantic qualification: not performed; every rubric/evidence/rationale gate remains 0.

## Result

- Any human reason aligned: `{overall['any_aligned_rows']}/{overall['n_rows']}` = `{overall['any_aligned_rate']:.4f}`
- All three human reasons aligned: `{overall['all_three_aligned_rows']}/{overall['n_rows']}` = `{overall['all_three_aligned_rate']:.4f}`
- At least one aligned reason agrees with aggregate `label_5`: `{overall['direct_rationale_candidate_rows']}/{overall['n_rows']}` = `{overall['direct_rationale_candidate_rate']:.4f}`
- Semantically qualified structured examples: `0` (expected at RAR-0A).

Across rater-source fields, deterministic statuses are:

```json
{json.dumps(dict(sorted(statuses.items())), ensure_ascii=False, indent=2)}
```

## Interpretation

This resolves the earlier contradictory recovery reports for the current paper-like train split.
The three canonical rater files provide exact, score-provenanced reasons for only a subset of train;
unmatched rows must remain score-only unless a separately provenance-checked source is admitted.
Aggregate-score consistency is a direct-rationale candidate gate, not semantic qualification.

The historical Exp53 input excludes the rubric, so it is not the matched S0 for RAR-SFT. A new
rubric-aware score-only run is required before R1/R2/R3 can be compared scientifically.

## Blocking items before structured conversion

1. Human-review and freeze the atomic criterion registry.
2. Record the exact Qwen model snapshot/revision from the server, not only its local path.
3. Freeze a converter schema/prompt and an independent verifier protocol.
4. Define the 120-example train-only audit sample and prepare two blind reviewer packets.
5. Do not train until field-level pass rates and human agreement are known.

## Input hashes

```json
{json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True)}
```
"""
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports/rar0a_alignment_report.md").write_text(report, encoding="utf-8")

    def markdown_rows(stratum_type: str, label_field: str) -> list[str]:
        table = [
            row for row in coverage_table(rows)
            if row["stratum_type"] == stratum_type
        ]
        output = ["| stratum | n | aligned n | aligned rate |", "|---|---:|---:|---:|"]
        for row in table:
            output.append(
                f"| {row[label_field]} | {row['n_rows']} | {row['any_aligned_rows']} | "
                f"{row['any_aligned_rate']:.4f} |"
            )
        return output

    data_card = "\n".join(
        [
            "# RAR-0A alignment data card",
            "",
            "This card describes provenance coverage, not rationale correctness.",
            "",
            "## Score coverage",
            "",
            *markdown_rows("score", "score"),
            "",
            "## Metric coverage",
            "",
            *markdown_rows("metric", "metric_id"),
            "",
            "## Language coverage",
            "",
            *markdown_rows("language", "language"),
            "",
            "## Known selection effects",
            "",
            "- Reason coverage is not missing completely at random: it is higher for labels 1-2 than for label 5.",
            "- The shuffled R2 control must exactly preserve R3's field-coverage mask within score x metric x language.",
            "- R1 must use the same exactly aligned source pool before omitting semantic gates; fuzzy recovery is not an acceptable definition of unfiltered.",
            "- Rows without qualified auxiliary fields stay in every method as score-only rows; they are not deleted.",
            "- The two ambiguous exact keys are excluded from auxiliary supervision pending provenance resolution.",
            "- No reason is semantically qualified at this stage, so evidence exact-span rate and score-reason semantic consistency are not yet reportable.",
            "",
        ]
    )
    (out_dir / "reports/rar0a_alignment_data_card.md").write_text(data_card, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reason-1", type=Path, default=REASON_SOURCES[0])
    parser.add_argument("--reason-2", type=Path, default=REASON_SOURCES[1])
    parser.add_argument("--reason-3", type=Path, default=REASON_SOURCES[2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_eval_path(args.train)
    source_paths = {1: args.reason_1, 2: args.reason_2, 3: args.reason_3}
    for path in [args.train, *source_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    train_rows = read_jsonl(args.train, protect_split=True)
    if len(train_rows) != 2654:
        raise ValueError(f"Expected 2654 paper-like train rows, found {len(train_rows)}")
    reason_rows = {rater: read_jsonl(path) for rater, path in source_paths.items()}
    public_rows, private_rows = audit_rows(train_rows, reason_rows)

    out_dir: Path = args.out_dir
    write_csv(out_dir / "tables/reason_alignment_details.csv", public_rows)
    write_csv(out_dir / "tables/reason_source_summary.csv", source_summary(public_rows))
    write_csv(out_dir / "tables/reason_coverage_by_stratum.csv", coverage_table(public_rows))
    write_jsonl(out_dir / "private/aligned_reason_candidates.jsonl", private_rows)

    hashes = {"train": file_sha256(args.train)}
    hashes.update({f"human_{rater}_reasons": file_sha256(path) for rater, path in source_paths.items()})
    protocol = {
        "experiment": "Exp54 RAR-0A deterministic alignment",
        "split": "paper_like_triple_seed42/train",
        "train_rows": len(train_rows),
        "dev_accessed": False,
        "test_accessed": False,
        "api_used": False,
        "model_used": False,
        "training_used": False,
        "matching_rule": "exact NFKC-whitespace-normalized question+answer+canonical_metric+generator_model; unique per rater source; no fuzzy fallback",
        "source_hashes": hashes,
        "base_model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "base_model_revision": "UNRESOLVED_FROM_SERVER",
        "historical_exp53_has_rubric_input": False,
        "matched_rubric_aware_score_only_required": True,
        "field_gates_after_rar0a": {"score": 1, "rubric_checks": 0, "evidence": 0, "rationale": 0},
    }
    write_json(out_dir / "protocol/rar0a_protocol_lock.json", protocol)
    write_report(out_dir, public_rows, hashes)
    print(json.dumps(coverage_record(public_rows, "overall"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
