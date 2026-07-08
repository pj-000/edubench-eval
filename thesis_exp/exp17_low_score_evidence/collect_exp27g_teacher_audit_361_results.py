"""Collect Exp27G 361-case teacher-audit outputs into lightweight summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.collect_exp27d_teacher_audit_results import (  # noqa: E402
    derived_overestimation_risk,
    failure_bucket,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27g_teacher_audited_361_seed42")
PROVIDERS = ("qwen", "deepseek")
STAGES = ("blind", "audit")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def label_region(score: int | None) -> str:
    if score is None:
        return "unknown"
    if score <= 2:
        return "low"
    if score == 3:
        return "mid"
    return "high"


def preview(value: Any, n: int = 220) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def schema_success(row: dict[str, Any]) -> bool:
    return bool(row.get("parsed")) and not row.get("parse_error") and not row.get("schema_errors")


def parsed_by_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        obj = parsed.get(key) if isinstance(parsed.get(key), dict) else {}
        sid = str(row.get("sample_id") or parsed.get("sample_id") or "")
        if sid and obj:
            out[sid] = obj
    return out


def load_outputs(out_dir: Path) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    rows_by_stage: dict[tuple[str, str], list[dict[str, Any]]] = {}
    parsed: dict[str, dict[str, dict[str, Any]]] = {}
    for provider in PROVIDERS:
        parsed[provider] = {}
        for stage in STAGES:
            path = out_dir / "annotations" / "parsed" / provider / f"exp27d_{stage}_outputs.jsonl"
            rows = read_jsonl(path)
            rows_by_stage[(provider, stage)] = rows
            key = "blind" if stage == "blind" else "audit"
            parsed[provider][stage] = parsed_by_id(rows, key)
    return rows_by_stage, parsed


def provider_view(parsed: dict[str, dict[str, dict[str, Any]]], provider: str, sid: str) -> dict[str, Any]:
    blind = parsed.get(provider, {}).get("blind", {}).get(sid, {})
    audit = parsed.get(provider, {}).get("audit", {}).get(sid, {})
    score = as_int(blind.get("teacher_score"))
    return {
        "score": score,
        "failure_bucket": failure_bucket(blind) if blind else "missing",
        "derived_risk": derived_overestimation_risk(blind) if blind else "missing",
        "score_cap": as_int(blind.get("score_cap")),
        "failure_visibility": blind.get("failure_visibility", ""),
        "major_failures": blind.get("major_failures", []),
        "training_use": audit.get("recommended_training_use", "missing") if audit else "missing",
        "hard_conflict": bool(audit.get("hard_conflict")) if audit else False,
        "label_quality": audit.get("label_quality", "") if audit else "",
        "label_noise_type": audit.get("label_noise_type", "") if audit else "",
    }


def conflict_types(original_score: int | None, q: dict[str, Any], d: dict[str, Any]) -> list[str]:
    types: list[str] = []
    q_score = q.get("score")
    d_score = d.get("score")
    if q_score is None or d_score is None:
        types.append("missing_teacher_score")
        return types
    gap = abs(int(q_score) - int(d_score))
    if gap >= 2:
        types.append("score_gap_ge2")
    if d_score >= q_score + 2:
        types.append("deepseek_lenient_qwen_strict")
    if q_score >= d_score + 2:
        types.append("qwen_lenient_deepseek_strict")
    if original_score is not None:
        if original_score <= 2 and (q_score >= 4 or d_score >= 4):
            types.append("low_human_teacher_high")
        if original_score >= 4 and (q_score <= 2 or d_score <= 2):
            types.append("high_human_teacher_low")
        if abs(q_score - original_score) >= 2 and abs(d_score - original_score) >= 2:
            types.append("both_teachers_disagree_with_human")
        elif abs(q_score - original_score) >= 2 or abs(d_score - original_score) >= 2:
            types.append("one_teacher_disagrees_with_human")
    if q.get("failure_bucket") != d.get("failure_bucket"):
        types.append("failure_bucket_disagreement")
    if q.get("derived_risk") != d.get("derived_risk"):
        types.append("derived_risk_disagreement")
    if q.get("training_use") != d.get("training_use"):
        types.append("training_use_disagreement")
    if q.get("hard_conflict") or d.get("hard_conflict"):
        types.append("audit_hard_conflict")
    return sorted(set(types))


def priority_for(original_score: int | None, q: dict[str, Any], d: dict[str, Any], types: list[str]) -> int:
    if "low_human_teacher_high" in types:
        return 1
    if "high_human_teacher_low" in types:
        return 2
    q_score = q.get("score")
    d_score = d.get("score")
    if q_score is not None and d_score is not None and abs(q_score - d_score) >= 3:
        return 3
    if "both_teachers_disagree_with_human" in types:
        return 4
    if "score_gap_ge2" in types:
        return 5
    if "failure_bucket_disagreement" in types or "derived_risk_disagreement" in types:
        return 6
    return 7


def provider_vs_human_rows(refs: dict[str, dict[str, Any]], parsed: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        pairs: list[tuple[int, int]] = []
        for sid, ref in refs.items():
            original = as_int(ref.get("original_score"))
            score = provider_view(parsed, provider, sid).get("score")
            if original is not None and score is not None:
                pairs.append((original, int(score)))
        diffs = [score - original for original, score in pairs]
        rows.append(
            {
                "provider": provider,
                "n": len(pairs),
                "mae_to_original_human": mean(abs(x) for x in diffs) if diffs else "",
                "signed_bias_to_original_human": mean(diffs) if diffs else "",
                "exact_to_original_human": mean(1 if x == 0 else 0 for x in diffs) if diffs else "",
                "adjacent_to_original_human": mean(1 if abs(x) <= 1 else 0 for x in diffs) if diffs else "",
                "low_human_teacher_high_count": sum(1 for original, score in pairs if original <= 2 and score >= 4),
                "high_human_teacher_low_count": sum(1 for original, score in pairs if original >= 4 and score <= 2),
            }
        )
    return rows


def api_summary_rows(rows_by_stage: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        for stage in STAGES:
            rows = rows_by_stage[(provider, stage)]
            out.append(
                {
                    "provider": provider,
                    "stage": stage,
                    "rows": len(rows),
                    "schema_ok": sum(schema_success(row) for row in rows),
                    "parse_or_schema_failed": sum(not schema_success(row) for row in rows),
                    "schema_repair_attempt_rows": sum(int(row.get("schema_repair_attempts") or 0) > 0 for row in rows),
                    "repair_changed_judgement_rows": sum(
                        bool(row.get("repair_changed_teacher_score"))
                        or bool(row.get("repair_changed_major_failures"))
                        or bool(row.get("repair_changed_score_cap"))
                        or bool(row.get("repair_changed_failure_visibility"))
                        for row in rows
                    ),
                }
            )
    return out


def collect(args: argparse.Namespace) -> dict[str, Any]:
    packets = {str(row["sample_id"]): row for row in read_jsonl(args.out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl")}
    refs = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.out_dir / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl")
    }
    rows_by_stage, parsed = load_outputs(args.out_dir)
    api_rows = api_summary_rows(rows_by_stage)
    provider_rows = provider_vs_human_rows(refs, parsed)

    queue: list[dict[str, Any]] = []
    for sid, ref in refs.items():
        original = as_int(ref.get("original_score"))
        q = provider_view(parsed, "qwen", sid)
        d = provider_view(parsed, "deepseek", sid)
        types = conflict_types(original, q, d)
        if not types:
            continue
        packet = packets.get(sid, {})
        teacher_input = packet.get("teacher_input", {}) if isinstance(packet.get("teacher_input"), dict) else {}
        priority = priority_for(original, q, d, types)
        queue.append(
            {
                "sample_id": sid,
                "pilot_group": ref.get("pilot_group", ""),
                "original_score": original,
                "original_region": label_region(original),
                "qwen_score": q.get("score"),
                "deepseek_score": d.get("score"),
                "score_gap": abs(int(q["score"]) - int(d["score"]))
                if q.get("score") is not None and d.get("score") is not None
                else "",
                "qwen_failure_bucket": q.get("failure_bucket"),
                "deepseek_failure_bucket": d.get("failure_bucket"),
                "qwen_derived_risk": q.get("derived_risk"),
                "deepseek_derived_risk": d.get("derived_risk"),
                "qwen_training_use": q.get("training_use"),
                "deepseek_training_use": d.get("training_use"),
                "conflict_type": ";".join(types),
                "adjudication_priority": priority,
                "question_preview": preview(teacher_input.get("question")),
                "answer_preview": preview(teacher_input.get("answer")),
                "rubric_preview": preview(teacher_input.get("rubric")),
            }
        )
    queue.sort(
        key=lambda row: (
            int(row["adjudication_priority"]),
            -int(row["score_gap"] or 0),
            row["sample_id"],
        )
    )
    for idx, row in enumerate(queue, start=1):
        row["top80_for_adjudication"] = idx <= 80

    conflict_counter: Counter[str] = Counter()
    for row in queue:
        for ctype in str(row["conflict_type"]).split(";"):
            if ctype:
                conflict_counter[ctype] += 1
    conflict_rows = [{"conflict_type": key, "count": value} for key, value in sorted(conflict_counter.items())]

    decision = {
        "packet_rows": len(packets),
        "api_complete": all(row["rows"] == len(packets) and row["schema_ok"] == len(packets) for row in api_rows),
        "conflict_queue_rows": len(queue),
        "top80_for_adjudication": min(80, len(queue)),
        "recommend_manual_or_gpt55_adjudication": len(queue) > 0,
        "recommended_next_step": "adjudicate_top80_then_build_teacher_audited_train_labels",
        "test_label_read": False,
        "raw_outputs_committed": False,
    }

    out = args.out_dir
    write_csv(out / "tables" / "exp27g_api_summary.csv", api_rows)
    write_csv(out / "tables" / "exp27g_provider_vs_original_human.csv", provider_rows)
    write_csv(out / "tables" / "exp27g_conflict_type_distribution.csv", conflict_rows)
    write_csv(out / "annotation" / "exp27g_human_qwen_deepseek_adjudication_queue.csv", queue)
    write_json(out / "decision" / "exp27g_collect_decision.json", decision)

    q_provider = {row["provider"]: row for row in provider_rows}
    report = [
        "# Exp27G 361 Teacher-Audit Collection",
        "",
        "This collection summarizes parsed teacher outputs only. Raw API outputs remain ignored and are not intended for commit.",
        "",
        "## API Completion",
        "",
    ]
    for row in api_rows:
        report.append(
            f"- {row['provider']}/{row['stage']}: rows={row['rows']}, schema_ok={row['schema_ok']}, "
            f"failed={row['parse_or_schema_failed']}"
        )
    report.extend(
        [
            "",
            "## Provider vs Original Human Label",
            "",
        ]
    )
    for provider in PROVIDERS:
        row = q_provider.get(provider, {})
        report.append(
            f"- {provider}: MAE={row.get('mae_to_original_human')}, bias={row.get('signed_bias_to_original_human')}, "
            f"low-human teacher-high={row.get('low_human_teacher_high_count')}, "
            f"high-human teacher-low={row.get('high_human_teacher_low_count')}"
        )
    report.extend(
        [
            "",
            "## Conflict Queue",
            "",
            f"- conflict rows: {len(queue)}",
            f"- top80_for_adjudication: {min(80, len(queue))}",
            "",
            "Next step: adjudicate the top conflict queue by comparing original human label, Qwen label, and DeepSeek label.",
        ]
    )
    write_text(out / "reports" / "exp27g_collect_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp27G 361 teacher-audit outputs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
