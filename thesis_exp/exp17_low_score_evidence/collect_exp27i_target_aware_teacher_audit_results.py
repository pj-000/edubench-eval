"""Collect Exp27I target-aware teacher-audit outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence import collect_exp27g_teacher_audit_361_results as exp27g_collect  # noqa: E402
from thesis_exp.exp17_low_score_evidence.prepare_exp27i_target_aware_teacher_audit_361_packets import (  # noqa: E402
    TARGET_VALUE,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42"
)
PROVIDERS = exp27g_collect.PROVIDERS
STAGES = exp27g_collect.STAGES


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_int(value: Any) -> int | None:
    return exp27g_collect.as_int(value)


def target_fields(parsed: dict[str, Any], stage: str) -> dict[str, Any]:
    key = "blind" if stage == "blind" else "audit"
    obj = parsed.get(key) if isinstance(parsed.get(key), dict) else {}
    return {
        "scored_target": obj.get("scored_target", ""),
        "target_confusion_risk": obj.get("target_confusion_risk", "") if stage == "blind" else "",
        "target_confusion_detected": obj.get("target_confusion_detected", "") if stage == "audit" else "",
        "target_scope_reason": obj.get("target_scope_reason", ""),
    }


def schema_success(row: dict[str, Any]) -> bool:
    return exp27g_collect.schema_success(row)


def target_scope_summary_rows(rows_by_stage: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        for stage in STAGES:
            rows = rows_by_stage[(provider, stage)]
            parsed_rows = [row.get("parsed") for row in rows if isinstance(row.get("parsed"), dict)]
            target_counter: Counter[str] = Counter()
            risk_counter: Counter[str] = Counter()
            confusion_detected_counter: Counter[str] = Counter()
            for parsed in parsed_rows:
                fields = target_fields(parsed, stage)
                target_counter[str(fields["scored_target"])] += 1
                if stage == "blind":
                    risk_counter[str(fields["target_confusion_risk"])] += 1
                else:
                    confusion_detected_counter[str(fields["target_confusion_detected"])] += 1
            out.append(
                {
                    "provider": provider,
                    "stage": stage,
                    "rows": len(rows),
                    "schema_ok": sum(schema_success(row) for row in rows),
                    "expected_target_count": target_counter.get(TARGET_VALUE, 0),
                    "unexpected_target_count": len(parsed_rows) - target_counter.get(TARGET_VALUE, 0),
                    "target_confusion_none": risk_counter.get("none", ""),
                    "target_confusion_possible": risk_counter.get("possible", ""),
                    "target_confusion_high": risk_counter.get("high", ""),
                    "audit_target_confusion_detected_true": confusion_detected_counter.get("True", ""),
                    "audit_target_confusion_detected_false": confusion_detected_counter.get("False", ""),
                }
            )
    return out


def provider_view(
    parsed: dict[str, dict[str, dict[str, Any]]],
    provider: str,
    sid: str,
) -> dict[str, Any]:
    view = exp27g_collect.provider_view(parsed, provider, sid)
    blind = parsed.get(provider, {}).get("blind", {}).get(sid, {})
    audit = parsed.get(provider, {}).get("audit", {}).get(sid, {})
    view.update(
        {
            "scored_target": blind.get("scored_target", ""),
            "target_confusion_risk": blind.get("target_confusion_risk", ""),
            "target_scope_reason": blind.get("target_scope_reason", ""),
            "audit_target_confusion_detected": audit.get("target_confusion_detected", ""),
            "audit_target_scope_reason": audit.get("target_scope_reason", ""),
        }
    )
    return view


def provider_vs_human_rows(
    refs: dict[str, dict[str, Any]],
    parsed: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        pairs: list[tuple[int, int]] = []
        target_ok = 0
        for sid, ref in refs.items():
            original = as_int(ref.get("original_score"))
            view = provider_view(parsed, provider, sid)
            score = view.get("score")
            if view.get("scored_target") == TARGET_VALUE:
                target_ok += 1
            if original is not None and score is not None:
                pairs.append((original, int(score)))
        diffs = [score - original for original, score in pairs]
        rows.append(
            {
                "provider": provider,
                "n": len(pairs),
                "target_ok_count": target_ok,
                "mae_to_original_human": mean(abs(x) for x in diffs) if diffs else "",
                "signed_bias_to_original_human": mean(diffs) if diffs else "",
                "exact_to_original_human": mean(1 if x == 0 else 0 for x in diffs) if diffs else "",
                "adjacent_to_original_human": mean(1 if abs(x) <= 1 else 0 for x in diffs) if diffs else "",
                "low_human_teacher_high_count": sum(1 for original, score in pairs if original <= 2 and score >= 4),
                "high_human_teacher_low_count": sum(1 for original, score in pairs if original >= 4 and score <= 2),
            }
        )
    return rows


def collect(args: argparse.Namespace) -> dict[str, Any]:
    packets = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl")
    }
    refs = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.out_dir / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl")
    }
    rows_by_stage, parsed = exp27g_collect.load_outputs(args.out_dir)
    api_rows = exp27g_collect.api_summary_rows(rows_by_stage)
    target_rows = target_scope_summary_rows(rows_by_stage)
    provider_rows = provider_vs_human_rows(refs, parsed)

    queue: list[dict[str, Any]] = []
    for sid, ref in refs.items():
        original = as_int(ref.get("original_score"))
        q = provider_view(parsed, "qwen", sid)
        d = provider_view(parsed, "deepseek", sid)
        types = exp27g_collect.conflict_types(original, q, d)
        if q.get("target_confusion_risk") in {"possible", "high"}:
            types.append(f"qwen_target_confusion_{q.get('target_confusion_risk')}")
        if d.get("target_confusion_risk") in {"possible", "high"}:
            types.append(f"deepseek_target_confusion_{d.get('target_confusion_risk')}")
        if q.get("audit_target_confusion_detected") is True:
            types.append("qwen_audit_target_confusion_detected")
        if d.get("audit_target_confusion_detected") is True:
            types.append("deepseek_audit_target_confusion_detected")
        types = sorted(set(str(item) for item in types if item))
        if not types:
            continue
        packet = packets.get(sid, {})
        teacher_input = packet.get("teacher_input", {}) if isinstance(packet.get("teacher_input"), dict) else {}
        priority = exp27g_collect.priority_for(original, q, d, types)
        if any("target_confusion" in item for item in types):
            priority = min(priority, 1)
        queue.append(
            {
                "sample_id": sid,
                "pilot_group": ref.get("pilot_group", ""),
                "original_score": original,
                "original_region": exp27g_collect.label_region(original),
                "qwen_score": q.get("score"),
                "deepseek_score": d.get("score"),
                "score_gap": abs(int(q["score"]) - int(d["score"]))
                if q.get("score") is not None and d.get("score") is not None
                else "",
                "qwen_failure_bucket": q.get("failure_bucket"),
                "deepseek_failure_bucket": d.get("failure_bucket"),
                "qwen_derived_risk": q.get("derived_risk"),
                "deepseek_derived_risk": d.get("derived_risk"),
                "qwen_target_confusion_risk": q.get("target_confusion_risk"),
                "deepseek_target_confusion_risk": d.get("target_confusion_risk"),
                "qwen_audit_target_confusion_detected": q.get("audit_target_confusion_detected"),
                "deepseek_audit_target_confusion_detected": d.get("audit_target_confusion_detected"),
                "qwen_training_use": q.get("training_use"),
                "deepseek_training_use": d.get("training_use"),
                "conflict_type": ";".join(types),
                "adjudication_priority": priority,
                "question_preview": exp27g_collect.preview(teacher_input.get("question")),
                "answer_preview": exp27g_collect.preview(teacher_input.get("answer")),
                "rubric_preview": exp27g_collect.preview(teacher_input.get("rubric")),
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
        row["top80_for_codex_direct_review"] = idx <= 80

    conflict_counter: Counter[str] = Counter()
    for row in queue:
        for ctype in str(row["conflict_type"]).split(";"):
            if ctype:
                conflict_counter[ctype] += 1
    conflict_rows = [{"conflict_type": key, "count": value} for key, value in sorted(conflict_counter.items())]

    api_complete = all(row["rows"] == len(packets) and row["schema_ok"] == len(packets) for row in api_rows)
    target_complete = all(row["unexpected_target_count"] == 0 for row in target_rows)
    decision = {
        "packet_rows": len(packets),
        "api_complete": api_complete,
        "target_scope_complete": target_complete,
        "conflict_queue_rows": len(queue),
        "top80_for_codex_direct_review": min(80, len(queue)),
        "recommended_next_step": "codex_direct_semantic_review_then_build_calibrated_361",
        "test_label_read": False,
        "raw_outputs_committed": False,
    }

    out = args.out_dir
    write_csv(out / "tables" / "exp27i_api_summary.csv", api_rows)
    write_csv(out / "tables" / "exp27i_target_scope_summary.csv", target_rows)
    write_csv(out / "tables" / "exp27i_provider_vs_original_human.csv", provider_rows)
    write_csv(out / "tables" / "exp27i_conflict_type_distribution.csv", conflict_rows)
    write_csv(out / "annotation" / "exp27i_human_qwen_deepseek_conflict_queue.csv", queue)
    write_json(out / "decision" / "exp27i_collect_decision.json", decision)

    provider_by_name = {row["provider"]: row for row in provider_rows}
    report = [
        "# Exp27I Target-Aware Teacher-Audit Collection",
        "",
        "This collection summarizes parsed teacher outputs only. Raw API outputs remain ignored and must not be committed.",
        "",
        "## API Completion",
        "",
    ]
    for row in api_rows:
        report.append(
            f"- {row['provider']}/{row['stage']}: rows={row['rows']}, schema_ok={row['schema_ok']}, "
            f"failed={row['parse_or_schema_failed']}"
        )
    report.extend(["", "## Target Scope", ""])
    for row in target_rows:
        report.append(
            f"- {row['provider']}/{row['stage']}: expected_target={row['expected_target_count']}, "
            f"unexpected_target={row['unexpected_target_count']}, "
            f"possible={row['target_confusion_possible']}, high={row['target_confusion_high']}"
        )
    report.extend(["", "## Provider vs Original Human Label", ""])
    for provider in PROVIDERS:
        row = provider_by_name.get(provider, {})
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
            f"- top80_for_codex_direct_review: {min(80, len(queue))}",
            "",
            "Next step: Codex directly reviews the top conflicts by reading the actual context, evaluator output, Qwen output, DeepSeek output, and original human label.",
        ]
    )
    write_text(out / "reports" / "exp27i_collect_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Exp27I target-aware teacher-audit outputs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(collect(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
