#!/usr/bin/env python3
"""Build the private Exp33A source-side reference without touching paper test.

This module resolves the locked Exp28 teacher annotation inputs from their
machine-readable run summaries, verifies them against the Exp28E campaign
decision, and joins source-only labels/reasons to the already selected Exp33A
rows.  The resulting JSONL is private and must never be used as reviewer input.
No API, model inference, training, GPU, or test split is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = Path("thesis_exp/data/splits/paper_like_triple_seed42")
DEFAULT_TEACHER_SUMMARY_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42/decision"
)
DEFAULT_EXP28E_DECISION = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/"
    "decision/exp28e_training_variant_decision.json"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp33_expert_reference/outputs/exp33a_expert_reference_seed42"
)
DEFAULT_HUMAN_REASON_FILES = (
    Path("5-grades/5_human_1.jsonl"),
    Path("5-grades/5_human_2.jsonl"),
    Path("5-grades/5_human_3.jsonl"),
    Path("5-grades/5_merge_human_metric_en.jsonl"),
    Path("5-grades/5_merge_human_metric_zh.jsonl"),
)

TEACHER_SPECS = (
    {
        "role": "primary",
        "provider": "qwen",
        "model": "qwen3.7-max",
        "subset": "all_train",
        "decision_count_key": "primary_teacher_valid_rows",
    },
    {
        "role": "secondary",
        "provider": "deepseek",
        "model": "deepseek-v4-pro",
        "subset": "secondary_route",
        "decision_count_key": "secondary_teacher_valid_rows",
    },
)

METRIC_ALIASES = {
    "Content Relevance & Scope Control": {"内容相关性与范围控制"},
    "Domain Knowledge Accuracy": {"领域知识准确性"},
    "Basic Factual Accuracy": {"基础事实准确性"},
    "Reasoning Process Rigor": {"推理过程严谨性"},
    "Instruction Following & Task Completion": {"指令遵循与任务完成"},
    "Scenario Element Integration": {"场景要素融合度", "场景要素整合"},
    "Personalization, Adaptation & Learning Support": {"个性化适配与学习支持"},
    "Higher-Order Thinking & Skill Development": {"促进高阶思维与能力发展"},
    "Clarity, Simplicity & Inspiration": {"清晰易懂与表达启发"},
    "Role & Tone Consistency": {"角色与语气一致性"},
    "Motivation, Guidance & Positive Feedback": {"鼓励支持与正向反馈", "动机引导与正向反馈", "激励引导与积极反馈"},
    "Error Identification & Correction Precision": {"错误识别与纠正精确性"},
}


def guarded(path: Path) -> Path:
    """Reject the sealed paper test by name before any filesystem read."""
    if path.name.casefold() == "test.jsonl":
        raise PermissionError("Exp33A forbids access to the sealed paper test split")
    return path


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    with guarded(repo_path(path)).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with guarded(repo_path(path)).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with guarded(repo_path(path)).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_teacher_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    all_rows = read_jsonl(path)
    valid = [
        row
        for row in all_rows
        if isinstance(row.get("annotation"), dict) and not (row.get("schema_errors") or [])
    ]
    ids = [str(row.get("sample_id") or "") for row in valid]
    if any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Teacher input has missing or duplicate sample_id: {path}")
    return valid, len(all_rows)


def resolve_teacher_inputs(
    summary_dir: Path = DEFAULT_TEACHER_SUMMARY_DIR,
    exp28e_decision_path: Path = DEFAULT_EXP28E_DECISION,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Resolve one locked annotation file per teacher; never guess a filename."""
    summary_root = guarded(repo_path(summary_dir))
    decision = read_json(exp28e_decision_path)
    if decision.get("teacher_scope") != "qwen_primary_plus_selective_deepseek_only":
        raise ValueError("Exp28E teacher scope is not the locked dual-teacher campaign")
    locked_protocol = str(decision.get("protocol") or "")
    if not locked_protocol:
        raise ValueError("Exp28E decision has no locked teacher protocol")

    summaries: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(summary_root.glob("*.json")):
        payload = read_json(path)
        if all(key in payload for key in ("provider", "model", "subset", "output")):
            summaries.append((path, payload))

    manifest: list[dict[str, Any]] = []
    resolved: dict[str, dict[str, Any]] = {}
    for spec in TEACHER_SPECS:
        candidates = [
            (summary_path, payload)
            for summary_path, payload in summaries
            if str(payload.get("provider")).casefold() == spec["provider"]
            and str(payload.get("model")).casefold() == spec["model"]
            and str(payload.get("subset")).casefold() == spec["subset"]
            and str(payload.get("protocol")).casefold() == locked_protocol.casefold()
        ]
        unique_outputs = {str(payload["output"]): (summary_path, payload) for summary_path, payload in candidates}
        if len(unique_outputs) != 1:
            values = sorted(unique_outputs)
            raise RuntimeError(
                f"Could not uniquely resolve {spec['role']} {spec['model']} {spec['subset']}: {values}"
            )
        output_value, (summary_path, summary) = next(iter(unique_outputs.items()))
        annotation_path = guarded(repo_path(Path(output_value)))
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        rows, row_count = valid_teacher_rows(annotation_path)
        expected = int(decision.get(spec["decision_count_key"], -1))
        if len(rows) != expected:
            raise ValueError(
                f"{spec['role']} valid-row mismatch: actual={len(rows)} locked_exp28e={expected}"
            )
        model_key = f"{spec['role']}_teacher_model"
        if str(decision.get(model_key)).casefold() != spec["model"]:
            raise ValueError(f"Exp28E {model_key} does not match resolved model")
        record = {
            "teacher_role": spec["role"],
            "provider": spec["provider"],
            "model": spec["model"],
            "protocol": str(summary.get("protocol") or ""),
            "subset": spec["subset"],
            "annotation_path": display_path(annotation_path),
            "resolution_summary_path": display_path(summary_path),
            "sha256": sha256_file(annotation_path),
            "row_count": row_count,
            "valid_row_count": len(rows),
            "locked_expected_valid_rows": expected,
            "reference_status": "teacher_model_annotation_not_independent_reference",
        }
        manifest.append(record)
        resolved[spec["role"]] = {
            "path": annotation_path,
            "rows": rows,
            "manifest": record,
        }
    return manifest, resolved


def norm_ws(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", str(value)).strip()


def norm_alnum(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "", norm_ws(value))


def load_human_reason_index(
    paths: Iterable[Path] = DEFAULT_HUMAN_REASON_FILES,
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    dict[tuple[str, str], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    exact: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    alnum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_audit: list[dict[str, Any]] = []
    for path in paths:
        absolute = guarded(repo_path(path))
        if not absolute.is_file():
            raise FileNotFoundError(absolute)
        rows = read_jsonl(absolute)
        source_audit.append(
            {"path": display_path(absolute), "sha256": sha256_file(absolute), "row_count": len(rows)}
        )
        source_name = display_path(absolute)
        for row in rows:
            enriched = dict(row)
            enriched["_source_file"] = source_name
            question = norm_ws(row.get("question"))
            answer = norm_ws(row.get("response"))
            exact[(question, answer)].append(enriched)
            alnum[(question, norm_alnum(answer))].append(enriched)
    return exact, alnum, source_audit


def metric_aliases(metric: str) -> set[str]:
    aliases = {metric}
    aliases.update(METRIC_ALIASES.get(metric, set()))
    return aliases


def human_rater_id(row: dict[str, Any]) -> str:
    match = re.search(r"human[_-]?([123])", norm_ws(row.get("eval")), flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"5_human_([123])", norm_ws(row.get("_source_file")))
    return match.group(1) if match else "other"


def recover_human_reasons(
    source: dict[str, Any],
    exact: dict[tuple[str, str], list[dict[str, Any]]],
    alnum: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    question = norm_ws(source.get("question"))
    answer = norm_ws(source.get("answer"))
    candidates = exact.get((question, answer), [])
    if not candidates:
        candidates = alnum.get((question, norm_alnum(answer)), [])
    aliases = metric_aliases(norm_ws(source.get("metric_canonical") or source.get("metric_raw")))
    metric_rows = [row for row in candidates if norm_ws(row.get("principle")) in aliases]
    merged_rows = [row for row in metric_rows if "5_merge_human_metric" in str(row.get("_source_file"))]
    chosen = merged_rows or metric_rows
    by_rater: dict[str, list[str]] = defaultdict(list)
    for row in chosen:
        reason = norm_ws(row.get("reason"))
        if reason and reason not in by_rater[human_rater_id(row)]:
            by_rater[human_rater_id(row)].append(reason)
    return {
        "human_reason_1": " | ".join(by_rater.get("1", [])) or None,
        "human_reason_2": " | ".join(by_rater.get("2", [])) or None,
        "human_reason_3": " | ".join(by_rater.get("3", [])) or None,
        "human_reason_other": " | ".join(by_rater.get("other", [])) or None,
        "human_reason_match_status": (
            "metric_rationale_recovered" if chosen else "question_answer_matched_metric_unmatched" if candidates else "unmatched"
        ),
        "human_reason_source_files": sorted({str(row.get("_source_file")) for row in chosen}),
    }


def annotation_projection(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "score": None,
            "score_range": None,
            "confidence": None,
            "evidence_flags": [],
            "major_failures": [],
            "score_cap": None,
            "reason": None,
            "rubric_assessment": [],
        }
    annotation = row["annotation"]
    score = int(annotation["score"])
    failures = [str(value) for value in (annotation.get("major_failures") or [])]
    evidence_flags: list[str] = []
    if failures and failures != ["no_major_failure"]:
        evidence_flags.append("major_failure_flagged")
    if annotation.get("score_cap") is not None:
        evidence_flags.append("score_cap_present")
    assessment = annotation.get("rubric_assessment") or []
    if any(isinstance(item, dict) and item.get("met") is False for item in assessment):
        evidence_flags.append("rubric_item_not_met")
    return {
        "score": score,
        "score_range": [score, score],
        "confidence": annotation.get("confidence"),
        "evidence_flags": evidence_flags,
        "major_failures": failures,
        "score_cap": annotation.get("score_cap"),
        "reason": annotation.get("reason"),
        "rubric_assessment": assessment,
    }


def score_direction(original: int, teacher: int | None) -> str | None:
    if teacher is None:
        return None
    if original <= 2 and teacher >= 4:
        return "low_to_high"
    if original >= 4 and teacher <= 2:
        return "high_to_low"
    if teacher > original:
        return "up"
    if teacher < original:
        return "down"
    return "same"


def build_private_source_reference(
    selection_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    teacher_resolved: dict[str, dict[str, Any]],
    reason_paths: Iterable[Path] = DEFAULT_HUMAN_REASON_FILES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        for row in rows:
            sample_id = str(row.get("record_id") or row.get("sample_id") or "")
            if not sample_id or (split, sample_id) in sources:
                raise ValueError(f"Missing or duplicate {split} sample id")
            sources[(split, sample_id)] = row

    primary = {str(row["sample_id"]): row for row in teacher_resolved["primary"]["rows"]}
    secondary = {str(row["sample_id"]): row for row in teacher_resolved["secondary"]["rows"]}
    train_ids = {sample_id for split, sample_id in sources if split == "train"}
    if set(primary) != train_ids:
        raise ValueError(f"Primary teacher coverage differs from train universe: {len(primary)} vs {len(train_ids)}")

    exact, alnum, source_audit = load_human_reason_index(reason_paths)
    output: list[dict[str, Any]] = []
    for selected in selection_rows:
        split = str(selected["source_split"])
        sample_id = str(selected["sample_id"])
        source = sources.get((split, sample_id))
        if source is None:
            raise ValueError(f"Selected sample is outside declared split: {split}/{sample_id}")
        original = int(source["label_5"])
        qwen = annotation_projection(primary.get(sample_id) if split == "train" else None)
        deepseek = annotation_projection(secondary.get(sample_id) if split == "train" else None)
        if split == "dev" and (qwen["score"] is not None or deepseek["score"] is not None):
            raise AssertionError("Clean-dev source join must not contain teacher annotations")
        human_reasons = recover_human_reasons(source, exact, alnum)
        q_direction = score_direction(original, qwen["score"])
        d_direction = score_direction(original, deepseek["score"])
        transition = f"qwen:{q_direction or 'missing'}|deepseek:{d_direction or 'missing'}"
        output.append(
            {
                "sample_id": sample_id,
                "source_split": split,
                "view": selected["view"],
                "question_key": str(source["question_key"]),
                "language": source.get("language"),
                "metric_family": source.get("metric_group"),
                "metric": source.get("metric_canonical"),
                "subject": source.get("subject_canonical"),
                "human_1": source.get("human_1_5"),
                "human_2": source.get("human_2_5"),
                "human_3": source.get("human_3_5"),
                "rounded_human_label": original,
                **human_reasons,
                "qwen_score": qwen["score"],
                "qwen_score_range": qwen["score_range"],
                "qwen_confidence": qwen["confidence"],
                "qwen_evidence_flags": qwen["evidence_flags"],
                "qwen_major_failures": qwen["major_failures"],
                "qwen_score_cap": qwen["score_cap"],
                "qwen_reason": qwen["reason"],
                "qwen_rubric_assessment": qwen["rubric_assessment"],
                "deepseek_score": deepseek["score"],
                "deepseek_score_range": deepseek["score_range"],
                "deepseek_confidence": deepseek["confidence"],
                "deepseek_evidence_flags": deepseek["evidence_flags"],
                "deepseek_major_failures": deepseek["major_failures"],
                "deepseek_score_cap": deepseek["score_cap"],
                "deepseek_reason": deepseek["reason"],
                "deepseek_rubric_assessment": deepseek["rubric_assessment"],
                "teacher_evidence_flags": sorted(set(qwen["evidence_flags"] + deepseek["evidence_flags"])),
                "teacher_direction": {"qwen": q_direction, "deepseek": d_direction},
                "campaign_transition_type": transition,
                "stratum_population": selected.get("stratum_population"),
                "stratum_sample": selected.get("stratum_sample"),
                "inclusion_probability": selected.get("inclusion_probability"),
                "design_weight": selected.get("design_weight"),
                "sampling_risk_reason": selected.get("sampling_risk_reason") or [],
                "reference_status": "source_only_private_not_blind_reviewer_input",
            }
        )
    return output, source_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_OUT_DIR / "private/exp33a_selected_sample_manifest.jsonl")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--teacher-summary-dir", type=Path, default=DEFAULT_TEACHER_SUMMARY_DIR)
    parser.add_argument("--exp28e-decision", type=Path, default=DEFAULT_EXP28E_DECISION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_rows = read_jsonl(args.selection_manifest)
    train_rows = read_jsonl(args.split_dir / "train.jsonl")
    dev_rows = read_jsonl(args.split_dir / "dev.jsonl")
    manifest, resolved = resolve_teacher_inputs(args.teacher_summary_dir, args.exp28e_decision)
    reference, source_audit = build_private_source_reference(selection_rows, train_rows, dev_rows, resolved)
    write_jsonl(args.out_dir / "private/exp33a_source_reference.jsonl", reference)
    write_csv(
        args.out_dir / "tables/exp33a_resolved_teacher_input_manifest.csv",
        manifest,
        [
            "teacher_role", "provider", "model", "protocol", "subset", "annotation_path",
            "resolution_summary_path", "sha256", "row_count", "valid_row_count",
            "locked_expected_valid_rows", "reference_status",
        ],
    )
    print(
        json.dumps(
            {
                "selected_rows": len(reference),
                "teacher_inputs": len(manifest),
                "human_reason_sources": len(source_audit),
                "test_access_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
