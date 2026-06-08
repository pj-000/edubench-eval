"""Normalize Exp6-3 mini-batch raw generations into candidate JSONL."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import MINI_BATCH_PROMPT_VERSION, ensure_mini_batch_dirs
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import (
    load_train_by_source_id,
    make_synthetic_id,
    mini_generated_path,
    mini_prompt_path,
    read_jsonl_if_exists,
    read_mini_table,
    write_empty_jsonl,
    write_mini_jsonl,
)
from thesis_exp.src.edujudge.utils.text_norm import stringify


NORMALIZED_PATH = mini_generated_path("normalized_synthetic_candidates.jsonl")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def parse_raw_response(text: str) -> tuple[dict[str, Any], str]:
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, "parsed_json"
        return {}, "json_not_object"
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed, "parsed_json_substring"
            except json.JSONDecodeError:
                pass
        return {}, "invalid_json"


def load_prompts_by_plan() -> dict[str, dict[str, Any]]:
    return {row.get("synthetic_plan_id", ""): row for row in read_jsonl_if_exists(mini_prompt_path("mini_batch_prompts.jsonl"))}


def normalize_raw_rows(input_path: Path) -> list[dict[str, Any]]:
    raw_rows = read_jsonl_if_exists(input_path)
    sources_by_plan = {row.get("synthetic_plan_id", ""): row for row in read_mini_table("mini_batch_source_selection.csv")}
    prompts_by_plan = load_prompts_by_plan()
    train_by_id = load_train_by_source_id()
    out: list[dict[str, Any]] = []

    for raw in raw_rows:
        plan_id = stringify(raw.get("synthetic_plan_id"))
        source_meta = sources_by_plan.get(plan_id, {})
        prompt_meta = prompts_by_plan.get(plan_id, {})
        source_id = stringify(raw.get("source_record_id") or source_meta.get("source_record_id"))
        train_row = train_by_id.get(source_id, {})
        plan_label = stringify(source_meta.get("target_label_5") or prompt_meta.get("target_label_5"))
        plan_error = stringify(source_meta.get("error_type") or prompt_meta.get("error_type"))
        parsed, parse_status = parse_raw_response(stringify(raw.get("raw_response")))
        notes: list[str] = [parse_status]
        returned_label = stringify(parsed.get("target_label_5"))
        label_mismatch = bool(returned_label and returned_label != plan_label)
        if returned_label and returned_label != plan_label:
            notes.append(f"returned_label_mismatch:{returned_label}; using_plan_label:{plan_label}")
        returned_error = stringify(parsed.get("error_type"))
        if returned_error and returned_error != plan_error:
            notes.append(f"returned_error_type_mismatch:{returned_error}; using_plan_error:{plan_error}")

        answer = stringify(parsed.get("answer_synthetic"))
        if not answer:
            notes.append("missing_answer_synthetic")
        synthetic_id = stringify(source_meta.get("synthetic_id") or prompt_meta.get("synthetic_id"))
        if not synthetic_id:
            synthetic_id = make_synthetic_id(plan_id, source_id, plan_label, plan_error)
        out.append(
            {
                "synthetic_id": synthetic_id,
                "generation_split_mode": source_meta.get("generation_split_mode") or prompt_meta.get("generation_split_mode") or "question_disjoint_formal",
                "synthetic_plan_id": plan_id,
                "source_record_id": source_id,
                "source_file": train_row.get("source_file", ""),
                "source_row_index": train_row.get("source_row_index", ""),
                "source_question_key": source_meta.get("source_question_key") or train_row.get("question_key", ""),
                "source_triple_key": source_meta.get("source_triple_key") or train_row.get("triple_key", ""),
                "source_split": "train" if train_row else source_meta.get("source_split", ""),
                "question": train_row.get("question", ""),
                "source_answer": train_row.get("answer", ""),
                "answer_synthetic": answer,
                "metric_canonical": source_meta.get("metric_canonical") or train_row.get("metric_canonical", ""),
                "rubric_text": train_row.get("rubric", ""),
                "language": source_meta.get("language") or train_row.get("language", ""),
                "scenario_canonical": train_row.get("scenario_canonical", ""),
                "subject_canonical": train_row.get("subject_canonical", ""),
                "education_level_canonical": train_row.get("education_level_canonical", ""),
                "target_label_5": plan_label,
                "label_source": "synthetic_design",
                "label_provenance": "pseudo_label",
                "score_raw": plan_label,
                "score_scale_detected": "1-5",
                "error_type": plan_error,
                "rationale_for_label": stringify(parsed.get("rationale_for_label")),
                "expected_failure_against_rubric": stringify(parsed.get("expected_failure_against_rubric")),
                "generation_method": "mini_batch_api",
                "generation_model": raw.get("generation_model", ""),
                "generation_prompt_version": MINI_BATCH_PROMPT_VERSION,
                "generation_timestamp": raw.get("created_at") or now_iso(),
                "generation_status": raw.get("generation_status", ""),
                "raw_generation": raw.get("raw_response", ""),
                "filter_status": "needs_review" if label_mismatch else "pending",
                "filter_reasons": [],
                "normalization_status": "pass" if parse_status.startswith("parsed_json") and answer else "fail",
                "normalization_notes": "; ".join(notes),
            }
        )
    return out


def normalize(input_path: Path | None = None) -> list[dict[str, Any]]:
    ensure_mini_batch_dirs()
    input_path = input_path or mini_generated_path("raw_generations.jsonl")
    if not input_path.exists():
        write_empty_jsonl(NORMALIZED_PATH)
        return []
    rows = normalize_raw_rows(input_path)
    write_mini_jsonl(NORMALIZED_PATH, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=None)
    args = parser.parse_args()
    rows = normalize(args.input_jsonl)
    print(f"Wrote {len(rows)} normalized mini-batch candidates to {NORMALIZED_PATH}")


if __name__ == "__main__":
    main()
