"""Prepare Exp27I target-aware teacher-audit packets.

Exp27G showed that a generic "Question / Answer" protocol can make a teacher
score the embedded student answer inside the question instead of the evaluator
output stored in the dataset's answer field. Exp27I keeps the same train-only
361 sample set, but rewrites the teacher packet and protocol so the scoring
target is explicit:

    score only teacher_input.answer, using teacher_input.question as context.

This script only prepares packets/protocol files. It does not call APIs, train,
or read dev/test labels beyond sample_id/question_key leakage guards inherited
from Exp27G preparation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence import prepare_exp27g_teacher_audit_361_packets as exp27g  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27i_target_aware_teacher_audit_361_seed42"
)

PROMPT_VERSION = "exp27i_target_aware_v1"
SCHEMA_VERSION = "exp27i_target_aware_teacher_audit_v1"
TARGET_VALUE = "evaluator_output_answer_field"


TARGET_CONTEXT_NOTE = """[CONTEXT_ONLY_ORIGINAL_TASK]
Use the content below only as context for judging the evaluator output. It may
contain an embedded "student answer", an original requested answer format, or
other task text. Do not assign a score to that embedded answer directly.
"""

TARGET_ANSWER_NOTE = """[EVALUATOR_OUTPUT_TO_SCORE]
This is the only text you must score. Treat it as the answer/evaluator output
being audited. Compare this evaluator output against the metric and rubric
using the context above.
"""


BLIND_PROMPT = f"""You are a strict educational assessment teacher-auditor.

Critical target rule:
- Score only the content under `[EVALUATOR_OUTPUT_TO_SCORE]` in the user message.
- The `[CONTEXT_ONLY_ORIGINAL_TASK]` section is context only. It may contain an
  embedded student answer or an answer key. Do not score that embedded answer.
- If your explanation says the student merely selected one option while the
  evaluator output contains feedback, JSON, or grading text, you are scoring the
  wrong target.
- Always set `scored_target` to `{TARGET_VALUE}`.
- Use `target_confusion_risk="possible"` or `"high"` if the sample is hard to
  disambiguate, but still score the evaluator output in the Answer field.

Task: assign a 1-5 score for ONE evaluator output using only the given context,
evaluator output, metric, rubric, and metadata. You must not assume or infer
any original human score. You must not use any hidden label or recovered human
rationale.

Return exactly one JSON object matching the provided blind schema. Do not wrap
the JSON in Markdown. Do not include chain-of-thought.

Schema-first rules:
- Use exactly the top-level keys and nested keys shown in the schema.
- Do not create extra JSON keys, translated keys, rubric-section keys, or
  explanatory sub-objects.
- Enum fields must use one of the exact English enum strings in the schema.
- `teacher_reason` should be a concise rubric-grounded explanation, not a
  step-by-step private reasoning trace.

Scoring rules:
- Ground the score in the rubric and metric, not in surface fluency.
- Low scores or deductions must identify a rubric-linked failure and a relevant
  `rubric_clause`.
- `teacher_reason` must not say "Score X", "I give X", or merely restate the
  `teacher_score` field.
- `major_failures` must not include label-conflict concepts, because blind
  scoring cannot see the original score.
- `answer_key_uncertainty` is the only place to mark possible answer-key or
  reference ambiguity.

Evidence rules:
- If `evidence_span` is not null, it must be a verbatim substring of the
  evaluator output under `[EVALUATOR_OUTPUT_TO_SCORE]`.
- If the failure is an absence such as missing reasoning, missing key point,
  missing personalization, missing scenario integration, missing explanation,
  or missing required format, set `evidence_span=null`,
  `evidence_type="missing_required_content"`, and fill
  `missing_evidence_reason`.
- If there is no material failure and the evaluator output is high quality, use
  `major_failures=["no_major_failure"]`, `evidence_span=null`,
  `evidence_type="not_applicable"`, `missing_evidence_reason=null`,
  `score_cap=null`, and `failure_visibility="no_major_failure"`.
- Do not use `major_failures=["no_major_failure"]` for score 1 or score 2.

Risk-field definitions:
- `score_region` is mechanical: score 1/2 = low, score 3 = mid, score 4/5 = high.
- `failure_visibility` describes whether the evaluator output's failure is
  explicit, hidden, missing required content, absent, or unclear.
- `surface_plausibility` describes how the evaluator output looks before careful
  rubric checking.
- `overestimation_risk` estimates whether a lenient LLM judge would incorrectly
  assign this evaluator output a 4/5.
- `score_cap` is the maximum reasonable score if a serious failure exists; use
  null when no cap is needed.

Do not penalize length, style, or wording unless the rubric requires it.
"""


AUDIT_PROMPT = f"""You are auditing whether the original human score is reliable.

Critical target rule:
- The blind teacher output should have scored only the evaluator output under
  `[EVALUATOR_OUTPUT_TO_SCORE]`.
- The `[CONTEXT_ONLY_ORIGINAL_TASK]` section is context only and may contain an
  embedded student answer. Do not audit by scoring that embedded answer.
- Always set `scored_target` to `{TARGET_VALUE}`.
- Set `target_confusion_detected=true` if the blind teacher output appears to
  grade the context/student answer instead of the evaluator output.

You will receive:
1. the original context, evaluator output, metric, rubric, and metadata,
2. the previous blind teacher output and its annotation id/hash,
3. the original human score for this train sample.

Return exactly one JSON object matching the provided audit schema. Do not wrap
the JSON in Markdown. Do not include chain-of-thought.

Important protocol rule:
- Do not copy or rewrite the blind object.
- Echo only `sample_id`, `blind_annotation_id`, `blind_annotation_hash`, and an
  `audit` object.
- The blind annotation is treated as fixed evidence for this audit pass. If the
  blind annotation seems questionable or target-confused, explain that inside
  `audit_reason` and choose conservative training use.

Audit rules:
- The teacher is an auditor, not a replacement gold-label source.
- Keep the blind teacher score conceptually separate from the original human
  score. Do not retroactively rationalize the original score.
- If the blind teacher score and original score differ by 0, use
  `score_agreement="exact"` unless another clear issue exists.
- If they differ by 1, usually use `score_agreement="adjacent"` and
  `label_quality="plausible_adjacent"`.
- If they differ by 2 or more, default to `score_agreement="conflict"`,
  `label_quality="suspected_conflict"`, `hard_conflict=true`, and
  `needs_human_review=true`, unless `audit_reason` clearly explains why the
  difference is acceptable.
- Use `label_noise_type` for rubric ambiguity, answer-key conflict,
  insufficient context, teacher strictness/leniency, target confusion, or
  annotator disagreement.
- `recommended_training_use` should be conservative:
  `high_weight` for reliable labels, `low_weight` for adjacent/plausible labels,
  `review_only` for conflicts or ambiguity, and `exclude` only for clearly
  unusable labels.
"""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_base_schema(name: str) -> dict[str, Any]:
    path = exp27g.SCHEMA_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def target_aware_blind_schema() -> dict[str, Any]:
    schema = load_base_schema("exp27d_teacher_blind_schema_v4.json")
    schema["title"] = "Exp27I Target-Aware Teacher Blind Annotation V1"
    blind = schema["properties"]["blind"]
    required = blind["required"]
    for key in ["scored_target", "target_confusion_risk", "target_scope_reason"]:
        if key not in required:
            required.append(key)
    blind["properties"]["scored_target"] = {"type": "string", "enum": [TARGET_VALUE]}
    blind["properties"]["target_confusion_risk"] = {
        "type": "string",
        "enum": ["none", "possible", "high"],
    }
    blind["properties"]["target_scope_reason"] = {"type": "string"}
    return schema


def target_aware_audit_schema() -> dict[str, Any]:
    schema = load_base_schema("exp27d_teacher_audit_schema_v4.json")
    schema["title"] = "Exp27I Target-Aware Teacher Label Audit V1"
    audit = schema["properties"]["audit"]
    required = audit["required"]
    for key in ["scored_target", "target_confusion_detected", "target_scope_reason"]:
        if key not in required:
            required.append(key)
    audit["properties"]["scored_target"] = {"type": "string", "enum": [TARGET_VALUE]}
    audit["properties"]["target_confusion_detected"] = {"type": "boolean"}
    audit["properties"]["target_scope_reason"] = {"type": "string"}
    noise_enum = audit["properties"]["label_noise_type"]["enum"]
    if "target_confusion" not in noise_enum:
        noise_enum.append("target_confusion")
    return schema


def json_metadata_with_target(metadata_text: str) -> str:
    try:
        meta = json.loads(metadata_text) if metadata_text else {}
        if not isinstance(meta, dict):
            meta = {"raw_metadata": metadata_text}
    except json.JSONDecodeError:
        meta = {"raw_metadata": metadata_text}
    meta.update(
        {
            "target_to_score": TARGET_VALUE,
            "context_field_role": "context_only_original_task",
            "answer_field_role": "evaluator_output_to_score",
        }
    )
    return json.dumps(meta, ensure_ascii=False, sort_keys=True)


def target_aware_packet(packet: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(packet)
    teacher_input = out.get("teacher_input") if isinstance(out.get("teacher_input"), dict) else {}
    teacher_input["question"] = f"{TARGET_CONTEXT_NOTE}\n{teacher_input.get('question', '')}".strip()
    teacher_input["answer"] = f"{TARGET_ANSWER_NOTE}\n{teacher_input.get('answer', '')}".strip()
    teacher_input["metadata"] = json_metadata_with_target(str(teacher_input.get("metadata", "")))
    out["teacher_input"] = teacher_input
    out["prompt_version"] = PROMPT_VERSION
    out["schema_version"] = SCHEMA_VERSION
    out["target_to_score"] = TARGET_VALUE
    source_meta = out.setdefault("source_meta", {})
    if isinstance(source_meta, dict):
        source_meta["target_to_score"] = TARGET_VALUE
        source_meta["target_disambiguation"] = "question_context_answer_is_scored_evaluator_output"
        source_meta["sample_hash"] = exp27g.sha1(json.dumps(teacher_input, ensure_ascii=False, sort_keys=True))
    return out


def write_protocol_files(out_dir: Path) -> dict[str, str]:
    prompt_dir = out_dir / "prompts"
    schema_dir = out_dir / "schema"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)

    blind_prompt_path = prompt_dir / "exp27d_blind_teacher_prompt_v4.md"
    audit_prompt_path = prompt_dir / "exp27d_label_audit_prompt_v4.md"
    blind_schema_path = schema_dir / "exp27d_teacher_blind_schema_v4.json"
    audit_schema_path = schema_dir / "exp27d_teacher_audit_schema_v4.json"

    write_text(blind_prompt_path, BLIND_PROMPT)
    write_text(audit_prompt_path, AUDIT_PROMPT)
    write_text(
        blind_schema_path,
        json.dumps(target_aware_blind_schema(), ensure_ascii=False, indent=2, sort_keys=True),
    )
    write_text(
        audit_schema_path,
        json.dumps(target_aware_audit_schema(), ensure_ascii=False, indent=2, sort_keys=True),
    )
    return {
        "blind_prompt_sha1": exp27g.file_sha1(blind_prompt_path),
        "audit_prompt_sha1": exp27g.file_sha1(audit_prompt_path),
        "blind_schema_sha1": exp27g.file_sha1(blind_schema_path),
        "audit_schema_sha1": exp27g.file_sha1(audit_schema_path),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir

    # Reuse the exact Exp27G train-only 361 sampling recipe, then overwrite the
    # protocol and packet wording so only the target semantics change.
    exp27g_args = argparse.Namespace(
        train_jsonl=args.train_jsonl,
        dev_jsonl=args.dev_jsonl,
        test_jsonl=args.test_jsonl,
        exp27f_adjudications=args.exp27f_adjudications,
        out_dir=out_dir,
        seed=args.seed,
        total_count=args.total_count,
        low_count=args.low_count,
        high_conflict_count=args.high_conflict_count,
        mid_count=args.mid_count,
        edu_count=args.edu_count,
        batch_size=args.batch_size,
    )
    base_decision = exp27g.prepare(exp27g_args)

    packet_path = out_dir / "packets" / "exp27d_v4_repilot_blind_packets.jsonl"
    ref_path = out_dir / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl"
    packets = [target_aware_packet(row) for row in read_jsonl(packet_path)]
    refs = read_jsonl(ref_path)

    for ref in refs:
        ref["prompt_version"] = PROMPT_VERSION
        ref["schema_version"] = SCHEMA_VERSION
        ref["target_to_score"] = TARGET_VALUE

    write_jsonl(packet_path, packets)
    write_jsonl(ref_path, refs)
    write_jsonl(out_dir / "packets" / "exp27i_target_aware_361_teacher_packets.jsonl", packets)
    write_jsonl(out_dir / "packets" / "exp27i_target_aware_361_audit_reference_private.jsonl", refs)
    protocol_hashes = write_protocol_files(out_dir)

    group_counts = Counter(packet.get("pilot_group", "") for packet in packets)
    label_counts = Counter(ref.get("original_score") for ref in refs)
    write_csv(
        out_dir / "tables" / "exp27i_target_aware_sampling_distribution.csv",
        [
            {"group": group, "count": count}
            for group, count in sorted(group_counts.items())
        ],
    )

    decision = {
        **base_decision,
        "experiment": "exp27i_target_aware_teacher_audit_361",
        "recommendation": "run_target_aware_dual_teacher_api_then_codex_direct_adjudication",
        "packet_rows": len(packets),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "target_to_score": TARGET_VALUE,
        "label_counts": dict(sorted((str(k), v) for k, v in label_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "protocol_fix": "question_is_context_only_answer_is_evaluator_output_to_score",
        "requires_codex_direct_review_after_api": True,
        **protocol_hashes,
    }
    write_json(out_dir / "decision" / "exp27i_prepare_decision.json", decision)

    report = [
        "# Exp27I Target-Aware Teacher-Audit Preparation",
        "",
        "Exp27I reuses the Exp27G train-only 361 sample set, but fixes the scoring target ambiguity.",
        "",
        "## Target Rule",
        "",
        "- The `question` field is wrapped as context only.",
        "- The `answer` field is wrapped as the evaluator output to score.",
        "- Blind and audit schemas require explicit target-scope fields.",
        "- The API stage still does not see dev/test labels or recovered human reasons.",
        "",
        "## Counts",
        "",
        f"- packets: {len(packets)}",
        f"- label_counts: `{dict(sorted(label_counts.items()))}`",
        f"- group_counts: `{dict(sorted(group_counts.items()))}`",
        "",
        "## Next Step",
        "",
        "Run Qwen and DeepSeek on these target-aware packets, then perform direct Codex semantic adjudication on teacher/human conflicts.",
    ]
    write_text(out_dir / "reports" / "exp27i_prepare_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp27I target-aware teacher-audit packets.")
    parser.add_argument("--train-jsonl", type=Path, default=exp27g.DEFAULT_TRAIN)
    parser.add_argument("--dev-jsonl", type=Path, default=exp27g.DEFAULT_DEV)
    parser.add_argument("--test-jsonl", type=Path, default=exp27g.DEFAULT_TEST)
    parser.add_argument("--exp27f-adjudications", type=Path, default=exp27g.DEFAULT_EXP27F_ADJUDICATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-count", type=int, default=361)
    parser.add_argument("--low-count", type=int, default=111)
    parser.add_argument("--high-conflict-count", type=int, default=80)
    parser.add_argument("--mid-count", type=int, default=70)
    parser.add_argument("--edu-count", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
