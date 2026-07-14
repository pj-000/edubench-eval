"""Validate compiled rubrics and apply the frozen Exp41A qualification gate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jsonschema
except ModuleNotFoundError:
    from thesis_exp.exp17_low_score_evidence import json_schema_compat as jsonschema

from thesis_exp.exp41_rubric_bridge.common import (  # noqa: E402
    ROOT, fit_compiled_rubric, lexical_tokens, normalize_text, read_jsonl, sha256_file,
    stable_hash, write_csv, write_json, write_jsonl,
)

DEFAULT_TOKENIZER = "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B"
LEAKAGE_MARKERS = ("human_1", "human_2", "human_3", "label_5", "judge_scores", "teacher_score", "predicted_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_TOKENIZER)
    return parser.parse_args()


def score_level_supported(level: int, quote: str) -> bool:
    text = normalize_text(quote)
    return bool(re.search(rf"(?:^|\D){level}\s*(?:分|[:：.)、-])", text))


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True, local_files_only=True)
    unit_path = args.out_dir / "private/rubric_units/exp41a_rubric_units.jsonl"
    compiled_path = args.out_dir / "private/compiled_rubrics/exp41a_qwen_compiled_rubrics.jsonl"
    if not unit_path.exists() or not compiled_path.exists():
        raise FileNotFoundError("Prepared rubric units and compiled rubric outputs are required")
    units = read_jsonl(unit_path)
    outputs = read_jsonl(compiled_path)
    unit_by_id = {str(row["rubric_unit_id"]): row for row in units}
    output_by_id = {str(row.get("rubric_unit_id")): row for row in outputs}
    schema = json.loads((args.out_dir / "schemas/exp41a_compiled_rubric_schema.json").read_text(encoding="utf-8"))
    duplicate_outputs = len(outputs) - len(output_by_id)
    rows = []
    criterion_types: Counter[str] = Counter()
    total_criteria = duplicate_criteria = 0
    compiled_hash_items = []
    fitted_outputs = []
    for unit_id in sorted(unit_by_id):
        unit = unit_by_id[unit_id]
        output = output_by_id.get(unit_id)
        errors = []
        if output is None:
            rows.append({"rubric_unit_id_hash": unit_id, "completed": False, "valid": False, "errors": "missing_output"})
            continue
        schema_errors = [error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(output)]
        errors.extend(schema_errors)
        if str(output.get("target_scope")) != "rubric_only":
            errors.append("target_scope")
        criteria = output.get("criteria") if isinstance(output.get("criteria"), list) else []
        ids = [str(item.get("criterion_id")) for item in criteria if isinstance(item, dict)]
        quotes = [normalize_text(item.get("rubric_quote", "")) for item in criteria if isinstance(item, dict)]
        duplicate_count = len(quotes) - len(set(quotes))
        duplicate_criteria += duplicate_count
        total_criteria += len(criteria)
        if len(ids) != len(set(ids)):
            errors.append("duplicate_criterion_id")
        if duplicate_count:
            errors.append("duplicate_criterion_quote")
        raw_normalized = normalize_text(unit["raw_rubric"])
        quote_valid = bool(quotes) and all(quote and quote in raw_normalized for quote in quotes)
        if not quote_valid:
            errors.append("invalid_exact_quote")
        question_valid = all(str(item.get("check_question", "")).strip().endswith(("?", "？")) for item in criteria if isinstance(item, dict))
        if not question_valid:
            errors.append("check_question_not_question")
        for item in criteria:
            if isinstance(item, dict):
                criterion_types[str(item.get("criterion_type", "unknown"))] += 1
        rules = output.get("score_level_rules") if isinstance(output.get("score_level_rules"), list) else []
        rule_valid = True
        for rule in rules:
            quote = normalize_text(rule.get("rubric_quote", ""))
            level = int(rule.get("score_level", 0) or 0)
            if quote not in raw_normalized or not score_level_supported(level, quote) or not set(rule.get("required_criteria", [])) <= set(ids):
                rule_valid = False
        if not rule_valid:
            errors.append("unsupported_score_level_rule")
        fitted = None
        truncated_criteria = 0
        try:
            ordered_output = output
            if not schema_errors:
                ordered_output = {
                    **output,
                    "criteria": sorted(
                        output["criteria"],
                        key=lambda item: (raw_normalized.find(normalize_text(item["rubric_quote"])), str(item["criterion_id"])),
                    ),
                }
            fitted, serialized, truncated_criteria = fit_compiled_rubric(ordered_output, tokenizer, 256) if not schema_errors else (None, "", 0)
        except (KeyError, TypeError, ValueError) as exc:
            serialized = ""
            errors.append(f"compiled_fit:{type(exc).__name__}")
        token_length = len(tokenizer.encode(serialized, add_special_tokens=False)) if serialized else 0
        length_valid = bool(serialized) and token_length <= 256
        if not length_valid:
            errors.append("compiled_length_gt_256")
        raw_tokens = lexical_tokens(unit["raw_rubric"])
        covered_tokens = set().union(*(lexical_tokens(quote) for quote in quotes)) if quotes else set()
        coverage = len(raw_tokens & covered_tokens) / len(raw_tokens) if raw_tokens else 1.0
        serialized_lower = json.dumps(output, ensure_ascii=False).lower()
        leakage = any(marker in serialized_lower for marker in LEAKAGE_MARKERS)
        if leakage:
            errors.append("answer_or_label_leakage")
        valid = not errors
        rows.append({
            "rubric_unit_id_hash": unit_id, "question_key_hash": stable_hash(unit["question_key"]),
            "language": unit["language"], "metric_group": unit["metric_group"], "answer_row_count": unit["answer_row_count"],
            "completed": True, "schema_valid": not schema_errors, "target_scope_valid": output.get("target_scope") == "rubric_only",
            "quote_valid": quote_valid, "criterion_count": len(criteria), "duplicate_criterion_count": duplicate_count,
            "check_question_valid": question_valid, "score_level_rules_valid": rule_valid, "quote_coverage": coverage,
            "compiled_token_length": token_length, "compiled_length_valid": length_valid,
            "original_criterion_count": len(criteria), "kept_criterion_count": len(fitted["criteria"]) if fitted else 0,
            "truncated_criterion_count": truncated_criteria,
            "answer_label_leakage": leakage, "valid": valid, "errors": "|".join(errors),
        })
        compiled_hash_items.append((unit_id, stable_hash(output)))
        if fitted:
            fitted_outputs.append({
                **fitted, "serialized_text": serialized, "compiled_token_length": token_length,
                "truncated_criterion_count": truncated_criteria,
            })
    expected = len(units)
    completed = sum(bool(row.get("completed")) for row in rows)
    schema_rate = sum(bool(row.get("schema_valid")) for row in rows) / expected
    target_scope_rate = sum(bool(row.get("target_scope_valid")) for row in rows) / expected
    quote_rate = sum(bool(row.get("quote_valid")) for row in rows) / expected
    leakage_count = sum(bool(row.get("answer_label_leakage")) for row in rows)
    duplicate_rate = duplicate_criteria / max(total_criteria, 1)
    coverage_rate = sum(float(row.get("quote_coverage", 0)) >= 0.60 for row in rows) / expected
    length_rate = sum(bool(row.get("compiled_length_valid")) for row in rows) / expected
    gates = {
        "all_rubric_units_completed": completed == expected == 1044,
        "schema_success_ge_0p99": schema_rate >= 0.99,
        "target_scope_success_eq_1": target_scope_rate == 1.0,
        "quote_validity_eq_1": quote_rate == 1.0,
        "answer_label_leakage_eq_0": leakage_count == 0,
        "duplicate_criterion_rate_le_0p01": duplicate_rate <= 0.01,
        "units_coverage_ge_0p60_rate_ge_0p95": coverage_rate >= 0.95,
        "compiled_length_le_256_rate_eq_1": length_rate == 1.0,
        "dev_access_zero": True, "test_access_zero": True,
    }
    go = all(gates.values())
    write_csv(args.out_dir / "tables/exp41a_compiler_validation.csv", rows)
    write_csv(args.out_dir / "tables/exp41a_rubric_quote_coverage.csv", [{
        "rubric_unit_id_hash": row["rubric_unit_id_hash"], "quote_coverage": row.get("quote_coverage", 0),
        "criterion_count": row.get("criterion_count", 0), "compiled_token_length": row.get("compiled_token_length", 0),
    } for row in rows])
    write_csv(args.out_dir / "tables/exp41a_criterion_type_distribution.csv", [
        {"criterion_type": key, "count": count, "rate": count / max(total_criteria, 1)} for key, count in sorted(criterion_types.items())
    ])
    write_jsonl(args.out_dir / "private/compiled_rubrics/exp41a_fitted_compiled_rubrics.jsonl", fitted_outputs)
    coverage_values = [float(row.get("quote_coverage", 0)) for row in rows]
    criterion_counts = [int(row.get("criterion_count", 0)) for row in rows]
    token_lengths = [int(row.get("compiled_token_length", 0)) for row in rows]
    summary = {
        "expected_rubric_units": expected, "completed_rubric_units": completed, "duplicate_outputs": duplicate_outputs,
        "schema_success_rate": schema_rate, "target_scope_success_rate": target_scope_rate,
        "exact_quote_validity_rate": quote_rate, "answer_label_leakage_count": leakage_count,
        "total_criteria": total_criteria, "duplicate_criterion_rate": duplicate_rate,
        "coverage_ge_0p60_rate": coverage_rate, "compiled_length_le_256_rate": length_rate,
        "quote_coverage_mean": float(np.mean(coverage_values)), "quote_coverage_p05": float(np.percentile(coverage_values, 5)),
        "quote_coverage_p50": float(np.percentile(coverage_values, 50)),
        "criterion_count_mean": float(np.mean(criterion_counts)), "criterion_count_p95": float(np.percentile(criterion_counts, 95)),
        "compiled_token_length_mean": sum(token_lengths) / expected,
        "compiled_token_length_p95": float(np.percentile(token_lengths, 95)),
        "compiled_token_length_max": max(int(row.get("compiled_token_length", 0)) for row in rows),
        "units_with_criterion_truncation": sum(int(row.get("truncated_criterion_count", 0)) > 0 for row in rows),
        "truncated_criterion_count": sum(int(row.get("truncated_criterion_count", 0)) for row in rows),
    }
    decision = {"status": "COMPILER_QUALIFICATION_GO" if go else "COMPILER_QUALIFICATION_NO_GO",
                "gates": gates, "recommend_groupcv_training": go, "stop_llm_rubric_compiler_route": not go,
                **summary, "dev_access_count": 0, "test_access_count": 0}
    write_json(args.out_dir / "decision/exp41a_compiler_qualification_decision.json", decision)
    write_json(args.out_dir / "hashes/exp41a_compiled_rubric_hashes.json", {
        "compiled_private_path": str(compiled_path), "compiled_rows": len(outputs), "compiled_file_sha256": sha256_file(compiled_path),
        "compiled_identity_content_sha256": stable_hash(compiled_hash_items), "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "dev_access_count": 0, "test_access_count": 0,
    })
    report = [
        "# Exp41A compiler qualification report", "", f"- Status: **{decision['status']}**",
        f"- Completed rubric units: `{completed}/{expected}`", f"- Schema success: `{schema_rate:.4f}`",
        f"- Exact quote validity: `{quote_rate:.4f}`", f"- Coverage >= 0.60: `{coverage_rate:.4f}`",
        f"- Quote coverage mean/p05/p50: `{summary['quote_coverage_mean']:.4f}` / `{summary['quote_coverage_p05']:.4f}` / `{summary['quote_coverage_p50']:.4f}`",
        f"- Criteria per unit mean/p95: `{summary['criterion_count_mean']:.2f}` / `{summary['criterion_count_p95']:.2f}`",
        f"- Duplicate criterion rate: `{duplicate_rate:.4f}`", f"- Compiled length <=256 rate: `{length_rate:.4f}`",
        f"- Compiled tokens mean/p95/max: `{summary['compiled_token_length_mean']:.2f}` / `{summary['compiled_token_length_p95']:.2f}` / `{summary['compiled_token_length_max']}`",
        f"- Units/criteria truncated at complete boundaries: `{summary['units_with_criterion_truncation']}` / `{summary['truncated_criterion_count']}`",
        f"- Answer/label leakage count: `{leakage_count}`", f"- Gates: `{json.dumps(gates, sort_keys=True)}`",
        f"- Recommend GroupCV training: `{str(go).lower()}`", f"- Stop LLM rubric compiler route: `{str(not go).lower()}`",
        "- The compiler received no answer, human score, rounded label, judge score, or model prediction.",
        "- No paper-like dev/test data were accessed.",
    ]
    report_path = args.out_dir / "reports/exp41a_compiler_qualification_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    if not go:
        write_json(args.out_dir / "decision/exp41a_groupcv_decision.json", {
            "status": "GROUPCV_NOT_RUN_BLOCKED_BY_COMPILER_QUALIFICATION", "groupcv_ran": False,
            "recommend_run_multiseed": False, "stop_llm_rubric_compiler_route": True,
            "blocking_compiler_status": decision["status"], "dev_access_count": 0, "test_access_count": 0,
        })
        groupcv_report = args.out_dir / "reports/exp41a_rubric_bridge_groupcv_report.md"
        groupcv_report.write_text(
            "# Exp41A RUBRIC-Bridge GroupCV report\n\n"
            "- Status: **GROUPCV_NOT_RUN_BLOCKED_BY_COMPILER_QUALIFICATION**\n"
            "- The frozen compiler qualification gate failed, so no variant was trained.\n"
            "- No paper-like dev/test data were accessed.\n",
            encoding="utf-8",
        )
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
