"""Validate generator output before constructing blind verifier packets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .audit_exp48a_question_novelty import audit
from .common import OUT, PRIVATE, TRAIN, ensure_output_layout, normalize_text, read_jsonl, sha256_path, validate_family, write_csv, write_json

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - environment preflight
    raise RuntimeError("Exp48A generation validation requires jsonschema") from exc


def language_matches(language: str, text: str) -> bool:
    han = sum("\u4e00" <= char <= "\u9fff" for char in text)
    ratio = han / max(1, len(text))
    return ratio >= 0.08 if language == "zh" else ratio < 0.08


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=Path, default=PRIVATE / "generated_families/exp48a_generated_families.jsonl")
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    ensure_output_layout(args.out_dir)
    families = read_jsonl(args.families)
    train = read_jsonl(args.train)
    schema = json.loads((Path(__file__).resolve().parent / "schemas/exp48a_synthetic_family_schema.json").read_text(encoding="utf-8"))
    novelty_rows, novelty = audit(families, train)
    source_questions = {normalize_text(row["question"]) for row in train}
    seen_questions: set[str] = set()
    seen_keys: set[str] = set()
    seen_family_ids: set[str] = set()
    validation = []
    private_failures = []
    style_rows = []
    for family in families:
        errors = validate_family(family)
        try:
            jsonschema.validate(family, schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"json_schema_invalid:{exc.json_path}")
        question = normalize_text(family.get("synthetic_question", ""))
        key = str(family.get("synthetic_question_key", ""))
        family_id = str(family.get("family_id", ""))
        if question in source_questions:
            errors.append("source_question_exact_duplicate")
        if question in seen_questions:
            errors.append("synthetic_question_duplicate")
        if key in seen_keys:
            errors.append("synthetic_question_key_duplicate")
        if not family_id or family_id in seen_family_ids:
            errors.append("family_id_duplicate_or_empty")
        seen_questions.add(question)
        seen_keys.add(key)
        seen_family_ids.add(family_id)
        for answer in family.get("answers", []):
            if not language_matches(str(family.get("language", "")), str(answer.get("text", ""))):
                errors.append(f"answer_language_mismatch:{answer.get('answer_id','unknown')}")
        errors = sorted(set(errors))
        valid = not errors
        validation.append({"status": "valid" if valid else "invalid", "count": 1})
        if errors:
            private_failures.append({"family_id": family.get("family_id"), "errors": errors})
        lengths = [len(normalize_text(answer.get("text", ""))) for answer in family.get("answers", [])]
        style_rows.append({
            "metric": family.get("metric", ""), "language": family.get("language", ""),
            "family_count": 1, "answer_count": len(lengths),
            "mean_answer_chars": sum(lengths) / max(1, len(lengths)),
            "max_min_length_ratio": max(lengths) / max(1, min(lengths)) if lengths else 0,
        })
    valid_count = sum(row["status"] == "valid" for row in validation)
    decision = {
        "status": "GENERATION_VALID" if len(families) == 60 and valid_count >= 54 and novelty["novelty_pass"] else "GENERATION_NO_GO",
        "generated_families": len(families), "valid_families": valid_count,
        "unique_synthetic_question_keys": len(seen_keys), "answers": sum(len(f.get("answers", [])) for f in families),
        **novelty, "dev_access_count": 0, "test_access_count": 0,
    }
    counts = Counter(row["status"] for row in validation)
    write_csv(args.out_dir / "tables/exp48a_generation_completion.csv", [{"status": key, "family_count": value} for key, value in sorted(counts.items())])
    aggregate_style = []
    for language in sorted({row["language"] for row in style_rows}):
        subset = [row for row in style_rows if row["language"] == language]
        aggregate_style.append({
            "language": language, "families": len(subset), "answers": sum(row["answer_count"] for row in subset),
            "mean_answer_chars": sum(row["mean_answer_chars"] for row in subset) / max(1, len(subset)),
            "max_length_ratio": max((row["max_min_length_ratio"] for row in subset), default=0),
        })
    write_csv(args.out_dir / "tables/exp48a_answer_length_style_audit.csv", aggregate_style)
    write_csv(args.out_dir / "tables/exp48a_question_novelty_audit.csv", [{"statistic": key, "value": value} for key, value in novelty.items()])
    write_json(args.out_dir / "private/generated_families/exp48a_generation_failures.json", private_failures)
    write_json(args.out_dir / "private/generated_families/exp48a_question_novelty_rows.json", novelty_rows)
    write_json(args.out_dir / "decision/exp48a_generation_decision.json", decision)
    write_json(args.out_dir / "hashes/exp48a_private_artifact_hashes.json", {"generated_families_sha256": sha256_path(args.families)})
    report = ["# Exp48A generation validation report", "", f"- Status: **{decision['status']}**"] + [f"- {key}: `{value}`" for key, value in decision.items() if key != "status"]
    (args.out_dir / "reports/exp48a_generation_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True))
    if decision["status"] != "GENERATION_VALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
