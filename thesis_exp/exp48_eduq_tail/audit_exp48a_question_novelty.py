"""Audit synthetic-question novelty against locked train questions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import OUT, PRIVATE, TRAIN, char_ngrams, ensure_output_layout, jaccard, normalize_text, read_jsonl, tokens, write_csv, write_json


def audit(families: list[dict], train_rows: list[dict]) -> tuple[list[dict], dict]:
    train_questions = list({normalize_text(row["question"]): row["question"] for row in train_rows}.values())
    train_norm = [normalize_text(text) for text in train_questions]
    train_chars = [char_ngrams(text) for text in train_questions]
    train_tokens = [tokens(text) for text in train_questions]
    rows = []
    for family in families:
        question = str(family["synthetic_question"])
        norm = normalize_text(question)
        cgrams = char_ngrams(question)
        tset = tokens(question)
        char_scores = [jaccard(cgrams, candidate) for candidate in train_chars]
        token_scores = [jaccard(tset, candidate) for candidate in train_tokens]
        rows.append({
            "family_id": family["family_id"], "normalized_exact_match": int(norm in train_norm),
            "max_char5_jaccard": max(char_scores, default=0.0),
            "max_token_jaccard": max(token_scores, default=0.0),
        })
    summary = {
        "families": len(rows), "exact_match_count": sum(row["normalized_exact_match"] for row in rows),
        "max_char5_jaccard": max((row["max_char5_jaccard"] for row in rows), default=0.0),
        "mean_max_char5_jaccard": sum(row["max_char5_jaccard"] for row in rows) / max(1, len(rows)),
        "max_token_jaccard": max((row["max_token_jaccard"] for row in rows), default=0.0),
        "mean_max_token_jaccard": sum(row["max_token_jaccard"] for row in rows) / max(1, len(rows)),
    }
    summary["novelty_pass"] = summary["exact_match_count"] == 0 and summary["max_char5_jaccard"] < 0.80 and summary["max_token_jaccard"] < 0.80
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=Path, default=PRIVATE / "generated_families/exp48a_generated_families.jsonl")
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    ensure_output_layout(args.out_dir)
    family_rows = read_jsonl(args.families)
    train_rows = read_jsonl(args.train)
    private_rows, summary = audit(family_rows, train_rows)
    write_csv(args.out_dir / "private/generated_families/exp48a_question_novelty_rows.csv", private_rows)
    write_csv(args.out_dir / "tables/exp48a_question_novelty_audit.csv", [{"statistic": key, "value": value} for key, value in summary.items()])
    write_json(args.out_dir / "private/generated_families/exp48a_question_novelty_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
