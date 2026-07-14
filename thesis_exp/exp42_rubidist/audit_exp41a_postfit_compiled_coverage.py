"""Audit Exp41 compiled-rubric content retained after the student token fit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp42_rubidist.common import EXP41_ROOT, ROOT, stable_hash, write_csv  # noqa: E402
from thesis_exp.exp41_rubric_bridge.common import read_jsonl  # noqa: E402

DEFAULT_TOKENIZER = "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--exp41-root", type=Path, default=EXP41_ROOT)
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_TOKENIZER)
    return parser.parse_args()


def band(index: int, total: int) -> str:
    position = (index + 0.5) / max(total, 1)
    return "early" if position <= 1 / 3 else "middle" if position <= 2 / 3 else "late"


def main() -> None:
    args = parse_args()
    private = args.exp41_root / "private"
    unit_path = private / "rubric_units/exp41a_rubric_units.jsonl"
    compiled_path = private / "compiled_rubrics/exp41a_qwen_compiled_rubrics.jsonl"
    fitted_path = private / "compiled_rubrics/exp41a_fitted_compiled_rubrics.jsonl"
    required = [unit_path, compiled_path, fitted_path]
    table_path = args.out_dir / "tables/exp42a_exp41_postfit_coverage_audit.csv"
    retention_path = args.out_dir / "tables/exp42a_exp41_criterion_retention.csv"
    report_path = args.out_dir / "reports/exp42a_exp41_postfit_coverage_report.md"
    if not all(path.exists() for path in required):
        write_csv(table_path, [{"status": "MISSING_PRIVATE_EXP41_COMPILER_DATA", "missing_count": sum(not path.exists() for path in required)}])
        write_csv(retention_path, [{"status": "MISSING_PRIVATE_EXP41_COMPILER_DATA"}])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("# Exp41 post-fit coverage audit\n\n- Status: **MISSING_PRIVATE_EXP41_COMPILER_DATA**\n- This does not block Exp42A.\n", encoding="utf-8")
        print(json.dumps({"status": "MISSING_PRIVATE_EXP41_COMPILER_DATA"}, sort_keys=True))
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, local_files_only=True)
    units = {row["rubric_unit_id"]: row for row in read_jsonl(unit_path)}
    compiled = {row["rubric_unit_id"]: row for row in read_jsonl(compiled_path)}
    fitted = {row["rubric_unit_id"]: row for row in read_jsonl(fitted_path)}
    if set(units) != set(compiled) or set(compiled) != set(fitted):
        raise RuntimeError("Exp41 unit/original/fitted rubric IDs differ")

    unit_rows = []
    type_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    position_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for unit_id in sorted(units):
        raw = str(units[unit_id].get("raw_rubric") or "")
        raw_tokens = max(len(tokenizer.encode(raw, add_special_tokens=False)), 1)
        original = compiled[unit_id]
        visible = fitted[unit_id]
        original_criteria = list(original.get("criteria", []))
        visible_ids = {str(row["criterion_id"]) for row in visible.get("criteria", [])}
        original_quote_tokens = sum(len(tokenizer.encode(str(row.get("rubric_quote") or ""), add_special_tokens=False)) for row in original_criteria)
        visible_quote_tokens = sum(
            len(tokenizer.encode(str(row.get("rubric_quote") or ""), add_special_tokens=False))
            for row in original_criteria
            if str(row["criterion_id"]) in visible_ids
        )
        for index, criterion in enumerate(original_criteria):
            kept = int(str(criterion["criterion_id"]) in visible_ids)
            criterion_type = str(criterion.get("criterion_type") or "other")
            position = band(index, len(original_criteria))
            type_totals[criterion_type][0] += 1
            type_totals[criterion_type][1] += kept
            position_totals[position][0] += 1
            position_totals[position][1] += kept
        original_rules = list(original.get("score_level_rules", []))
        kept_rules = list(visible.get("score_level_rules", []))
        unit_rows.append(
            {
                "status": "OK",
                "rubric_unit_hash": stable_hash(unit_id)[:16],
                "original_criterion_count": len(original_criteria),
                "kept_criterion_count": len(visible_ids),
                "truncated_criterion_count": len(original_criteria) - len(visible_ids),
                "retained_criterion_ratio": len(visible_ids) / max(len(original_criteria), 1),
                "original_exact_quote_token_coverage": min(original_quote_tokens / raw_tokens, 1.0),
                "postfit_student_visible_exact_quote_token_coverage": min(visible_quote_tokens / raw_tokens, 1.0),
                "original_score_level_rule_count": len(original_rules),
                "kept_score_level_rule_count": len(kept_rules),
                "score_level_rule_retention": len(kept_rules) / max(len(original_rules), 1),
            }
        )
    write_csv(table_path, unit_rows)
    retention_rows = []
    for category, totals in sorted(type_totals.items()):
        retention_rows.append({"group_type": "criterion_type", "group_value": category, "original_count": totals[0], "kept_count": totals[1], "retention_rate": totals[1] / max(totals[0], 1)})
    for category in ("early", "middle", "late"):
        totals = position_totals[category]
        retention_rows.append({"group_type": "ordinal_position", "group_value": category, "original_count": totals[0], "kept_count": totals[1], "retention_rate": totals[1] / max(totals[0], 1)})
    write_csv(retention_path, retention_rows)

    coverage = np.asarray([row["postfit_student_visible_exact_quote_token_coverage"] for row in unit_rows], dtype=float)
    retained = np.asarray([row["retained_criterion_ratio"] for row in unit_rows], dtype=float)
    rules_original = sum(row["original_score_level_rule_count"] for row in unit_rows)
    rules_kept = sum(row["kept_score_level_rule_count"] for row in unit_rows)
    positions = {row["group_value"]: row["retention_rate"] for row in retention_rows if row["group_type"] == "ordinal_position"}
    prefix_bias = float(positions.get("early", 0.0) - positions.get("late", 0.0))
    report = [
        "# Exp41 post-fit compiled-rubric coverage audit",
        "",
        "- Status: **COMPLETED**",
        f"- Rubric units: `{len(unit_rows)}`",
        f"- Mean post-fit exact-quote token coverage: `{float(np.mean(coverage)):.6f}`",
        f"- P05/P50 post-fit coverage: `{float(np.percentile(coverage, 5)):.6f}` / `{float(np.percentile(coverage, 50)):.6f}`",
        f"- Fraction retaining at least 60% coverage: `{float(np.mean(coverage >= 0.60)):.6f}`",
        f"- Mean retained-criterion ratio: `{float(np.mean(retained)):.6f}`",
        f"- Score-level-rule retention: `{rules_kept / max(rules_original, 1):.6f}`",
        f"- Early-minus-late criterion retention (prefix-position bias): `{prefix_bias:.6f}`",
        "- This descriptive audit does not affect the Exp42A decision.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "rubric_units": len(unit_rows), "mean_postfit_coverage": float(np.mean(coverage)), "prefix_position_bias": prefix_bias}, sort_keys=True))


if __name__ == "__main__":
    main()
