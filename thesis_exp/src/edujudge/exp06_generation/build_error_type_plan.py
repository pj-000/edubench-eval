"""Write Exp6-2 low-score synthetic error type and target matrix plans."""

from __future__ import annotations

from collections import Counter
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import ERROR_TYPES, TARGET_LABEL_COUNTS_PER_METRIC_LANGUAGE
from thesis_exp.src.edujudge.exp06_generation.common import METRIC_ERROR_TYPE_MAP, load_split, write_table


ERROR_TYPE_FIELDS = [
    "error_type",
    "description",
    "target_label_range",
    "applicable_metrics",
    "prompt_instruction",
    "filter_rules",
    "expected_risk",
]

TARGET_FIELDS = [
    "metric_canonical",
    "language",
    "target_label_5",
    "target_count",
    "error_types",
    "notes",
]


ERROR_TYPE_PLAN = {
    "factual_error": {
        "description": "Introduce materially incorrect facts, definitions, formulas, or domain claims.",
        "target_label_range": "1-2",
        "prompt_instruction": "Make the answer plausible but include one or more important factual mistakes aligned with the rubric.",
        "filter_rules": "must not be nonsensical; must not disclose intentional wrongness",
        "expected_risk": "May accidentally produce a correct answer; needs optional judge/human consistency check.",
    },
    "reasoning_gap": {
        "description": "Omit key reasoning steps or use invalid logic while keeping surface structure natural.",
        "target_label_range": "1-3",
        "prompt_instruction": "Create an answer with incomplete or flawed reasoning that a student might write.",
        "filter_rules": "answer must contain enough content to evaluate; no explicit meta commentary",
        "expected_risk": "May be too subtle for label 1; label 2/3 safer.",
    },
    "instruction_violation": {
        "description": "Fail core task requirements such as format, requested fields, or response constraints.",
        "target_label_range": "1-2",
        "prompt_instruction": "Violate one central instruction while keeping the answer fluent and on-topic.",
        "filter_rules": "must still answer the question; no refusal unless source task naturally allows it",
        "expected_risk": "Can create obvious artifacts; filter for naturalness.",
    },
    "scenario_mismatch": {
        "description": "Answer in a way that mismatches the educational scenario, learner profile, tone, or context.",
        "target_label_range": "1-3",
        "prompt_instruction": "Ignore or misuse the scenario context while producing a plausible answer.",
        "filter_rules": "must match source language; no copied original answer",
        "expected_risk": "Might affect multiple metrics; record target metric explicitly.",
    },
    "rubric_violation": {
        "description": "Target the provided rubric by intentionally missing high-value rubric requirements.",
        "target_label_range": "1-2",
        "prompt_instruction": "Use the rubric to omit or violate the criteria needed for a high score.",
        "filter_rules": "metric and rubric must be non-empty",
        "expected_risk": "Requires rubric quality; skip if rubric is missing.",
    },
    "superficial_fluency": {
        "description": "Produce fluent but generic, shallow, or unsupported content that looks polished.",
        "target_label_range": "2-3",
        "prompt_instruction": "Make the answer sound polished but shallow, generic, or weakly grounded.",
        "filter_rules": "must not be too short; must preserve natural answer style",
        "expected_risk": "Often label 3 rather than 1/2; use for boundary negatives.",
    },
    "overconfident_wrong": {
        "description": "State incorrect conclusions with high confidence and no uncertainty.",
        "target_label_range": "1-2",
        "prompt_instruction": "Make the answer confidently wrong without hedging or revealing the flaw.",
        "filter_rules": "must not contain safety disclaimers or meta notes",
        "expected_risk": "Can be too adversarial; keep pedagogically plausible.",
    },
}


def build_error_type_rows() -> list[dict[str, Any]]:
    metric_to_errors = METRIC_ERROR_TYPE_MAP
    rows = []
    for error_type in ERROR_TYPES:
        applicable = sorted(metric for metric, errors in metric_to_errors.items() if error_type in errors)
        spec = ERROR_TYPE_PLAN[error_type]
        rows.append(
            {
                "error_type": error_type,
                "description": spec["description"],
                "target_label_range": spec["target_label_range"],
                "applicable_metrics": applicable,
                "prompt_instruction": spec["prompt_instruction"],
                "filter_rules": spec["filter_rules"],
                "expected_risk": spec["expected_risk"],
            }
        )
    return rows


def build_target_matrix() -> list[dict[str, Any]]:
    train = load_split("train")
    combos = sorted({(row["metric_canonical"], row["language"]) for row in train})
    low_counts = Counter((row["metric_canonical"], row["language"]) for row in train if int(row["label_5"]) <= 2)
    rows = []
    for metric, language in combos:
        for target_label, count in TARGET_LABEL_COUNTS_PER_METRIC_LANGUAGE.items():
            rows.append(
                {
                    "metric_canonical": metric,
                    "language": language,
                    "target_label_5": target_label,
                    "target_count": count,
                    "error_types": METRIC_ERROR_TYPE_MAP.get(metric, []),
                    "notes": f"train low-score count for this metric/language={low_counts.get((metric, language), 0)}; train-only source required",
                }
            )
    return rows


def build_diagnostic_target_matrix() -> list[dict[str, Any]]:
    train = load_split("train")
    combos = sorted({(row["metric_canonical"], row["language"]) for row in train})
    rows = []
    for metric, language in combos:
        for target_label in [1, 2, 3, 4, 5]:
            count = 4 if target_label in {1, 2} else 2
            rows.append(
                {
                    "metric_canonical": metric,
                    "language": language,
                    "target_label_5": target_label,
                    "target_count": count,
                    "error_types": METRIC_ERROR_TYPE_MAP.get(metric, []) if target_label <= 3 else ["high_quality_synthetic_placeholder"],
                    "notes": "Optional D1/D4 synthetic-only diagnostic matrix; not part of the first low-score augmentation generation.",
                }
            )
    return rows


def main() -> None:
    write_table("error_type_plan.csv", build_error_type_rows(), ERROR_TYPE_FIELDS)
    write_table("generation_target_matrix.csv", build_target_matrix(), TARGET_FIELDS)
    write_table("synthetic_only_diagnostic_target_matrix.csv", build_diagnostic_target_matrix(), TARGET_FIELDS)
    print("Wrote error_type_plan.csv, generation_target_matrix.csv, and synthetic_only_diagnostic_target_matrix.csv")


if __name__ == "__main__":
    main()
