"""Aggregate the frozen LR=5e-6 train-only SORC-DPO mechanism evidence.

The input row-level diagnostics are private.  This analyzer reads them only to
perform exact within-pair comparisons and writes aggregate statistics without
record IDs, pair IDs, text, or token sequences.  It never reads dev or test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Callable

from thesis_exp.exp54_rar_sft import REPO_ROOT


SEEDS = (42, 43, 44)
ARMS = ("P1_FIELD_DPO", "P2_SORC_SCORE", "P3_JOINT_SORC")
SCORE_PAIR_TYPES = ("adjacent_score", "severe_l2h", "h2l_guard")
DEFAULT_INPUT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_lr5e6_followup/train_signal_diagnostics"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_lr5e6_followup/mechanism_diagnostic"
)
NUMERIC_FIELDS = (
    "beta_scaled_contrast",
    "risk_adjusted_margin",
    "reference_raw_margin",
    "policy_raw_margin",
    "chosen_logp_change",
    "rejected_logp_change",
)
RATE_FIELDS = (
    "contrast_positive",
    "offset_satisfied",
    "policy_prefers_chosen",
    "chosen_logp_increased",
    "rejected_logp_decreased",
)
PAIR_IDENTITY_FIELDS = (
    "pair_task",
    "pair_type",
    "pair_source",
    "gold_label",
    "rejected_score",
    "metric_id",
    "language",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: empty diagnostic")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _describe_values(values: list[float]) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("descriptive statistic values are empty or non-finite")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": _quantile(values, 0.10),
        "p90": _quantile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty group")
    return {
        "count": len(rows),
        "numeric": {
            field: _describe_values([float(row[field]) for row in rows])
            for field in NUMERIC_FIELDS
        },
        "rates": {
            field: statistics.fmean(bool(row[field]) for row in rows)
            for field in RATE_FIELDS
        },
    }


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_rows = [row for row in rows if row["pair_task"] == "score"]
    rationale_rows = [
        row for row in rows if row["pair_task"] == "rationale"
    ]
    output: dict[str, Any] = {
        "all": summarize_group(rows),
        "score": summarize_group(score_rows),
        "score_by_pair_type": {
            pair_type: summarize_group(
                [row for row in score_rows if row["pair_type"] == pair_type]
            )
            for pair_type in SCORE_PAIR_TYPES
        },
        "score_by_source": {
            source: summarize_group(
                [row for row in score_rows if row["pair_source"] == source]
            )
            for source in sorted({str(row["pair_source"]) for row in score_rows})
        },
        "low_score_labels": {
            str(label): summarize_group(
                [row for row in score_rows if int(row["gold_label"]) == label]
            )
            for label in (1, 2)
        },
    }
    if rationale_rows:
        output["rationale"] = summarize_group(rationale_rows)
    return output


def _index_score_rows(
    rows: list[dict[str, Any]], arm: str
) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        if row["arm"] != arm or row["pair_task"] != "score":
            continue
        pair_id = str(row["pair_id"])
        if pair_id in output:
            raise ValueError(f"{arm}: duplicate score pair")
        output[pair_id] = row
    if not output:
        raise ValueError(f"{arm}: no score pairs")
    return output


def paired_arm_delta(
    rows: list[dict[str, Any]],
    *,
    left_arm: str,
    right_arm: str,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    left = _index_score_rows(rows, left_arm)
    right = _index_score_rows(rows, right_arm)
    if set(left) != set(right):
        raise ValueError(f"{left_arm}/{right_arm}: score inventories differ")
    deltas: dict[str, list[float]] = {
        "beta_scaled_contrast": [],
        "policy_raw_margin": [],
        "chosen_logp_change": [],
        "rejected_logp_change": [],
    }
    for pair_id in sorted(left):
        first = left[pair_id]
        second = right[pair_id]
        if any(first[field] != second[field] for field in PAIR_IDENTITY_FIELDS):
            raise ValueError(
                f"{left_arm}/{right_arm}: paired metadata differs"
            )
        if predicate is not None and not predicate(first):
            continue
        for field in deltas:
            deltas[field].append(float(second[field]) - float(first[field]))
    if not deltas["beta_scaled_contrast"]:
        raise ValueError("paired arm comparison selected no rows")
    primary = deltas["beta_scaled_contrast"]
    return {
        "comparison": f"{right_arm}_minus_{left_arm}",
        "count": len(primary),
        "delta": {
            field: _describe_values(values)
            for field, values in deltas.items()
        },
        "beta_scaled_contrast_delta_direction": {
            "positive_rate": statistics.fmean(value > 0.0 for value in primary),
            "zero_rate": statistics.fmean(value == 0.0 for value in primary),
            "negative_rate": statistics.fmean(value < 0.0 for value in primary),
        },
    }


def _paired_suite(
    rows: list[dict[str, Any]], left_arm: str, right_arm: str
) -> dict[str, Any]:
    return {
        "all_score_pairs": paired_arm_delta(
            rows, left_arm=left_arm, right_arm=right_arm
        ),
        "by_pair_type": {
            pair_type: paired_arm_delta(
                rows,
                left_arm=left_arm,
                right_arm=right_arm,
                predicate=lambda row, expected=pair_type: (
                    row["pair_type"] == expected
                ),
            )
            for pair_type in SCORE_PAIR_TYPES
        },
        "by_source": {
            source: paired_arm_delta(
                rows,
                left_arm=left_arm,
                right_arm=right_arm,
                predicate=lambda row, expected=source: (
                    row["pair_source"] == expected
                ),
            )
            for source in sorted(
                {
                    str(row["pair_source"])
                    for row in rows
                    if row["arm"] == left_arm and row["pair_task"] == "score"
                }
            )
        },
        "low_score_labels": {
            str(label): paired_arm_delta(
                rows,
                left_arm=left_arm,
                right_arm=right_arm,
                predicate=lambda row, expected=label: (
                    int(row["gold_label"]) == expected
                ),
            )
            for label in (1, 2)
        },
    }


def analyze(input_root: Path) -> dict[str, Any]:
    per_seed = {}
    source_hashes = {}
    for seed in SEEDS:
        seed_root = input_root / f"seed_{seed}"
        aggregate_path = seed_root / "aggregate_report.json"
        private_path = seed_root / "private/pair_diagnostics.jsonl"
        aggregate = _read_json(aggregate_path)
        if (
            aggregate.get("status")
            != "SORC_DPO_TRAIN_SIGNAL_DIAGNOSTIC_COMPLETE"
            or int(aggregate.get("seed", -1)) != seed
            or float(aggregate.get("beta", -1.0)) != 0.1
            or aggregate.get("dev_accessed") is not False
            or aggregate.get("test_accessed") is not False
        ):
            raise ValueError(f"seed {seed}: aggregate contract differs")
        private_hash = _sha256(private_path)
        if (
            aggregate["source_hashes"]["private_pair_diagnostics"]
            != private_hash
        ):
            raise ValueError(f"seed {seed}: private diagnostic hash differs")
        rows = _read_jsonl(private_path)
        if {str(row["arm"]) for row in rows} != set(ARMS):
            raise ValueError(f"seed {seed}: arm inventory differs")
        for arm in ARMS:
            expected = int(aggregate["arms"][arm]["pair_count"])
            actual = sum(row["arm"] == arm for row in rows)
            if actual != expected:
                raise ValueError(f"seed {seed}/{arm}: row count differs")
        per_seed[str(seed)] = {
            "arms": {
                arm: summarize_arm(
                    [row for row in rows if row["arm"] == arm]
                )
                for arm in ARMS
            },
            "paired_score_comparisons": {
                "P2_minus_P1": _paired_suite(
                    rows, "P1_FIELD_DPO", "P2_SORC_SCORE"
                ),
                "P3_minus_P2": _paired_suite(
                    rows, "P2_SORC_SCORE", "P3_JOINT_SORC"
                ),
            },
            "adapter_update": {
                arm: aggregate["arms"][arm]["adapter_update"]["global"]
                for arm in ARMS
            },
        }
        source_hashes[f"seed_{seed}_aggregate"] = _sha256(aggregate_path)
        source_hashes[f"seed_{seed}_private"] = private_hash

    def seed_values(path: tuple[str, ...]) -> list[float]:
        values = []
        for seed in map(str, SEEDS):
            value: Any = per_seed[seed]
            for key in path:
                value = value[key]
            values.append(float(value))
        return values

    p2_minus_p1 = {
        pair_type: seed_values(
            (
                "paired_score_comparisons",
                "P2_minus_P1",
                "by_pair_type",
                pair_type,
                "delta",
                "beta_scaled_contrast",
                "mean",
            )
        )
        for pair_type in SCORE_PAIR_TYPES
    }
    p2_satisfaction = seed_values(
        (
            "arms",
            "P2_SORC_SCORE",
            "score",
            "rates",
            "offset_satisfied",
        )
    )
    p3_rationale_positive = seed_values(
        (
            "arms",
            "P3_JOINT_SORC",
            "rationale",
            "rates",
            "contrast_positive",
        )
    )
    mechanism_checks = {
        "p2_strict_offset_satisfaction_zero_all_seeds": all(
            value == 0.0 for value in p2_satisfaction
        ),
        "p2_minus_p1_severe_l2h_beta_margin_positive_all_seeds": all(
            value > 0.0 for value in p2_minus_p1["severe_l2h"]
        ),
        "p2_minus_p1_h2l_guard_beta_margin_negative_all_seeds": all(
            value < 0.0 for value in p2_minus_p1["h2l_guard"]
        ),
        "p3_rationale_contrast_positive_rate_at_least_0_89_all_seeds": all(
            value >= 0.89 for value in p3_rationale_positive
        ),
    }
    return {
        "schema_version": "exp54-sorc-dpo-lr5e6-mechanism-v1",
        "status": "SORC_DPO_LR5E6_MECHANISM_DIAGNOSTIC_COMPLETE",
        "beta": 0.1,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "mechanism_checks": mechanism_checks,
        "three_seed_summary": {
            "P2_minus_P1_mean_beta_margin_delta_by_pair_type": {
                pair_type: {
                    "per_seed": values,
                    "mean_across_seeds": statistics.fmean(values),
                }
                for pair_type, values in p2_minus_p1.items()
            },
            "P2_score_strict_offset_satisfaction_rate": {
                "per_seed": p2_satisfaction,
                "mean_across_seeds": statistics.fmean(p2_satisfaction),
            },
            "P3_rationale_contrast_positive_rate": {
                "per_seed": p3_rationale_positive,
                "mean_across_seeds": statistics.fmean(
                    p3_rationale_positive
                ),
            },
        },
        "interpretation_boundary": {
            "supported": (
                "the ordinal offset produced a consistent relative increase "
                "in severe-L2H train margin versus P1"
            ),
            "not_supported": (
                "the prescribed offsets were reached or ODPO improved dev "
                "independently of Field-DPO"
            ),
            "tradeoff_observed": (
                "P2 simultaneously reduced the H2L-guard train margin versus "
                "P1, so risk pressure was asymmetric rather than uniformly "
                "stronger"
            ),
        },
        "source_hashes": source_hashes,
        "row_level_values_public": False,
        "dev_accessed": False,
        "test_accessed": False,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["three_seed_summary"]
    lines = [
        "# SORC-DPO LR=5e-6 train-only mechanism diagnosis",
        "",
        "This report uses only frozen train-pair diagnostics. It does not read "
        "dev or test and publishes no row-level identifiers or text.",
        "",
        "## P2 ordinal-offset effect relative to P1",
        "",
        "| Score block | Seed 42 | Seed 43 | Seed 44 | Mean |",
        "|---|---:|---:|---:|---:|",
    ]
    values_by_type = summary[
        "P2_minus_P1_mean_beta_margin_delta_by_pair_type"
    ]
    for pair_type in SCORE_PAIR_TYPES:
        values = values_by_type[pair_type]
        per_seed = values["per_seed"]
        lines.append(
            f"| {pair_type} | {per_seed[0]:+.6f} | {per_seed[1]:+.6f} | "
            f"{per_seed[2]:+.6f} | {values['mean_across_seeds']:+.6f} |"
        )
    satisfaction = summary[
        "P2_score_strict_offset_satisfaction_rate"
    ]["mean_across_seeds"]
    rationale = summary["P3_rationale_contrast_positive_rate"][
        "mean_across_seeds"
    ]
    lines.extend(
        [
            "",
            f"- P2 strict offset satisfaction: {100 * satisfaction:.2f}%.",
            "- Severe-L2H P2−P1 beta-margin delta was positive in all seeds.",
            "- H2L-guard P2−P1 beta-margin delta was negative in all seeds.",
            f"- P3 rationale contrast-positive rate: {100 * rationale:.2f}%.",
            "",
            "## Interpretation",
            "",
            "The ODPO offset was not inert: it consistently redirected train "
            "margin toward the severe low-to-high block. However, no score "
            "pair reached its full prescribed offset, the high-score guard "
            "margin weakened relative to P1, and dev did not establish an "
            "independent P2-over-P1 benefit. This is mechanism evidence for "
            "risk-conditioned pressure, not confirmatory evidence that the "
            "ordinal offset improved generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = analyze(args.input_root)
    report_path = args.output_root / "mechanism_report.json"
    markdown_path = args.output_root / "report.md"
    _atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(markdown_path, _markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "mechanism_checks": report["mechanism_checks"],
                "report_sha256": _sha256(report_path),
                "dev_accessed": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
