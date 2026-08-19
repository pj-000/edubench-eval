"""Collect three train-only SORC-DPO signal diagnostics into a public report."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file
from thesis_exp.exp54_rar_sft.diagnose_sorc_dpo_train_signal import ARMS


DEFAULT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_train_signal_diagnostics"
)
SEEDS = (42, 43, 44)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _mean(values: list[float]) -> float:
    if len(values) != len(SEEDS):
        raise ValueError("multi-seed statistic is incomplete")
    return statistics.fmean(values)


def collect(root: Path) -> dict[str, Any]:
    reports = {}
    for seed in SEEDS:
        path = root / f"seed_{seed}/aggregate_report.json"
        report = _read_json(path)
        if (
            report.get("status")
            != "SORC_DPO_TRAIN_SIGNAL_DIAGNOSTIC_COMPLETE"
            or int(report.get("seed", -1)) != seed
            or report.get("dev_accessed") is not False
            or report.get("test_accessed") is not False
        ):
            raise ValueError(f"seed {seed}: diagnostic report differs")
        if set(report["arms"]) != set(ARMS):
            raise ValueError(f"seed {seed}: arm inventory differs")
        reports[seed] = report

    arms = {}
    for arm in ARMS:
        overall = {
            seed: reports[seed]["arms"][arm]["aggregates"]["overall"]
            for seed in SEEDS
        }
        score_task = {
            seed: reports[seed]["arms"][arm]["aggregates"]["by"][
                "pair_task"
            ].get("score")
            for seed in SEEDS
        }
        rationale_task = {
            seed: reports[seed]["arms"][arm]["aggregates"]["by"][
                "pair_task"
            ].get("rationale")
            for seed in SEEDS
        }
        arm_result = {
            "pair_count_per_seed": int(overall[42]["count"]),
            "preference_contrast_mean": _mean(
                [
                    float(
                        overall[seed]["numeric"]["preference_contrast"][
                            "mean"
                        ]
                    )
                    for seed in SEEDS
                ]
            ),
            "beta_scaled_contrast_mean": _mean(
                [
                    float(
                        overall[seed]["numeric"]["beta_scaled_contrast"][
                            "mean"
                        ]
                    )
                    for seed in SEEDS
                ]
            ),
            "contrast_positive_rate": _mean(
                [
                    float(overall[seed]["rates"]["contrast_positive"])
                    for seed in SEEDS
                ]
            ),
            "chosen_logp_increased_rate": _mean(
                [
                    float(
                        overall[seed]["rates"]["chosen_logp_increased"]
                    )
                    for seed in SEEDS
                ]
            ),
            "rejected_logp_decreased_rate": _mean(
                [
                    float(
                        overall[seed]["rates"]["rejected_logp_decreased"]
                    )
                    for seed in SEEDS
                ]
            ),
            "relative_adapter_update_l2": _mean(
                [
                    float(
                        reports[seed]["arms"][arm]["adapter_update"][
                            "global"
                        ]["relative_delta_l2"]
                    )
                    for seed in SEEDS
                ]
            ),
            "per_seed": {},
        }
        for seed in SEEDS:
            score = overall[seed]["score_decision"]
            arm_result["per_seed"][str(seed)] = {
                "preference_contrast_mean": float(
                    overall[seed]["numeric"]["preference_contrast"]["mean"]
                ),
                "preference_contrast_median": float(
                    overall[seed]["numeric"]["preference_contrast"]["median"]
                ),
                "beta_scaled_contrast_mean": float(
                    overall[seed]["numeric"]["beta_scaled_contrast"]["mean"]
                ),
                "offset_satisfied_rate": float(
                    overall[seed]["rates"]["offset_satisfied"]
                ),
                "reference_score_accuracy": (
                    None
                    if score is None
                    else float(score["reference_gold_accuracy"])
                ),
                "policy_score_accuracy": (
                    None
                    if score is None
                    else float(score["policy_gold_accuracy"])
                ),
                "relative_adapter_update_l2": float(
                    reports[seed]["arms"][arm]["adapter_update"]["global"][
                        "relative_delta_l2"
                    ]
                ),
            }
        if all(value is not None for value in score_task.values()):
            arm_result["score_task"] = {
                "preference_contrast_mean": _mean(
                    [
                        float(
                            score_task[seed]["numeric"][
                                "preference_contrast"
                            ]["mean"]
                        )
                        for seed in SEEDS
                    ]
                ),
                "offset_satisfied_rate": _mean(
                    [
                        float(
                            score_task[seed]["rates"]["offset_satisfied"]
                        )
                        for seed in SEEDS
                    ]
                ),
            }
        if all(value is not None for value in rationale_task.values()):
            arm_result["rationale_task"] = {
                "preference_contrast_mean": _mean(
                    [
                        float(
                            rationale_task[seed]["numeric"][
                                "preference_contrast"
                            ]["mean"]
                        )
                        for seed in SEEDS
                    ]
                ),
                "contrast_positive_rate": _mean(
                    [
                        float(
                            rationale_task[seed]["rates"][
                                "contrast_positive"
                            ]
                        )
                        for seed in SEEDS
                    ]
                ),
            }
        arms[arm] = arm_result

    p2_score_rates = [
        float(
            reports[seed]["arms"]["P2_SORC_SCORE"]["aggregates"]["by"][
                "pair_task"
            ]["score"]["rates"]["offset_satisfied"]
        )
        for seed in SEEDS
    ]
    relative_updates = [
        float(
            reports[seed]["arms"][arm]["adapter_update"]["global"][
                "relative_delta_l2"
            ]
        )
        for seed in SEEDS
        for arm in ARMS
    ]
    medians = [
        float(
            reports[seed]["arms"][arm]["aggregates"]["overall"]["numeric"][
                "preference_contrast"
            ]["median"]
        )
        for seed in SEEDS
        for arm in ARMS
    ]
    trigger = {
        "all_p2_score_offset_satisfied_rates_zero": all(
            value == 0.0 for value in p2_score_rates
        ),
        "maximum_relative_adapter_update_l2_below_0_001": (
            max(relative_updates) < 0.001
        ),
        "all_overall_preference_contrast_medians_zero": all(
            value == 0.0 for value in medians
        ),
    }
    trigger["lr_only_followup_trigger_satisfied"] = all(trigger.values())
    return {
        "schema_version": "exp54-sorc-dpo-train-signal-multiseed-v1",
        "status": "SORC_DPO_TRAIN_SIGNAL_MULTISEED_COMPLETE",
        "interpretation": (
            "optimization_dose_insufficient_under_original_5e-7_27_step_run"
        ),
        "formal_null_result_preserved": True,
        "arms": arms,
        "lr_only_followup_trigger": trigger,
        "allowed_followup": {
            "learning_rate_from": 5e-7,
            "learning_rate_to": 5e-6,
            "all_other_scientific_variables_fixed": True,
            "seed42_scout_first": True,
            "seeds43_44_allowed_only_after_scout_gate": True,
        },
        "source_hashes": {
            f"seed_{seed}_aggregate_report": sha256_file(
                root / f"seed_{seed}/aggregate_report.json"
            )
            for seed in SEEDS
        },
        "dev_accessed": False,
        "test_accessed": False,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SORC-DPO train-only signal diagnosis",
        "",
        "| Arm | Mean raw margin change | Mean β-scaled change | "
        "Relative LoRA update |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = report["arms"][arm]
        lines.append(
            f"| {arm} | {value['preference_contrast_mean']:.6f} | "
            f"{value['beta_scaled_contrast_mean']:.6f} | "
            f"{100 * value['relative_adapter_update_l2']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "All three P2 score runs had zero examples whose β-scaled "
            "policy-reference contrast exceeded the frozen ordinal offset.",
            "",
            "Decision: preserve the original formal null result and run only "
            "the preregistered LR-only exploratory follow-up (5e-7 → 5e-6), "
            "starting with seed 42. No dev/test data were read.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    report = collect(args.root)
    report_path = args.root / "multiseed_report.json"
    markdown_path = args.root / "report.md"
    _atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(markdown_path, _markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "lr_only_followup_trigger_satisfied": report[
                    "lr_only_followup_trigger"
                ]["lr_only_followup_trigger_satisfied"],
                "report_sha256": sha256_file(report_path),
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
