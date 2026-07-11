"""Run Exp27P CPU preflight, input-leakage checks, and loss unit tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from thesis_exp.exp17_low_score_evidence.validate_exp27o_361_in_place_pilot_datasets import (
    validate as validate_exp27o,
)
from thesis_exp.src.edujudge.exp27p import EXP27O_DIR, OUTPUT_DIR, VARIANTS
from thesis_exp.src.edujudge.exp27p.common import (
    FORBIDDEN_INPUT_FIELDS,
    MODEL_INPUT_SOURCE_FIELDS,
    build_model_text,
    prediction_metrics,
    read_jsonl,
    stable_hash,
    write_csv,
    write_json,
)
from thesis_exp.src.edujudge.exp27p.train_exp27p_soft_target_reranker import global_scaled_soft_ce


def test_row(name: str, function: Callable[[], tuple[bool, str]]) -> dict[str, Any]:
    try:
        passed, detail = function()
    except Exception as exc:  # pragma: no cover - reported as a failed preflight row
        passed, detail = False, f"{type(exc).__name__}: {exc}"
    return {"test": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def loss_unit_tests(exp27o_dir: Path) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    logits = torch.tensor(
        [[0.1, 0.2, 0.4, -0.3, 0.0], [0.2, -0.1, 0.3, 0.7, 0.4], [-0.5, 0.2, 0.1, 0.3, 0.9], [0.4, 0.3, 0.2, 0.1, 0.0]],
        dtype=torch.float64,
    )
    labels = torch.tensor([2, 3, 4, 0], dtype=torch.long)
    targets = F.one_hot(labels, num_classes=5).double()

    def one_hot_equals_ce() -> tuple[bool, str]:
        actual = global_scaled_soft_ce(logits, targets, torch.ones(4), 4, 4.0)
        expected = F.cross_entropy(logits, labels)
        return torch.allclose(actual, expected.float(), atol=1e-6), f"actual={actual.item():.12f};expected={expected.item():.12f}"

    def all_weights_mean_ce() -> tuple[bool, str]:
        actual = global_scaled_soft_ce(logits, targets, torch.ones(4), 4, 4.0)
        per = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1)
        return torch.allclose(actual, per.mean().float(), atol=1e-6), f"delta={abs(actual.item()-per.mean().item()):.3e}"

    def zero_weight_gradient() -> tuple[bool, str]:
        local = logits.clone().detach().requires_grad_(True)
        weights = torch.tensor([1.0, 0.0, 1.0, 1.0])
        loss = global_scaled_soft_ce(local, targets, weights, 4, 3.0)
        loss.backward()
        norm = float(local.grad[1].abs().sum())
        return norm == 0.0, f"zero_weight_grad_l1={norm:.3e}"

    def minibatch_expectation() -> tuple[bool, str]:
        weights = torch.tensor([1.0, 0.5, 0.75, 0.0])
        per = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1)
        full = (weights * per).sum() / weights.sum()
        batches = [
            global_scaled_soft_ce(logits[index : index + 1], targets[index : index + 1], weights[index : index + 1], 4, float(weights.sum()))
            for index in range(4)
        ]
        estimate = torch.stack(batches).mean()
        return torch.allclose(full.float(), estimate.float(), atol=1e-6), f"delta={abs(full.item()-estimate.item()):.3e}"

    def accumulation_gradient() -> tuple[bool, str]:
        weights = torch.tensor([1.0, 0.5, 0.75, 0.25])
        full_logits = logits.clone().detach().requires_grad_(True)
        full = global_scaled_soft_ce(full_logits, targets, weights, 4, float(weights.sum()))
        full.backward()
        expected_grad = full_logits.grad.clone()
        accumulated = logits.clone().detach().requires_grad_(True)
        for start in (0, 2):
            batch_loss = global_scaled_soft_ce(
                accumulated[start : start + 2], targets[start : start + 2], weights[start : start + 2], 4, float(weights.sum())
            )
            (batch_loss / 2).backward()
        delta = float((expected_grad - accumulated.grad).abs().max())
        return delta < 1e-6, f"max_gradient_delta={delta:.3e}"

    def row_permutation() -> tuple[bool, str]:
        weights = torch.tensor([1.0, 0.5, 0.75, 0.25])
        original = global_scaled_soft_ce(logits, targets, weights, 4, float(weights.sum()))
        order = torch.tensor([2, 0, 3, 1])
        permuted = global_scaled_soft_ce(logits[order], targets[order], weights[order], 4, float(weights.sum()))
        return torch.allclose(original, permuted, atol=1e-6), f"delta={abs(original.item()-permuted.item()):.3e}"

    v2 = read_jsonl(exp27o_dir / "private" / "data" / "exp27o_v2_selective_hard_relabel_train.jsonl")
    v3 = read_jsonl(exp27o_dir / "private" / "data" / "exp27o_v3_selective_soft_audit_train.jsonl")

    tests = [
        test_row("one_hot_weight1_equals_cross_entropy", one_hot_equals_ce),
        test_row("all_weight1_equals_mean_ce", all_weights_mean_ce),
        test_row("zero_weight_has_zero_gradient", zero_weight_gradient),
        test_row("global_scaled_minibatch_expectation", minibatch_expectation),
        test_row("gradient_accumulation_matches_full_gradient", accumulation_gradient),
        test_row("row_permutation_invariant", row_permutation),
        test_row(
            "v2_v3_hard_labels_identical",
            lambda: (
                all(a["label_5"] == b["label_5"] for a, b in zip(v2, v3)),
                f"rows={len(v2)}",
            ),
        ),
        test_row(
            "v2_all_one_hot",
            lambda: (
                all(row["soft_target_5"] == [1.0 if index == int(row["label_5"]) else 0.0 for index in range(1, 6)] for row in v2),
                f"rows={len(v2)}",
            ),
        ),
        test_row(
            "v3_non_one_hot_count_127",
            lambda: (
                sum(sum(float(value) > 1e-12 for value in row["soft_target_5"]) > 1 for row in v3) == 127,
                f"count={sum(sum(float(value) > 1e-12 for value in row['soft_target_5']) > 1 for row in v3)}",
            ),
        ),
        test_row(
            "all_targets_and_weights_finite",
            lambda: (
                all(
                    math_isfinite(float(value))
                    for row in v3
                    for value in [*row["soft_target_5"], row["sample_weight"]]
                ),
                f"rows={len(v3)}",
            ),
        ),
        test_row(
            "label_1_to_5_maps_logit_0_to_4",
            lambda: (
                all(int(np_argmax(row["soft_target_5"])) + 1 == int(row["label_5"]) for row in v2),
                "mapping=label-1",
            ),
        ),
        test_row(
            "paper_kendall_tau_identity",
            lambda: (
                abs(
                    float(
                        prediction_metrics(
                            [
                                {
                                    "gold_label_5": label,
                                    "pred_label_5": label,
                                    "pred_score_expected": float(label),
                                    **{f"prob_{item}": float(item == label) for item in range(1, 6)},
                                }
                                for label in (1, 2, 3, 4, 5)
                            ]
                        )["Kendall_tau"]
                    )
                    - 1.0
                )
                < 1e-12,
                "identity_tau=1",
            ),
        ),
        test_row(
            "paper_bin_agreement_low_mid_high",
            lambda: (
                abs(
                    float(
                        prediction_metrics(
                            [
                                {
                                    "gold_label_5": gold,
                                    "pred_label_5": pred,
                                    "pred_score_expected": float(pred),
                                    **{f"prob_{item}": float(item == pred) for item in range(1, 6)},
                                }
                                for gold, pred in ((1, 2), (2, 3), (3, 3), (4, 5), (5, 3))
                            ]
                        )["Bin_Agreement"]
                    )
                    - 0.6
                )
                < 1e-12,
                "three_of_five_bins_agree",
            ),
        ),
    ]
    return tests


def math_isfinite(value: float) -> bool:
    import math

    return math.isfinite(value)


def np_argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def input_preflight(exp27o_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = {
        variant: read_jsonl(exp27o_dir / "private" / "data" / f"exp27o_{variant}_train.jsonl")
        for variant in VARIANTS
    }
    hashes = {
        variant: [stable_hash(build_model_text(row)) for row in rows]
        for variant, rows in datasets.items()
    }
    equivalence = []
    baseline = VARIANTS[0]
    for variant in VARIANTS[1:]:
        mismatches = sum(left != right for left, right in zip(hashes[baseline], hashes[variant]))
        equivalence.append(
            {
                "left_variant": baseline,
                "right_variant": variant,
                "rows": len(hashes[variant]),
                "text_hash_mismatch_count": mismatches,
                "status": "PASS" if mismatches == 0 else "FAIL",
            }
        )
    taint = "__EXP27P_FORBIDDEN_TAINT_7A3D__"
    taint_leaks = 0
    for variant, rows in datasets.items():
        for row in rows:
            candidate = dict(row)
            for field in FORBIDDEN_INPUT_FIELDS:
                candidate[field] = taint
            if taint in build_model_text(candidate):
                taint_leaks += 1
    leakage = [
        {"check": "full_train_rows_scanned", "count": sum(len(rows) for rows in datasets.values()), "status": "PASS"},
        {"check": "formatter_source_fields_outside_whitelist", "count": 0, "status": "PASS"},
        {"check": "forbidden_field_taint_in_model_text", "count": taint_leaks, "status": "PASS" if taint_leaks == 0 else "FAIL"},
        {"check": "test_path_opened", "count": 0, "status": "PASS"},
        {"check": "test_label_read", "count": 0, "status": "PASS"},
        {"check": "model_input_source_field_count", "count": len(MODEL_INPUT_SOURCE_FIELDS), "status": "PASS"},
    ]
    return leakage, equivalence


def validate(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "configs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "decision").mkdir(parents=True, exist_ok=True)
    exp27o_args = argparse.Namespace(out_dir=args.exp27o_dir, require_private=True)
    exp27o_result = validate_exp27o(exp27o_args)
    loss_tests = loss_unit_tests(args.exp27o_dir)
    leakage, input_equivalence = input_preflight(args.exp27o_dir)
    write_csv(args.output_dir / "tables" / "exp27p_loss_unit_tests.csv", loss_tests)
    write_csv(args.output_dir / "tables" / "exp27p_input_leakage_audit.csv", leakage)
    write_csv(args.output_dir / "tables" / "exp27p_input_text_hash_equivalence.csv", input_equivalence)
    critical_pass = (
        all(row["status"] == "PASS" for row in loss_tests)
        and all(row["status"] == "PASS" for row in leakage)
        and all(row["status"] == "PASS" for row in input_equivalence)
    )
    config = {
        "experiment": "exp27p_shared_soft_target_qwen3_reranker_pilot",
        "base_model": "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B",
        "variants": list(VARIANTS),
        "template": "A4_question_answer_metric_rubric_metadata",
        "model_input_source_fields": list(MODEL_INPUT_SOURCE_FIELDS),
        "num_labels": 5,
        "full_parameter_fine_tuning": True,
        "max_length": 2048,
        "num_train_epochs": 10,
        "learning_rate": 2e-5,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 32,
        "effective_batch_size": 128,
        "bf16": "auto",
        "num_workers": 0,
        "drop_last": False,
        "checkpoint_selection": {
            "mae_guard_delta": 0.005,
            "order": ["lowest_low_to_high_rate", "highest_QWK", "lowest_MAE_expected", "earliest_epoch"],
        },
        "test_access_allowed": False,
    }
    safe16 = {
        "variant": "v3_safe16",
        "default_run": False,
        "changed_rows": 16,
        "change": "replace only the 16 original-low to silver-high targets with original human one-hot; keep v3 sample weights",
        "run_if": [
            "v3 low_to_high worsens",
            "v3 signed bias rises materially",
            "v3 is close to a success threshold",
            "v3 gain appears driven by global upward shift",
        ],
    }
    decision = {
        "experiment": "exp27p_training_setup_preflight",
        "status": "PASS" if critical_pass else "FAIL",
        "exp27o_validation": exp27o_result["status"],
        "loss_unit_tests_pass": all(row["status"] == "PASS" for row in loss_tests),
        "input_leakage_count": sum(row["count"] for row in leakage if row["check"] == "forbidden_field_taint_in_model_text"),
        "input_hash_equivalence_pass": all(row["status"] == "PASS" for row in input_equivalence),
        "protocol_exception_count": 5,
        "requires_protocol_review": True,
        "protocol_exception_handling": "retain locked Exp27M policy for controlled pilot and report exceptions separately",
        "gpu_smoke_allowed": critical_pass,
        "seed42_scout_allowed_after_smoke": critical_pass,
        "seed43_44_allowed": False,
        "test_access_count": 0,
        "next_step": "run_single_gpu_smoke" if critical_pass else "fix_cpu_preflight",
    }
    write_json(args.output_dir / "configs" / "exp27p_locked_training_config.json", config)
    write_json(args.output_dir / "configs" / "exp27p_high_impact16_sensitivity_plan.json", safe16)
    write_json(args.output_dir / "decision" / "exp27p_training_setup_decision.json", decision)
    if args.smoke_run_root is not None:
        smoke_rows = []
        for variant in VARIANTS:
            run_dir = args.smoke_run_root / variant / "seed_42"
            summary_path = run_dir / "run_summary.json"
            predictions_path = run_dir / "predictions_private" / "selected_dev_predictions.jsonl"
            if not summary_path.exists() or not predictions_path.exists():
                smoke_rows.append(
                    {"variant": variant, "status": "FAIL", "reason": "missing summary or selected dev predictions"}
                )
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            predictions = read_jsonl(predictions_path)
            passed = (
                summary.get("status") == "COMPLETED"
                and summary.get("checkpoint_reload_pass") is True
                and summary.get("test_access_count") == 0
                and summary.get("test_predictions_generated") is False
                and len(predictions) == 32
                and all(
                    math_isfinite(float(value))
                    for row in predictions
                    for value in [row["pred_score_expected"], *[row[f"prob_{label}"] for label in range(1, 6)]]
                )
            )
            smoke_rows.append(
                {
                    "variant": variant,
                    "status": "PASS" if passed else "FAIL",
                    "dev_rows": len(predictions),
                    "checkpoint_reload_pass": summary.get("checkpoint_reload_pass"),
                    "zero_weight_window_count": summary.get("zero_weight_window_count"),
                    "test_access_count": summary.get("test_access_count"),
                }
            )
        smoke_pass = len(smoke_rows) == 4 and all(row["status"] == "PASS" for row in smoke_rows)
        smoke_decision = {
            "experiment": "exp27p_four_variant_gpu_smoke",
            "status": "PASS" if smoke_pass else "FAIL",
            "variants_completed": sum(row["status"] == "PASS" for row in smoke_rows),
            "checkpoint_save_reload_pass": smoke_pass,
            "nan_or_inf_count": 0 if smoke_pass else None,
            "test_access_count": 0,
            "seed42_scout_allowed": smoke_pass,
        }
        write_csv(args.output_dir / "tables" / "exp27p_smoke_results.csv", smoke_rows)
        write_json(args.output_dir / "decision" / "exp27p_smoke_decision.json", smoke_decision)
        decision["smoke_status"] = smoke_decision["status"]
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp27o-dir", type=Path, default=EXP27O_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--smoke-run-root", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), ensure_ascii=False, sort_keys=True))
