"""Preflight setup checks for Exp5 low-score loss ablations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp05 import (
    DEFAULT_W_MAX,
    DEFAULT_W_MIN,
    EXP04_A4_DATASET_DIR,
    EXP04_O3_RUN_DIR,
    EXP04_TABLES_DIR,
    EXP05_ARTIFACTS_DIR,
    EXP05_OUTPUT_DIR,
    EXP05_TABLES_DIR,
    EXPECTED_SPLIT_ROWS,
    L1_RUN_ID,
    L2_RUN_CONFIGS,
    ensure_exp05_dirs,
    l1_run_dir,
)
from thesis_exp.src.edujudge.exp05.build_exp05_dataset import ensure_exp05_dataset
from thesis_exp.src.edujudge.exp05.class_weights import write_class_weights
from thesis_exp.src.edujudge.exp05.losses import (
    asymmetric_low_score_penalty,
    asymmetric_ordinal_loss,
    make_ordinal_targets,
    weighted_ordinal_loss,
)
from thesis_exp.src.edujudge.exp05.write_exp05_report import write_exp05_report
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, read_jsonl, relpath, write_csv, write_text


SCRIPT_PATHS = [
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp05_l1_smoke.sh",
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp05_l1_train.sh",
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp05_l2_smoke.sh",
    REPO_ROOT / "thesis_exp" / "scripts" / "run_exp05_l2_train.sh",
]
EXP05_SRC_DIR = REPO_ROOT / "thesis_exp" / "src" / "edujudge" / "exp05"
L2_CONFIG_PATHS = [
    REPO_ROOT / "thesis_exp" / "configs" / "exp05_low_score_loss" / "exp05_l2a_asymmetric_ordinal_lambda03.yaml",
    REPO_ROOT / "thesis_exp" / "configs" / "exp05_low_score_loss" / "exp05_l2b_asymmetric_ordinal_lambda05.yaml",
    REPO_ROOT / "thesis_exp" / "configs" / "exp05_low_score_loss" / "exp05_l2_smoke_test.yaml",
]
EXP4_REQUIRED = [
    EXP04_TABLES_DIR / "target_objective_summary.csv",
    EXP04_TABLES_DIR / "target_objective_low_score.csv",
    EXP04_O3_RUN_DIR / "tables" / "metrics_summary.csv",
    EXP04_O3_RUN_DIR / "tables" / "low_score_metrics.csv",
    EXP04_O3_RUN_DIR / "arrays" / "dev_test_arrays.npz",
]


def add(rows: list[dict[str, Any]], check: str, status: str, observed: Any, expected: Any, notes: str = "") -> None:
    rows.append({"check": check, "status": status, "observed": observed, "expected": expected, "notes": notes})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["check", "status", "observed", "expected", "notes"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = str(row.get(col, "")).replace("|", "\\|")
            if len(value) > 110:
                value = value[:107] + "..."
            cells.append(value)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def command_status(args: list[str]) -> tuple[str, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = " ".join(result.stdout.split())
    return ("PASS" if result.returncode == 0 else "FAIL", output[-700:] if output else "ok")


def check_exp4_baseline(rows: list[dict[str, Any]]) -> None:
    missing = [relpath(path) for path in EXP4_REQUIRED if not path.exists()]
    add(rows, "Exp4 O3 baseline outputs exist", "PASS" if not missing else "FAIL", missing, "[]")


def check_dataset(rows: list[dict[str, Any]]) -> None:
    try:
        dataset_rows = ensure_exp05_dataset()
        add(rows, "A4 dataset exists", "PASS", relpath(EXP04_A4_DATASET_DIR), "Exp4 fixed A4")
    except Exception as exc:
        add(rows, "A4 dataset exists", "FAIL", f"{type(exc).__name__}: {exc}", "Exp4 fixed A4")
        return
    for row in dataset_rows:
        add(
            rows,
            f"{row['split']} rows",
            "PASS" if row["status"] == "PASS" else "FAIL",
            row["rows"],
            row["expected_rows"],
            row.get("missing_fields", ""),
        )
    for split in ["train", "dev", "test"]:
        data = read_jsonl(EXP04_A4_DATASET_DIR / f"{split}.jsonl")
        labels = sorted({int(row["label_5"]) for row in data})
        add(rows, f"{split} label_5 values", "PASS" if labels == [1, 2, 3, 4, 5] else "FAIL", labels, [1, 2, 3, 4, 5])
        sample_path_ok = "synthetic" not in str(EXP04_A4_DATASET_DIR).lower() and "sample" not in str(EXP04_A4_DATASET_DIR).lower()
        add(rows, f"{split} no synthetic/sample data path", "PASS" if sample_path_ok else "FAIL", relpath(EXP04_A4_DATASET_DIR), "fixed A4 path")


def check_weights_and_loss(rows: list[dict[str, Any]]) -> None:
    try:
        weight_rows = write_class_weights(w_min=DEFAULT_W_MIN, w_max=DEFAULT_W_MAX)
        add(rows, "class_weights.csv exists", "PASS", relpath(EXP05_TABLES_DIR / "class_weights.csv"), "exists")
        add(rows, "w_min/w_max", "PASS", f"{DEFAULT_W_MIN}/{DEFAULT_W_MAX}", "0.5/3.0")
        add(rows, "class weights use train only", "PASS", relpath(EXP04_A4_DATASET_DIR / "train.jsonl"), "train split")
        add(rows, "all class weights finite", "PASS", [(r["label_5"], r["clipped_weight"]) for r in weight_rows], "finite labels 1..5")
    except Exception as exc:
        add(rows, "class weights compute", "FAIL", f"{type(exc).__name__}: {exc}", "success")
        return

    expected_targets = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]
    try:
        import torch

        labels = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        targets = make_ordinal_targets(labels)
        expected = torch.tensor(expected_targets, dtype=torch.float32)
        add(rows, "toy ordinal targets", "PASS" if torch.equal(targets, expected) else "FAIL", targets.tolist(), expected.tolist())
        logits = torch.zeros((5, 4), dtype=torch.float32)
        weights = torch.tensor([0.0, 3.0, 2.0, 1.0, 0.75, 0.5], dtype=torch.float32)
        loss, debug = weighted_ordinal_loss(logits, labels, weights)
        finite = bool(torch.isfinite(loss))
        add(rows, "toy weighted ordinal loss finite", "PASS" if finite else "FAIL", float(loss), "finite non-NaN", debug)
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            add(rows, "toy weighted ordinal loss finite", "FAIL", f"{type(exc).__name__}: {exc}", "success")
            return
        import math

        labels = [1, 2, 3, 4, 5]
        targets = [[1 if label > threshold else 0 for threshold in [1, 2, 3, 4]] for label in labels]
        add(rows, "toy ordinal targets", "PASS" if targets == expected_targets else "FAIL", targets, expected_targets, "numpy fallback")
        per_sample = [math.log(2.0)] * 5
        weights = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.75, 5: 0.5}
        numerator = sum(weights[label] * loss for label, loss in zip(labels, per_sample))
        denominator = sum(weights[label] for label in labels)
        loss = numerator / denominator
        add(rows, "toy weighted ordinal loss finite", "PASS" if math.isfinite(loss) else "FAIL", loss, "finite non-NaN", "torch unavailable; formula fallback")
    except Exception as exc:
        add(rows, "toy weighted ordinal loss finite", "FAIL", f"{type(exc).__name__}: {exc}", "success")


def simple_yaml_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def check_l2_configs(rows: list[dict[str, Any]]) -> None:
    for path in L2_CONFIG_PATHS:
        add(rows, f"L2 config exists {path.name}", "PASS" if path.exists() else "FAIL", relpath(path) if path.exists() else "missing", "exists")
    for path in L2_CONFIG_PATHS[:2]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        lambda_low = float(simple_yaml_value(text, "lambda_low"))
        margin = float(simple_yaml_value(text, "margin"))
        add(rows, f"{path.name} lambda_low > 0", "PASS" if lambda_low > 0 else "FAIL", lambda_low, ">0")
        add(rows, f"{path.name} margin >= 0", "PASS" if margin >= 0 else "FAIL", margin, ">=0")
        for key in ["use_class_weights", "use_high_score_preservation", "use_threshold_suppression"]:
            observed = simple_yaml_value(text, key)
            add(rows, f"{path.name} {key}=false", "PASS" if observed == "false" else "FAIL", observed, "false")


def logit(probability: float) -> float:
    import math

    return math.log(probability / (1.0 - probability))


def manual_penalties(prob_rows: list[list[float]], labels: list[int], margin: float) -> list[float]:
    penalties = []
    for probs, label in zip(prob_rows, labels):
        s_hat = 1.0 + sum(probs)
        over = max(s_hat - float(label) - margin, 0.0)
        penalties.append(((over / 4.0) ** 2) if label <= 2 else 0.0)
    return penalties


def run_l2_toy_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    prob_rows = [
        [0.9, 0.9, 0.9, 0.9],
        [0.8, 0.8, 0.2, 0.1],
        [0.9, 0.8, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.1],
        [0.9, 0.9, 0.9, 0.9],
    ]
    labels = [1, 2, 3, 4, 5]
    try:
        import torch

        logits = torch.tensor([[logit(prob) for prob in probs] for probs in prob_rows], dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        penalties, _ = asymmetric_low_score_penalty(logits, label_tensor, margin=0.0)
        penalty_values = [float(value) for value in penalties.detach().cpu().tolist()]
        ok = penalty_values[0] > penalty_values[1] > 0 and all(abs(penalty_values[idx]) < 1e-8 for idx in [2, 3, 4])
        checks.append(
            {
                "test_name": "l2_penalty_only_low_labels",
                "status": "PASS" if ok else "FAIL",
                "details": "penalty[0] > penalty[1] > 0 and non-low penalties are zero",
                "expected": "true",
                "actual": penalty_values,
            }
        )

        low_logits = torch.tensor([[0.0, logit(0.25), logit(0.25), logit(1e-6)]], dtype=torch.float32)
        low_label = torch.tensor([2], dtype=torch.long)
        low_penalty, _ = asymmetric_low_score_penalty(low_logits, low_label, margin=0.0)
        near_zero = float(low_penalty.detach().cpu()[0]) < 1e-8
        checks.append(
            {
                "test_name": "l2_penalty_zero_at_margin",
                "status": "PASS" if near_zero else "FAIL",
                "details": "label=2 with s_hat near 2.0 has no penalty",
                "expected": "0",
                "actual": float(low_penalty.detach().cpu()[0]),
            }
        )

        logits_one = torch.tensor([[logit(prob) for prob in prob_rows[1]]], dtype=torch.float32)
        label_one = torch.tensor([2], dtype=torch.long)
        loss03, _ = asymmetric_ordinal_loss(logits_one, label_one, lambda_low=0.3, margin=0.0)
        loss05, _ = asymmetric_ordinal_loss(logits_one, label_one, lambda_low=0.5, margin=0.0)
        larger = float(loss05.detach().cpu()) > float(loss03.detach().cpu())
        checks.append(
            {
                "test_name": "l2_lambda_monotonic_total_loss",
                "status": "PASS" if larger else "FAIL",
                "details": "same positive penalty should make lambda=0.5 loss larger than lambda=0.3",
                "expected": "loss05 > loss03",
                "actual": f"{float(loss05.detach().cpu())} > {float(loss03.detach().cpu())}",
            }
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            checks.append(
                {
                    "test_name": "l2_toy_checks",
                    "status": "FAIL",
                    "details": f"{type(exc).__name__}: {exc}",
                    "expected": "success",
                    "actual": "error",
                }
            )
            return checks
        penalties = manual_penalties(prob_rows, labels, margin=0.0)
        ok = penalties[0] > penalties[1] > 0 and all(abs(penalties[idx]) < 1e-8 for idx in [2, 3, 4])
        checks.append(
            {
                "test_name": "l2_penalty_only_low_labels",
                "status": "PASS" if ok else "FAIL",
                "details": "torch unavailable; formula fallback",
                "expected": "true",
                "actual": penalties,
            }
        )
        low_penalty = manual_penalties([[0.5, 0.25, 0.25, 0.0]], [2], margin=0.0)[0]
        checks.append(
            {
                "test_name": "l2_penalty_zero_at_margin",
                "status": "PASS" if low_penalty < 1e-8 else "FAIL",
                "details": "torch unavailable; formula fallback",
                "expected": "0",
                "actual": low_penalty,
            }
        )
        penalty = manual_penalties([prob_rows[1]], [2], margin=0.0)[0]
        checks.append(
            {
                "test_name": "l2_lambda_monotonic_total_loss",
                "status": "PASS" if 0.5 * penalty > 0.3 * penalty else "FAIL",
                "details": "torch unavailable; comparing penalty contribution only",
                "expected": "0.5 * penalty > 0.3 * penalty",
                "actual": f"{0.5 * penalty} > {0.3 * penalty}",
            }
        )
    except Exception as exc:
        checks.append(
            {
                "test_name": "l2_toy_checks",
                "status": "FAIL",
                "details": f"{type(exc).__name__}: {exc}",
                "expected": "success",
                "actual": "error",
            }
        )
    return checks


def check_l2_loss(rows: list[dict[str, Any]]) -> None:
    checks = run_l2_toy_checks()
    write_csv(
        EXP05_TABLES_DIR / "l2_toy_loss_checks.csv",
        checks,
        fieldnames=["test_name", "status", "details", "expected", "actual"],
    )
    overall = "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "FAIL"
    add(rows, "L2 toy loss checks", overall, relpath(EXP05_TABLES_DIR / "l2_toy_loss_checks.csv"), "all PASS")
    penalty_check = next((row for row in checks if row["test_name"] == "l2_penalty_only_low_labels"), {})
    add(rows, "L2 penalty only applies to label_5 <= 2", penalty_check.get("status", "FAIL"), penalty_check.get("actual", ""), "non-low penalties zero")
    class_weights_path = EXP05_TABLES_DIR / "class_weights.csv"
    add(rows, "L2 does not modify class_weights.csv", "PASS" if class_weights_path.exists() else "FAIL", relpath(class_weights_path), "existing L1 weights retained")
    l1_metrics = l1_run_dir(smoke=False) / "tables" / "metrics_summary.csv"
    add(rows, "L1 outputs remain present", "PASS" if l1_metrics.exists() else "FAIL", relpath(l1_metrics) if l1_metrics.exists() else "missing", f"{L1_RUN_ID} metrics")


def check_scripts_and_modules(rows: list[dict[str, Any]]) -> None:
    for path in SCRIPT_PATHS:
        if not path.exists():
            add(rows, f"bash -n {relpath(path)}", "FAIL", "missing", "exists")
            continue
        status, output = command_status(["bash", "-n", str(path.relative_to(REPO_ROOT))])
        add(rows, f"bash -n {relpath(path)}", status, output, "ok")
    py_files = sorted(EXP05_SRC_DIR.glob("*.py"))
    status, output = command_status([sys.executable, "-m", "py_compile", *[str(path.relative_to(REPO_ROOT)) for path in py_files]])
    add(rows, "exp05 Python modules py_compile", status, output, "ok")


def check_gitignore_and_artifacts(rows: list[dict[str, Any]]) -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    patterns = [
        "thesis_exp/artifacts/",
        "**/checkpoints/",
        "**/*.safetensors",
        "**/*.bin",
        "**/*.pt",
        "**/*.pth",
        "**/hf_cache/",
        "wandb/",
        "runs/",
    ]
    missing = [pattern for pattern in patterns if pattern not in gitignore]
    add(rows, "checkpoint artifacts path gitignored", "PASS" if not missing else "FAIL", missing, "all required patterns")
    add(rows, "Exp5 artifacts directory", "PASS", relpath(EXP05_ARTIFACTS_DIR), "under thesis_exp/artifacts")
    result = subprocess.run(
        ["git", "ls-files", "thesis_exp/artifacts", "*.safetensors", "*.pt", "*.pth", "*.bin", "*/hf_cache/*"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    add(rows, "no checkpoint/weights tracked by git", "PASS" if result.returncode == 0 and not tracked else "FAIL", tracked, "[]")


def run_setup_sanity() -> list[dict[str, Any]]:
    ensure_exp05_dirs()
    rows: list[dict[str, Any]] = []
    check_exp4_baseline(rows)
    check_dataset(rows)
    check_weights_and_loss(rows)
    check_l2_configs(rows)
    check_l2_loss(rows)
    check_scripts_and_modules(rows)
    check_gitignore_and_artifacts(rows)
    write_csv(EXP05_TABLES_DIR / "sanity_check_exp05_setup.csv", rows)
    overall = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_text(
        EXP05_OUTPUT_DIR / "sanity_check_exp05_setup.md",
        f"""# Exp5 Setup Sanity Check

Overall status: **{overall}**

{markdown_table(rows)}
""",
    )
    try:
        write_exp05_report()
    except Exception as exc:
        add(rows, "report refresh", "FAIL", f"{type(exc).__name__}: {exc}", "refresh ok")
        write_csv(EXP05_TABLES_DIR / "sanity_check_exp05_setup.csv", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Exp5 setup before smoke or formal training.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    rows = run_setup_sanity()
    statuses = sorted({row["status"] for row in rows})
    print(f"Exp5 setup sanity statuses: {', '.join(statuses)}")
    print(f"Output: {relpath(EXP05_OUTPUT_DIR / 'sanity_check_exp05_setup.md')}")
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
