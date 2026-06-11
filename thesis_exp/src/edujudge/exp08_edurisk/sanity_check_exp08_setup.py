"""Setup sanity checks for Exp8 EduRisk implementation."""

from __future__ import annotations

import py_compile
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.src.edujudge.exp08_edurisk import (
    DEFAULT_ALPHA_RISK,
    DEFAULT_BETA_BCE,
    DEFAULT_CLASS_BALANCE_BETA,
    DEFAULT_LAMBDA_HL,
    DEFAULT_LAMBDA_LH,
    DEFAULT_TAU,
    EXP08_CONFIG_DIR,
    EXP08_OUTPUT_DIR,
    EXP08_RUN_ID,
    EXP08_SRC_DIR,
    EXP08_TABLES_DIR,
    ensure_exp08_dirs,
    exp08_run_dir,
)
from thesis_exp.src.edujudge.exp08_edurisk.coral_distribution import (
    expected_score_from_q,
    pred_label_argmax,
    pred_label_cumulative,
    q_from_logits,
    q_raw_sanity,
)
from thesis_exp.src.edujudge.exp08_edurisk.collect_exp08_results import collect
from thesis_exp.src.edujudge.exp08_edurisk.data import (
    class_balanced_weight_rows,
    class_weight_vector,
    dataset_sanity_rows,
    exp0_to_exp7_tracked_output_changes,
    read_split,
    tracked_weight_files,
    write_class_balanced_weights,
)
from thesis_exp.src.edujudge.exp08_edurisk.losses import (
    edurisk_loss,
    make_cumulative_targets,
    make_edurisk_cost_matrix,
    make_soft_ordinal_targets,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


SCRIPT_PATHS = [
    Path("thesis_exp/scripts/run_exp08_qder1_smoke.sh"),
    Path("thesis_exp/scripts/run_exp08_qder1_train.sh"),
    Path("thesis_exp/scripts/sync_exp08_qder1_to_server.sh"),
]


def add(rows: list[dict[str, Any]], check_name: str, passed: bool, details: Any = "") -> None:
    rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})


def check_py_compile(rows: list[dict[str, Any]]) -> None:
    for path in sorted(EXP08_SRC_DIR.glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
            add(rows, f"py_compile {relpath(path)}", True)
        except py_compile.PyCompileError as exc:
            add(rows, f"py_compile {relpath(path)}", False, str(exc))


def check_scripts(rows: list[dict[str, Any]]) -> None:
    for path in SCRIPT_PATHS:
        exists = path.exists()
        add(rows, f"script exists {relpath(path)}", exists)
        if not exists:
            continue
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        add(rows, f"bash -n {relpath(path)}", result.returncode == 0, (result.stderr or result.stdout).strip())


def check_config(rows: list[dict[str, Any]]) -> None:
    formal = EXP08_CONFIG_DIR / "exp08_qder1_edurisk_human_only.yaml"
    smoke = EXP08_CONFIG_DIR / "exp08_qder1_edurisk_smoke.yaml"
    for path in [formal, smoke]:
        add(rows, f"config exists {relpath(path)}", path.exists())
        if path.exists():
            text = path.read_text(encoding="utf-8")
            add(rows, f"config declares run id {path.name}", EXP08_RUN_ID in text)
            add(rows, f"config disables synthetic {path.name}", "use_synthetic: false" in text)
    if formal.exists():
        text = formal.read_text(encoding="utf-8")
        add(rows, "formal beta is 0.99", "class_balance_beta: 0.99" in text)
        add(rows, "formal normalized cost enabled", "normalized_cost: true" in text)
        add(rows, "formal decode primary cumulative", "decode_primary: cumulative" in text)
        add(rows, "formal decode secondary argmax_q", "decode_secondary: argmax_q" in text)


def check_toy_math(rows: list[dict[str, Any]]) -> dict[str, float]:
    toy_logits = np.array([[3.0, 2.0, 0.5, -1.0], [2.2, 1.1, -0.7, -2.0], [4.0, 3.0, 2.0, 1.0]])
    labels = np.array([1, 2, 5])
    probs, q_raw, q_safe = q_from_logits(toy_logits)
    sanity = q_raw_sanity(q_raw)
    add(rows, "toy CORAL logits shape", toy_logits.shape == (3, 4), toy_logits.shape)
    add(rows, "toy q_raw shape", q_raw.shape == (3, 5), q_raw.shape)
    add(rows, "toy q_safe shape", q_safe.shape == (3, 5), q_safe.shape)
    add(rows, "toy q_raw sane before guard", bool(sanity["ok"]), sanity)
    add(rows, "toy q_safe sums to one", bool(np.allclose(q_safe.sum(axis=1), 1.0)), q_safe.sum(axis=1).tolist())
    soft = make_soft_ordinal_targets(labels, tau=DEFAULT_TAU)
    cumulative = make_cumulative_targets(labels)
    cost = make_edurisk_cost_matrix(labels, DEFAULT_LAMBDA_LH, DEFAULT_LAMBDA_HL, normalized_cost=True)
    add(rows, "soft target shape", soft.shape == (3, 5), soft.shape)
    add(rows, "soft target rows sum to one", bool(np.allclose(soft.sum(axis=1), 1.0)), soft.sum(axis=1).tolist())
    add(rows, "cumulative target shape", cumulative.shape == (3, 4), cumulative.shape)
    add(rows, "normalized risk cost shape", cost.shape == (3, 5), cost.shape)
    add(rows, "normalized risk cost max bounded", float(np.max(cost)) <= 3.0, float(np.max(cost)))
    preds_cum = pred_label_cumulative(probs)
    preds_argmax = pred_label_argmax(q_raw)
    expected = expected_score_from_q(q_raw)
    add(rows, "cumulative predictions finite", bool(np.isfinite(preds_cum).all()), preds_cum.tolist())
    add(rows, "argmax predictions finite", bool(np.isfinite(preds_argmax).all()), preds_argmax.tolist())
    add(rows, "expected scores finite", bool(np.isfinite(expected).all()), expected.tolist())
    weight_vector = np.array([1.46, 1.56, 0.68, 0.65, 0.65], dtype=np.float64)
    config = {
        "tau": DEFAULT_TAU,
        "alpha_risk": DEFAULT_ALPHA_RISK,
        "beta_bce": DEFAULT_BETA_BCE,
        "lambda_lh": DEFAULT_LAMBDA_LH,
        "lambda_hl": DEFAULT_LAMBDA_HL,
        "normalized_cost": True,
    }
    loss, debug = edurisk_loss(toy_logits, labels, config, weight_vector)
    add(rows, "toy EduRisk loss finite", bool(np.isfinite(loss)), loss)
    for key in [
        "L_total",
        "L_softCE",
        "L_risk",
        "L_cumBCE",
        "mean_weight",
        "min_weight",
        "max_weight",
        "weighted_L_softCE",
        "weighted_L_risk",
        "weighted_L_cumBCE",
        "mean_q_raw_min",
        "mean_q_raw_sum",
        "max_q_raw_sum_error",
    ]:
        add(rows, f"toy debug has {key}", key in debug and np.isfinite(float(debug[key])), debug.get(key))
    write_csv(EXP08_TABLES_DIR / "toy_loss_component_scale.csv", [debug])
    write_csv(exp08_run_dir(False) / "tables" / "toy_loss_component_scale.csv", [debug])
    return debug


def check_class_weights(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_rows = read_split(Path("thesis_exp/outputs/exp06_synthetic_low_score/training_datasets/QD-S0_human_only"), "train")
    weight_rows = class_balanced_weight_rows(train_rows, beta=DEFAULT_CLASS_BALANCE_BETA)
    vector = class_weight_vector(train_rows, beta=DEFAULT_CLASS_BALANCE_BETA)
    write_class_balanced_weights(train_rows, DEFAULT_CLASS_BALANCE_BETA, [EXP08_TABLES_DIR, exp08_run_dir(False) / "tables"])
    add(rows, "class weights have five rows", len(weight_rows) == 5, len(weight_rows))
    add(rows, "class weights beta is 0.99", DEFAULT_CLASS_BALANCE_BETA == 0.99, DEFAULT_CLASS_BALANCE_BETA)
    add(rows, "class weights no clipping", min(vector) > 0.5 and max(vector) < 2.0, vector)
    add(rows, "class weights mean approximately one", abs(float(np.mean(vector)) - 1.0) < 1e-9, vector)
    return weight_rows


def check_repository_safety(rows: list[dict[str, Any]]) -> None:
    weights = tracked_weight_files()
    add(rows, "no checkpoint/weights tracked", not weights, ", ".join(weights))
    changed = exp0_to_exp7_tracked_output_changes()
    add(rows, "no tracked Exp0-Exp7 output modifications", not changed, ", ".join(changed))


def write_setup_report(rows: list[dict[str, Any]]) -> None:
    write_csv(EXP08_TABLES_DIR / "sanity_check_exp08_setup.csv", rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp8 Setup Sanity Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "Training executed: `no`.",
        "API called: `no`.",
        "Synthetic generated: `no`.",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |" for row in rows)
    write_text(EXP08_OUTPUT_DIR / "sanity_check_exp08_setup.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp8 setup sanity failed. See {relpath(EXP08_OUTPUT_DIR / 'sanity_check_exp08_setup.md')}")


def main() -> None:
    ensure_exp08_dirs()
    rows: list[dict[str, Any]] = []
    rows.extend(dataset_sanity_rows())
    check_config(rows)
    check_py_compile(rows)
    check_scripts(rows)
    check_class_weights(rows)
    check_toy_math(rows)
    collect()
    check_repository_safety(rows)
    write_setup_report(rows)
    print("Exp8 setup sanity PASS")


if __name__ == "__main__":
    main()
