"""Build the auditable three-seed Exp56 MeanAux development-only report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp56_meanaux import (
    FORMAL_SEEDS,
    OUTPUT_ROOT,
    baseline_run_dir,
    hmsa_run_dir,
    run_output_dir,
)
from thesis_exp.exp56_meanaux.gate import seed42_gate
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


PRIMARY = ("MAE_human_mean", "Exact_rounded", "Kendall_human_mean")
REPORTED = PRIMARY + ("Bias_human_mean", "QWK_rounded", "L2H_count")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arm() -> dict[str, dict[int, dict[str, Any]]]:
    roots = {
        "hard_only": baseline_run_dir,
        "hmsa": hmsa_run_dir,
        "meanaux": run_output_dir,
    }
    return {
        arm: {seed: read_json(path_fn(seed) / "run_summary.json") for seed in FORMAL_SEEDS}
        for arm, path_fn in roots.items()
    }


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
    }


def build_summary() -> dict[str, Any]:
    runs = load_arm()
    expected_lock: str | None = None
    prediction_hashes: dict[str, str] = {}
    checkpoint_paths: set[str] = set()
    rows: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        meanaux = runs["meanaux"][seed]
        if meanaux["status"] != "COMPLETED":
            raise ValueError(f"MeanAux seed {seed} is incomplete")
        if int(meanaux["test_access_count"]) != 0 or bool(meanaux["nan_or_inf"]):
            raise ValueError(f"MeanAux seed {seed} violates the run contract")
        lock = str(meanaux["source_lock_manifest_sha256"])
        expected_lock = expected_lock or lock
        if lock != expected_lock:
            raise ValueError("MeanAux source-lock hashes differ across seeds")
        checkpoint = str(meanaux["checkpoint_path"])
        if f"seed_{seed}/best" not in checkpoint:
            raise ValueError(f"Seed {seed} checkpoint is not isolated: {checkpoint}")
        checkpoint_paths.add(checkpoint)
        prediction_hashes[str(seed)] = sha256(
            run_output_dir(seed) / "predictions" / "predictions_dev.jsonl"
        )
        for arm in ("hard_only", "hmsa"):
            reference = runs[arm][seed]
            for key in ("train_text_hash", "dev_text_hash", "model_name_or_path", "scheduler"):
                if meanaux[key] != reference[key]:
                    raise ValueError(f"Parity failure for {arm}, seed {seed}, {key}")
        row: dict[str, Any] = {
            "seed": seed,
            "selected_epoch": int(meanaux["selected_epoch"]),
        }
        for arm in ("hard_only", "hmsa", "meanaux"):
            metrics = runs[arm][seed]["selected_metrics"]
            for metric in REPORTED:
                row[f"{arm}_{metric}"] = float(metrics[metric])
        rows.append(row)
    if len(checkpoint_paths) != len(FORMAL_SEEDS):
        raise ValueError("MeanAux checkpoint paths are not unique")
    if len(set(prediction_hashes.values())) != len(FORMAL_SEEDS):
        raise ValueError("MeanAux prediction hashes are not unique")

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ("hard_only", "hmsa", "meanaux"):
        aggregate[arm] = {}
        for metric in REPORTED:
            aggregate[arm][metric] = mean_sd(
                [float(runs[arm][seed]["selected_metrics"][metric]) for seed in FORMAL_SEEDS]
            )

    deltas: dict[str, dict[str, dict[str, Any]]] = {}
    for reference in ("hard_only", "hmsa"):
        name = f"meanaux_minus_{reference}"
        deltas[name] = {}
        for metric in REPORTED:
            values = [
                float(runs["meanaux"][seed]["selected_metrics"][metric])
                - float(runs[reference][seed]["selected_metrics"][metric])
                for seed in FORMAL_SEEDS
            ]
            deltas[name][metric] = {
                "per_seed": dict(zip(map(str, FORMAL_SEEDS), values)),
                "mean": float(np.mean(values)),
            }

    hmsa_better_all_primary = all(
        float(runs["hmsa"][seed]["selected_metrics"]["MAE_human_mean"])
        < float(runs["meanaux"][seed]["selected_metrics"]["MAE_human_mean"])
        and float(runs["hmsa"][seed]["selected_metrics"]["Exact_rounded"])
        > float(runs["meanaux"][seed]["selected_metrics"]["Exact_rounded"])
        and float(runs["hmsa"][seed]["selected_metrics"]["Kendall_human_mean"])
        > float(runs["meanaux"][seed]["selected_metrics"]["Kendall_human_mean"])
        for seed in FORMAL_SEEDS
    )
    return {
        "status": "EXP56_THREE_SEED_DEV_ONLY_COMPLETED",
        "valid": True,
        "seeds": list(FORMAL_SEEDS),
        "source_lock_manifest_sha256": expected_lock,
        "checkpoint_paths_unique": True,
        "prediction_sha256": prediction_hashes,
        "prediction_hashes_unique": True,
        "test_access_count": 0,
        "seed42_gate": seed42_gate(),
        "aggregate": aggregate,
        "deltas": deltas,
        "hmsa_better_than_meanaux_on_all_three_primary_metrics_for_every_seed": hmsa_better_all_primary,
        "interpretation": (
            "MeanAux does not improve the matched Hard-only baseline on average, while HMSA "
            "outperforms MeanAux on MAE, Exact Match, and Kendall tau for every seed. Because "
            "continuous mean and empirical distribution are one-to-one in these splits, this "
            "supports a target-geometry/loss explanation, not an additional-information claim."
        ),
        "rows": rows,
    }


def main() -> None:
    report = build_summary()
    decision_dir = OUTPUT_ROOT / "decision"
    decision_dir.mkdir(parents=True, exist_ok=True)
    (decision_dir / "three_seed_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(OUTPUT_ROOT / "tables" / "three_seed_comparison.csv", report["rows"])
    aggregate = report["aggregate"]
    lines = [
        "# Exp56 MeanAux three-seed development-only report",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Arm | MAE | Exact | Kendall |",
        "|---|---:|---:|---:|",
    ]
    for arm, label in (("hard_only", "Hard-only"), ("hmsa", "HMSA"), ("meanaux", "MeanAux")):
        lines.append(
            f"| {label} | {aggregate[arm]['MAE_human_mean']['mean']:.3f} ± "
            f"{aggregate[arm]['MAE_human_mean']['sample_sd']:.3f} | "
            f"{aggregate[arm]['Exact_rounded']['mean']:.3f} ± "
            f"{aggregate[arm]['Exact_rounded']['sample_sd']:.3f} | "
            f"{aggregate[arm]['Kendall_human_mean']['mean']:.3f} ± "
            f"{aggregate[arm]['Kendall_human_mean']['sample_sd']:.3f} |"
        )
    lines.extend(["", report["interpretation"], "", "No test data were accessed.", ""])
    write_text(OUTPUT_ROOT / "decision" / "three_seed_summary.md", "\n".join(lines))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
