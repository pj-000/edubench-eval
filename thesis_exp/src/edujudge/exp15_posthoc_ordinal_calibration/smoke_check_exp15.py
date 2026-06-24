"""Smoke checks for Exp15 calibration math."""

from __future__ import annotations

import numpy as np

from thesis_exp.src.edujudge.exp15_posthoc_ordinal_calibration import EXP15_OUTPUT_DIR, ensure_exp15_dirs
from thesis_exp.src.edujudge.exp15_posthoc_ordinal_calibration.calibration import (
    CalibratorSpec,
    cumulative_decode,
    fit_isotonic_thresholds,
    raw_probs_from_spec,
    threshold_brier,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


def main() -> None:
    ensure_exp15_dirs()
    logits = np.array(
        [
            [4.0, 2.0, -2.0, -4.0],
            [4.0, 3.0, 2.0, -2.0],
            [5.0, 4.0, 3.0, 2.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([2, 3, 5], dtype=np.int64)
    spec = CalibratorSpec("smoke_pava", "smoke", "PAVA smoke")
    probs = raw_probs_from_spec(logits, spec)
    preds = cumulative_decode(probs, spec.decode_thresholds)
    models = fit_isotonic_thresholds(probs, labels)
    rows = [
        {
            "case": "decode_shape",
            "status": "PASS" if preds.tolist() == [3, 4, 5] else "FAIL",
            "details": preds.tolist(),
        },
        {
            "case": "brier_finite",
            "status": "PASS" if np.isfinite(threshold_brier(probs, labels)) else "FAIL",
            "details": threshold_brier(probs, labels),
        },
        {
            "case": "isotonic_models",
            "status": "PASS" if len(models) == 4 else "FAIL",
            "details": len(models),
        },
    ]
    smoke_dir = EXP15_OUTPUT_DIR / "smoke_test"
    write_csv(smoke_dir / "tables" / "exp15_smoke_cases.csv", rows)
    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    write_text(
        smoke_dir / "reports" / "exp15_smoke_check.md",
        "\n".join(
            [
                "# Exp15 Smoke Check",
                "",
                f"Status: `{status}`",
                "",
                "| case | status | details |",
                "| --- | --- | --- |",
                *[f"| {row['case']} | {row['status']} | {row['details']} |" for row in rows],
            ]
        ),
    )
    if status != "PASS":
        raise SystemExit("Exp15 smoke check failed")
    print("Exp15 smoke PASS")


if __name__ == "__main__":
    main()
