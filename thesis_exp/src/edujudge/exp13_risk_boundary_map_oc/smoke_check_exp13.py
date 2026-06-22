"""Lightweight Exp13 smoke check for risk-boundary losses."""

from __future__ import annotations

import numpy as np

from thesis_exp.src.edujudge.exp13_risk_boundary_map_oc import EXP13_OUTPUT_DIR, ensure_exp13_dirs
from thesis_exp.src.edujudge.exp13_risk_boundary_map_oc.risk_boundary_losses import (
    l2h_squared_hinge_risk_loss_from_logits,
    t3_calibration_loss_from_logits,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


def main() -> None:
    ensure_exp13_dirs()
    smoke_dir = EXP13_OUTPUT_DIR / "smoke_test"
    tables_dir = smoke_dir / "tables"
    reports_dir = smoke_dir / "reports"

    logits = np.array(
        [
            [3.0, 2.0, 0.8, -1.0],
            [2.0, 1.0, 0.1, -2.0],
            [3.0, 2.5, 1.5, 0.5],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([1, 2, 3, 5], dtype=np.int64)
    loss, debug = l2h_squared_hinge_risk_loss_from_logits(logits, labels)
    no_low_loss, no_low_debug = l2h_squared_hinge_risk_loss_from_logits(logits[2:], labels[2:])
    t3_loss, t3_debug = t3_calibration_loss_from_logits(logits, labels)
    torch_backward_ok: bool | str = "SKIP_NO_TORCH"
    try:
        import torch

        torch_logits = torch.tensor(logits, dtype=torch.float32, requires_grad=True)
        torch_labels = torch.tensor(labels, dtype=torch.long)
        torch_loss, _ = l2h_squared_hinge_risk_loss_from_logits(torch_logits, torch_labels)
        torch_loss.backward()
        torch_backward_ok = bool(torch_logits.grad is not None and torch.isfinite(torch_logits.grad).all().item())
    except ModuleNotFoundError:
        torch_backward_ok = "SKIP_NO_TORCH"
    except Exception:
        torch_backward_ok = False

    rows = [
        {"case": "l2h_projected_squared_hinge", "loss": loss, **debug},
        {"case": "no_low_batch", "loss": no_low_loss, **no_low_debug},
        {"case": "t3_brier", "loss": t3_loss, **t3_debug},
        {"case": "torch_backward", "loss": float(loss), "torch_backward_ok": torch_backward_ok},
    ]
    status = "PASS"
    if not np.isfinite(float(loss)) or float(loss) <= 0.0:
        status = "FAIL"
    if abs(float(no_low_loss)) > 1e-12:
        status = "FAIL"
    if not np.isfinite(float(t3_loss)) or float(t3_loss) <= 0.0:
        status = "FAIL"
    if torch_backward_ok is False:
        status = "FAIL"

    write_csv(tables_dir / "exp13_risk_loss_smoke_cases.csv", rows)
    write_text(
        reports_dir / "exp13_smoke_check.md",
        "\n".join(
            [
                "# Exp13 Smoke Check",
                "",
                f"Status: `{status}`",
                "",
                "This smoke check validates the low-to-high squared hinge loss and optional T3 calibration loss.",
                "It does not run model training and does not overwrite formal results.",
            ]
        ),
    )
    print(f"Exp13 smoke {status}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
