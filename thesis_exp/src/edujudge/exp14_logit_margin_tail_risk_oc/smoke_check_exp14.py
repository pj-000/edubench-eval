"""Lightweight Exp14 smoke check for logit-margin loss."""

from __future__ import annotations

import numpy as np

from thesis_exp.src.edujudge.exp14_logit_margin_tail_risk_oc import EXP14_OUTPUT_DIR, ensure_exp14_dirs
from thesis_exp.src.edujudge.exp14_logit_margin_tail_risk_oc.logit_margin_losses import (
    l2h_logit_margin_tail_loss_from_logits,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


def main() -> None:
    ensure_exp14_dirs()
    smoke_dir = EXP14_OUTPUT_DIR / "smoke_test"
    tables_dir = smoke_dir / "tables"
    reports_dir = smoke_dir / "reports"

    logits = np.array(
        [
            [3.0, 2.0, 0.8, -1.0],
            [2.0, 1.0, -0.1, -2.0],
            [2.5, 1.2, 1.6, -0.5],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([1, 2, 2, 5], dtype=np.int64)
    all_loss, all_debug = l2h_logit_margin_tail_loss_from_logits(logits, labels, tail_fraction=1.0)
    tail_loss, tail_debug = l2h_logit_margin_tail_loss_from_logits(logits, labels, tail_fraction=0.5)
    no_low_loss, no_low_debug = l2h_logit_margin_tail_loss_from_logits(logits[-1:], labels[-1:], tail_fraction=0.5)
    torch_backward_ok: bool | str = "SKIP_NO_TORCH"
    try:
        import torch

        torch_logits = torch.tensor(logits, dtype=torch.float32, requires_grad=True)
        torch_labels = torch.tensor(labels, dtype=torch.long)
        torch_loss, _ = l2h_logit_margin_tail_loss_from_logits(torch_logits, torch_labels, tail_fraction=0.5)
        torch_loss.backward()
        torch_backward_ok = bool(torch_logits.grad is not None and torch.isfinite(torch_logits.grad).all().item())
    except ModuleNotFoundError:
        torch_backward_ok = "SKIP_NO_TORCH"
    except Exception:
        torch_backward_ok = False

    rows = [
        {"case": "all_low", "loss": all_loss, **all_debug},
        {"case": "top_tail_0p50", "loss": tail_loss, **tail_debug},
        {"case": "no_low_batch", "loss": no_low_loss, **no_low_debug},
        {"case": "torch_backward", "loss": float(tail_loss), "torch_backward_ok": torch_backward_ok},
    ]
    status = "PASS"
    if not np.isfinite(float(all_loss)) or float(all_loss) <= 0.0:
        status = "FAIL"
    if not np.isfinite(float(tail_loss)) or float(tail_loss) <= 0.0:
        status = "FAIL"
    if float(tail_loss) < float(all_loss):
        status = "FAIL"
    if abs(float(no_low_loss)) > 1e-12:
        status = "FAIL"
    if torch_backward_ok is False:
        status = "FAIL"

    write_csv(tables_dir / "exp14_logit_margin_smoke_cases.csv", rows)
    write_text(
        reports_dir / "exp14_smoke_check.md",
        "\n".join(
            [
                "# Exp14 Smoke Check",
                "",
                f"Status: `{status}`",
                "",
                "This smoke check validates the raw-logit threshold-3 margin loss.",
                "It does not run model training and does not overwrite scout results.",
            ]
        ),
    )
    print(f"Exp14 smoke {status}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
