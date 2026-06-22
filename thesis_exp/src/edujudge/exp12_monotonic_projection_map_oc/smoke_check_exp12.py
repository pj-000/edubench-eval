"""Lightweight Exp12 smoke check that does not overwrite formal results."""

from __future__ import annotations

import numpy as np

from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc import EXP12_OUTPUT_DIR, ensure_exp12_dirs
from thesis_exp.src.edujudge.exp12_monotonic_projection_map_oc.monotone_projection import project_nonincreasing_probs
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


def main() -> None:
    ensure_exp12_dirs()
    smoke_dir = EXP12_OUTPUT_DIR / "smoke_test"
    smoke_tables = smoke_dir / "tables"
    smoke_reports = smoke_dir / "reports"
    raw = np.array(
        [
            [0.9, 0.8, 0.7, 0.2],
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.8, 0.6, 0.4],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    projected = project_nonincreasing_probs(raw)
    rows = []
    for idx, (p, q) in enumerate(zip(raw, projected), start=1):
        rows.append(
            {
                "case_id": idx,
                "raw": p.tolist(),
                "projected": q.tolist(),
                "nonincreasing": bool(np.all(q[:-1] >= q[1:] - 1e-12)),
                "has_nan": bool(np.isnan(q).any()),
                "l2_delta": float(np.sqrt(np.sum((q - p) ** 2))),
            }
        )
    write_csv(smoke_tables / "exp12_projection_smoke_cases.csv", rows)
    status = "PASS" if all(row["nonincreasing"] and not row["has_nan"] for row in rows) else "FAIL"
    write_text(
        smoke_reports / "exp12_smoke_check.md",
        "\n".join(
            [
                "# Exp12 Smoke Check",
                "",
                f"Status: `{status}`",
                "",
                "This smoke check only validates projection behavior and lightweight output generation.",
                "It writes under `thesis_exp/outputs/exp12_monotonic_projection_map_oc/smoke_test/` and does not overwrite formal results.",
            ]
        ),
    )
    print(f"Exp12 smoke {status}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
