"""Validate Exp10 smoke runs."""

from __future__ import annotations

import math
import os
import json
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp10_qdpr2_module_ablation import (
    EXP10_OUTPUT_DIR,
    EXP10_REPORTS_DIR,
    EXP10_TABLES_DIR,
    ablation_metadata,
    ensure_exp10_dirs,
)
from thesis_exp.src.edujudge.utils.io import read_csv, relpath, write_csv, write_text


DEFAULT_SMOKE_ABLATIONS = ["full_qdpr2", "no_pair", "no_pair_same_pair_batches", "no_anchor"]
FIELDS = ["ablation_name", "check_name", "status", "details"]


def selected_ablations() -> list[str]:
    raw = os.environ.get("EXP10_SMOKE_ABLATIONS", "").strip()
    return raw.split() if raw else list(DEFAULT_SMOKE_ABLATIONS)


def add(rows: list[dict[str, Any]], ablation: str, check_name: str, passed: bool, details: Any = "") -> None:
    rows.append(
        {
            "ablation_name": ablation,
            "check_name": check_name,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _first_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = read_csv(path)
    return rows[0] if rows else {}


def check_loss_total(row: dict[str, str]) -> tuple[bool, str]:
    total = _float(row.get("loss_total") or row.get("L_total"))
    recomputed = (
        _float(row.get("weighted_loss_point"))
        + _float(row.get("weighted_loss_pair"))
        + _float(row.get("weighted_loss_anchor"))
        + _float(row.get("weighted_loss_mono"))
    )
    if not math.isfinite(total) or not math.isfinite(recomputed):
        return False, f"total={total}; recomputed={recomputed}"
    return abs(total - recomputed) < 1e-4, f"total={total}; recomputed={recomputed}"


def check_ablation(rows: list[dict[str, Any]], ablation: str) -> None:
    meta = ablation_metadata(ablation)
    run_dir = EXP10_OUTPUT_DIR / "smoke_test" / ablation
    loss_config_path = run_dir / "tables" / "loss_config.csv"
    loss_debug_path = run_dir / "tables" / "loss_debug_history.csv"
    metrics_path = run_dir / "tables" / "metrics_summary.csv"
    metadata_path = run_dir / "run_metadata.json"
    loss_config = _first_csv(loss_config_path)
    debug = _first_csv(loss_debug_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    add(rows, ablation, "loss_config.csv exists", bool(loss_config), relpath(loss_config_path))
    add(rows, ablation, "loss_debug_history.csv exists", bool(debug), relpath(loss_debug_path))
    add(rows, ablation, "metrics_summary.csv exists", metrics_path.exists(), relpath(metrics_path))
    add(rows, ablation, "run completed", metadata.get("status") == "completed", metadata.get("status", "missing"))
    if not loss_config:
        return

    add(rows, ablation, "force_pair_training loads", _bool(loss_config.get("force_pair_training")) == bool(meta["force_pair_training"]))
    add(rows, ablation, "use_pair_training resolves", _bool(loss_config.get("use_pair_training")) == bool(meta["use_pair_training"]))
    add(rows, ablation, "dataloader_mode matches", loss_config.get("dataloader_mode") == meta["dataloader_mode"])
    add(rows, ablation, "lambda_pair matches", _float(loss_config.get("lambda_pair")) == float(meta["lambda_pair"]))

    if ablation == "no_pair":
        add(rows, ablation, "old no_pair uses pointwise loader", loss_config.get("dataloader_mode") == "pointwise")
    if ablation == "no_pair_same_pair_batches":
        add(rows, ablation, "strict run uses pair loader", loss_config.get("dataloader_mode") == "pair")
        add(rows, ablation, "strict lambda_pair is zero", _float(loss_config.get("lambda_pair")) == 0.0)
        if debug:
            add(rows, ablation, "strict weighted_loss_pair is zero", abs(_float(debug.get("weighted_loss_pair"))) < 1e-12)
            ok, details = check_loss_total(debug)
            add(rows, ablation, "strict total excludes L_pair contribution", ok, details)


def main() -> None:
    ensure_exp10_dirs()
    rows: list[dict[str, Any]] = []
    for ablation in selected_ablations():
        check_ablation(rows, ablation)
    write_csv(EXP10_TABLES_DIR / "exp10_smoke_check.csv", rows, fieldnames=FIELDS)
    failures = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Exp10 Smoke Check",
        "",
        f"Status: `{'PASS' if not failures else 'FAIL'}`",
        "",
        "| ablation | check | status | details |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['ablation_name']} | {row['check_name']} | {row['status']} | {row.get('details', '')} |"
        for row in rows
    )
    write_text(EXP10_REPORTS_DIR / "exp10_smoke_check.md", "\n".join(lines))
    if failures:
        raise SystemExit(f"Exp10 smoke check FAIL. See {relpath(EXP10_REPORTS_DIR / 'exp10_smoke_check.md')}")
    print(f"Exp10 smoke check PASS: {relpath(EXP10_REPORTS_DIR / 'exp10_smoke_check.md')}")


if __name__ == "__main__":
    main()
