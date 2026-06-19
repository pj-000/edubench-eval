"""Preflight-only Exp10 ablation matrix report."""

from __future__ import annotations

import os
from typing import Any

import yaml

from thesis_exp.src.edujudge.exp10_qdpr2_module_ablation import (
    ABLATION_ORDER,
    EXP10_CONFIG_DIR,
    EXP10_REPORTS_DIR,
    EXP10_TABLES_DIR,
    ablation_metadata,
    ensure_exp10_dirs,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_csv, write_text


FIELDS = [
    "ablation_name",
    "display_name",
    "lambda_point",
    "lambda_pair",
    "lambda_anchor",
    "lambda_mono",
    "force_pair_training",
    "use_pair_training",
    "expected_dataloader_mode",
    "config_dataloader_mode",
    "strict_module_ablation",
    "removed_module",
    "active_losses",
    "status",
    "details",
]


def selected_ablations() -> list[str]:
    raw = os.environ.get("EXP10_ABLATIONS", "").strip()
    if not raw:
        return list(ABLATION_ORDER)
    return raw.split()


def _read_config(name: str) -> dict[str, Any]:
    path = EXP10_CONFIG_DIR / f"exp10_{name}.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in selected_ablations():
        if name not in ABLATION_ORDER:
            rows.append(
                {
                    "ablation_name": name,
                    "status": "FAIL",
                    "details": "unknown ablation",
                }
            )
            continue
        meta = ablation_metadata(name)
        config = _read_config(name)
        details = []
        for key in ["lambda_point", "lambda_pair", "lambda_anchor", "lambda_mono", "force_pair_training"]:
            if str(config.get(key)) != str(meta[key]):
                details.append(f"{key}: config={config.get(key)} expected={meta[key]}")
        if config.get("dataloader_mode") != meta["dataloader_mode"]:
            details.append(f"dataloader_mode: config={config.get('dataloader_mode')} expected={meta['dataloader_mode']}")
        if bool(config.get("strict_module_ablation", False)) != bool(meta["strict_module_ablation"]):
            details.append(
                "strict_module_ablation: "
                f"config={config.get('strict_module_ablation')} expected={meta['strict_module_ablation']}"
            )
        row = {
            "ablation_name": name,
            "display_name": meta["display_name"],
            "lambda_point": meta["lambda_point"],
            "lambda_pair": meta["lambda_pair"],
            "lambda_anchor": meta["lambda_anchor"],
            "lambda_mono": meta["lambda_mono"],
            "force_pair_training": meta["force_pair_training"],
            "use_pair_training": meta["use_pair_training"],
            "expected_dataloader_mode": meta["dataloader_mode"],
            "config_dataloader_mode": config.get("dataloader_mode", ""),
            "strict_module_ablation": meta["strict_module_ablation"],
            "removed_module": meta["removed_module"],
            "active_losses": meta["active_losses"],
            "status": "PASS" if not details else "FAIL",
            "details": "; ".join(details),
        }
        rows.append(row)
    return rows


def write_report(rows: list[dict[str, Any]]) -> None:
    write_csv(EXP10_TABLES_DIR / "exp10_preflight_ablation_matrix.csv", rows, fieldnames=FIELDS)
    lines = [
        "# Exp10 Preflight Ablation Matrix",
        "",
        "Training executed: no",
        "",
        "| ablation | lambdas | force_pair_training | use_pair_training | expected_dataloader_mode | strict_module_ablation | status |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lambdas = (
            f"point={row.get('lambda_point')}, pair={row.get('lambda_pair')}, "
            f"anchor={row.get('lambda_anchor')}, mono={row.get('lambda_mono')}"
        )
        lines.append(
            "| {ablation} | {lambdas} | {force} | {use_pair} | {mode} | {strict} | {status} |".format(
                ablation=row.get("ablation_name", ""),
                lambdas=lambdas,
                force=row.get("force_pair_training", ""),
                use_pair=row.get("use_pair_training", ""),
                mode=row.get("expected_dataloader_mode", ""),
                strict=row.get("strict_module_ablation", ""),
                status=row.get("status", ""),
            )
        )
    write_text(EXP10_REPORTS_DIR / "exp10_preflight_ablation_matrix.md", "\n".join(lines))


def main() -> None:
    ensure_exp10_dirs()
    rows = build_rows()
    write_report(rows)
    for row in rows:
        print(
            "ablation_name={ablation_name} lambda_point={lambda_point} lambda_pair={lambda_pair} "
            "lambda_anchor={lambda_anchor} lambda_mono={lambda_mono} force_pair_training={force_pair_training} "
            "use_pair_training={use_pair_training} expected_dataloader_mode={expected_dataloader_mode} "
            "strict_module_ablation={strict_module_ablation} status={status}".format(**row)
        )
    if any(row.get("status") == "FAIL" for row in rows):
        raise SystemExit(f"Exp10 preflight FAIL. See {relpath(EXP10_REPORTS_DIR / 'exp10_preflight_ablation_matrix.md')}")
    print(f"Exp10 preflight PASS: {relpath(EXP10_REPORTS_DIR / 'exp10_preflight_ablation_matrix.md')}")


if __name__ == "__main__":
    main()
