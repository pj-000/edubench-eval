"""Validate Exp19-R5 DPO LLaMA-Factory configs before scout training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import write_csv, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5_dpo_scout")

DEFAULT_CONFIGS = [
    Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5c_score_risk_dpo_seed42/"
        "configs/llamafactory_qwen3_4b_r5c_dpo_from_r2c.yaml"
    ),
    Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5c_score_risk_dpo_seed42/"
        "configs/llamafactory_qwen3_4b_r5c_dpo_from_r1b.yaml"
    ),
    Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5d_evidence_consistency_dpo_seed42/"
        "configs/llamafactory_qwen3_4b_r5d_dpo_from_r2c.yaml"
    ),
    Path(
        "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5e_hard_synthetic_dpo_control_seed42/"
        "configs/llamafactory_qwen3_4b_r5e_dpo_control_from_r2c.yaml"
    ),
]

FIELDS = [
    "config_path",
    "config_exists",
    "config_is_dict",
    "dataset_dir",
    "dataset_dir_exists",
    "dataset",
    "dataset_info_snippet",
    "dataset_info_snippet_exists",
    "dataset_name_found",
    "data_file_from_snippet",
    "data_file_exists",
    "stage",
    "stage_is_dpo",
    "pref_loss",
    "pref_loss_exists",
    "pref_beta",
    "pref_beta_exists",
    "adapter_name_or_path",
    "adapter_exists",
    "ref_model_adapters",
    "ref_adapter_exists",
    "error_count",
    "warning_count",
    "status",
    "errors",
    "warnings",
]


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def as_path(value: Any) -> Path:
    text = str(value or "").strip()
    return Path(text).expanduser()


def load_yaml(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exercised by CLI failures
        return None, [f"yaml_parse_error: {exc}"]
    if not isinstance(data, dict):
        return None, ["config did not parse to a dict"]
    return data, []


def snippet_candidates(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.glob("dataset_info*.json")
        if path.name != "dataset_info.json" and path.is_file()
    )


def load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def find_dataset_info(dataset_dir: Path, dataset_name: str) -> tuple[Path | None, dict[str, Any] | None]:
    for candidate in snippet_candidates(dataset_dir):
        data = load_json_object(candidate)
        if data and dataset_name in data:
            entry = data.get(dataset_name)
            return candidate, entry if isinstance(entry, dict) else None
    return None, None


def adapter_exists(value: Any) -> bool:
    if value is None or str(value).strip() == "":
        return False
    path = as_path(value)
    return path.exists()


def validate_config(config_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    row: dict[str, Any] = {key: "" for key in FIELDS}
    row["config_path"] = rel(config_path)
    row["config_exists"] = config_path.exists()
    if not config_path.exists():
        errors.append("config missing")
        row["status"] = "FAIL"
        row["errors"] = "; ".join(errors)
        row["error_count"] = len(errors)
        row["warning_count"] = len(warnings)
        return row

    config, load_errors = load_yaml(config_path)
    errors.extend(load_errors)
    row["config_is_dict"] = bool(config)
    if config is None:
        row["status"] = "FAIL"
        row["errors"] = "; ".join(errors)
        row["error_count"] = len(errors)
        row["warning_count"] = len(warnings)
        return row

    dataset_dir = as_path(config.get("dataset_dir"))
    dataset_name = str(config.get("dataset") or "").strip()
    row["dataset_dir"] = str(dataset_dir)
    row["dataset_dir_exists"] = dataset_dir.exists()
    row["dataset"] = dataset_name
    if not dataset_dir.exists():
        errors.append("dataset_dir missing")
    if not dataset_name:
        errors.append("dataset missing")

    snippet_path: Path | None = None
    dataset_entry: dict[str, Any] | None = None
    if dataset_dir.exists() and dataset_name:
        snippet_path, dataset_entry = find_dataset_info(dataset_dir, dataset_name)
    row["dataset_info_snippet"] = rel(snippet_path) if snippet_path else ""
    row["dataset_info_snippet_exists"] = bool(snippet_path)
    row["dataset_name_found"] = bool(dataset_entry)
    if not snippet_path:
        errors.append("dataset_info snippet missing")
    if snippet_path and not dataset_entry:
        errors.append("dataset name not found in snippet")

    data_file = ""
    if dataset_entry:
        data_file = str(dataset_entry.get("file_name") or "")
        data_path = dataset_dir / data_file
        row["data_file_from_snippet"] = data_file
        row["data_file_exists"] = data_path.exists()
        if not data_path.exists():
            warnings.append("full DPO data file is missing locally; it may exist only on server")

    stage = str(config.get("stage") or "").strip()
    row["stage"] = stage
    row["stage_is_dpo"] = stage == "dpo"
    if stage != "dpo":
        errors.append("stage is not dpo")

    pref_loss = config.get("pref_loss")
    pref_beta = config.get("pref_beta")
    row["pref_loss"] = pref_loss
    row["pref_loss_exists"] = pref_loss not in (None, "")
    row["pref_beta"] = pref_beta
    row["pref_beta_exists"] = pref_beta not in (None, "")
    if pref_loss in (None, ""):
        errors.append("pref_loss missing")
    if pref_beta in (None, ""):
        errors.append("pref_beta missing")

    adapter = config.get("adapter_name_or_path")
    ref_adapter = config.get("ref_model_adapters")
    row["adapter_name_or_path"] = str(adapter or "")
    row["adapter_exists"] = adapter_exists(adapter)
    row["ref_model_adapters"] = str(ref_adapter or "")
    row["ref_adapter_exists"] = adapter_exists(ref_adapter)
    if adapter and not row["adapter_exists"]:
        warnings.append("adapter_name_or_path does not exist in this environment")
    if ref_adapter and not row["ref_adapter_exists"]:
        warnings.append("ref_model_adapters does not exist in this environment")

    row["error_count"] = len(errors)
    row["warning_count"] = len(warnings)
    row["status"] = "PASS" if not errors else "FAIL"
    row["errors"] = "; ".join(errors)
    row["warnings"] = "; ".join(warnings)
    return row


def report_lines(rows: list[dict[str, Any]]) -> list[str]:
    failed = [row for row in rows if row.get("status") != "PASS"]
    warnings = [row for row in rows if int(row.get("warning_count") or 0) > 0]
    lines = [
        "# Exp19-R5 DPO Config Validation",
        "",
        "This pre-flight check validates R5C/R5D/R5E LLaMA-Factory DPO configs before scout training.",
        "Adapter path checks are warnings because local and server environments can differ.",
        "The scout train script validates these base configs, then overrides `MAX_STEPS=100` and `PREF_BETA=0.05` by default at runtime.",
        "",
        f"- configs checked: {len(rows)}",
        f"- failed configs: {len(failed)}",
        f"- configs with warnings: {len(warnings)}",
        "",
        "| config | status | dataset | stage | pref_beta | warnings | errors |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{Path(str(row['config_path'])).name}` | {row['status']} | `{row['dataset']}` | "
            f"{row['stage']} | {row['pref_beta']} | {row['warnings']} | {row['errors']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This script does not read train/dev/test examples.",
            "- This script does not train or run inference.",
            "- Full DPO JSON files may remain gitignored and server-side.",
        ]
    )
    return lines


def validate(args: argparse.Namespace) -> dict[str, Any]:
    config_paths = args.config or DEFAULT_CONFIGS
    rows = [validate_config(path) for path in config_paths]
    write_csv(args.out_dir / "tables" / "exp19_r5_dpo_config_validation.csv", rows, FIELDS)
    write_text(
        args.out_dir / "reports" / "exp19_r5_dpo_config_validation_report.md",
        "\n".join(report_lines(rows)),
    )
    failed = [row for row in rows if row.get("status") != "PASS"]
    summary = {
        "configs_checked": len(rows),
        "failed": len(failed),
        "warnings": sum(int(row.get("warning_count") or 0) for row in rows),
        "table": str(args.out_dir / "tables" / "exp19_r5_dpo_config_validation.csv"),
        "report": str(args.out_dir / "reports" / "exp19_r5_dpo_config_validation_report.md"),
    }
    if failed and not args.allow_fail:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Exp19-R5 DPO configs.")
    parser.add_argument("--config", type=Path, action="append", help="Config path. Defaults to R5C/R5D/R5E scout set.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allow-fail", action="store_true", help="Write report but exit 0 even if validation fails.")
    args = parser.parse_args()
    summary = validate(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
