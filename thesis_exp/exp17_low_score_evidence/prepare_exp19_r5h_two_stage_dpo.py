"""Prepare Exp19-R5H two-stage DPO calibration data/configs.

R5H tests a second-stage calibration idea after R5F2/R5G showed a trade-off:
real-only low-risk DPO reduces low-score overestimation, but can over-correct
and hurt high-score protection. R5H starts from a low-risk adapter and applies
only lightweight high-protection DPO pairs.

This script only prepares train-side DPO data and LLaMA-Factory configs. It
does not train, does not read test, and keeps full DPO JSON under gitignored
``data/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_r5a_onpolicy_dpo import score_of  # noqa: E402
from thesis_exp.exp17_low_score_evidence.prepare_exp19_r5c_score_risk_dpo import (  # noqa: E402
    target_from_message,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    MODEL_PATH,
    dpo_dataset_entry,
    sha1,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5h_two_stage_dpo_seed42")
DEFAULT_R5F2_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5f2_rejection_mining_seed42")
DEFAULT_R5G_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5g_risk_calibrated_dpo_seed42")

DATASET_NAME = "edubench_r5h_high_protection_only_dpo_train"
DATASET_FILE = "data/edubench_r5h_high_protection_only_dpo_train.json"

R5F2_REAL_ADAPTER = "saves/edubench/qwen3-4b/r5f2_real_only_small_from_r2c_maxsteps100_lora"
R5G_A3_ADAPTER = "saves/edubench/qwen3-4b/r5g_a3_real_only_s50_b0p05_lr5em6_lora"

RUN_CONFIGS = [
    {
        "run_name": "r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6",
        "init_adapter": R5F2_REAL_ADAPTER,
        "init_family": "r5f2_real_only_from_r2c",
        "max_steps": 10,
        "pref_beta": 0.02,
        "learning_rate": "1.0e-6",
    },
    {
        "run_name": "r5h_h2_from_r5f2_real_highprotect_s20_b0p02_lr1e6",
        "init_adapter": R5F2_REAL_ADAPTER,
        "init_family": "r5f2_real_only_from_r2c",
        "max_steps": 20,
        "pref_beta": 0.02,
        "learning_rate": "1.0e-6",
    },
    {
        "run_name": "r5h_h3_from_r5f2_real_highprotect_s30_b0p02_lr1e6",
        "init_adapter": R5F2_REAL_ADAPTER,
        "init_family": "r5f2_real_only_from_r2c",
        "max_steps": 30,
        "pref_beta": 0.02,
        "learning_rate": "1.0e-6",
    },
    {
        "run_name": "r5h_h4_from_r5f2_real_highprotect_s20_b0p03_lr2e6",
        "init_adapter": R5F2_REAL_ADAPTER,
        "init_family": "r5f2_real_only_from_r2c",
        "max_steps": 20,
        "pref_beta": 0.03,
        "learning_rate": "2.0e-6",
    },
    {
        "run_name": "r5h_h5_from_r5g_a3_highprotect_s20_b0p02_lr1e6",
        "init_adapter": R5G_A3_ADAPTER,
        "init_family": "r5g_a3",
        "max_steps": 20,
        "pref_beta": 0.02,
        "learning_rate": "1.0e-6",
    },
    {
        "run_name": "r5h_h6_from_r5g_a3_highprotect_s30_b0p02_lr1e6",
        "init_adapter": R5G_A3_ADAPTER,
        "init_family": "r5g_a3",
        "max_steps": 30,
        "pref_beta": 0.02,
        "learning_rate": "1.0e-6",
    },
]


def default_source_files(r5f2_dir: Path, r5g_dir: Path) -> list[Path]:
    return [
        r5f2_dir / "data" / "edubench_r5f2_score_risk_main_dpo_train.json",
        r5g_dir / "data" / "edubench_r5g_ratio_70_30_dpo_train.json",
        r5g_dir / "data" / "edubench_r5g_ratio_60_40_dpo_train.json",
        r5g_dir / "data" / "edubench_r5g_ratio_50_50_dpo_train.json",
    ]


def read_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return data


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assistant_content(row: dict[str, Any], key: str) -> str:
    value = row.get(key) or {}
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return ""


def has_no_major_failure(target: dict[str, Any]) -> bool:
    failures = target.get("major_failures")
    return isinstance(failures, list) and "no_major_failure" in failures


def has_false_failure(target: dict[str, Any]) -> bool:
    failures = target.get("major_failures")
    if not isinstance(failures, list):
        return False
    return any(str(item) not in {"no_major_failure", "unclear"} for item in failures)


def score_cap_le3(target: dict[str, Any]) -> bool:
    value = target.get("score_cap")
    if value is None:
        return False
    try:
        return int(round(float(value))) <= 3
    except Exception:
        return False


def is_high_protection_pair(row: dict[str, Any]) -> bool:
    risk_type = str(row.get("risk_type", ""))
    if risk_type not in {"high_to_low_score_risk", "high_to_low_protection"}:
        return False
    chosen = target_from_message(row.get("chosen") or {})
    rejected = target_from_message(row.get("rejected") or {})
    chosen_score = score_of(chosen)
    rejected_score = score_of(rejected)
    chosen_ok = chosen_score is not None and chosen_score >= 4 and has_no_major_failure(chosen)
    chosen_ok = chosen_ok and chosen.get("score_cap") is None
    rejected_low = rejected_score is not None and rejected_score <= 3
    rejected_ok = rejected_low and (has_false_failure(rejected) or score_cap_le3(rejected))
    return bool(chosen_ok and rejected_ok)


def pair_hash(row: dict[str, Any]) -> str:
    payload = {
        "messages": row.get("messages", []),
        "chosen": assistant_content(row, "chosen"),
        "rejected": assistant_content(row, "rejected"),
        "source_sample_id": row.get("source_sample_id", ""),
    }
    return sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def normalize_pair(row: dict[str, Any], source_file: Path, source_index: int) -> dict[str, Any]:
    item = dict(row)
    item["risk_type"] = "high_to_low_score_risk"
    item["r5h_stage2_role"] = "high_protection_only"
    item["r5h_source_file"] = str(source_file)
    item["r5h_source_index"] = source_index
    return item


def collect_high_pairs(source_files: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in source_files:
        if not path.exists():
            source_rows.append(
                {
                    "source_file": str(path),
                    "exists": False,
                    "rows": 0,
                    "matching_high_pairs": 0,
                    "new_unique_high_pairs": 0,
                }
            )
            continue
        rows = read_json_array(path)
        matching = 0
        new_unique = 0
        for idx, row in enumerate(rows):
            if not is_high_protection_pair(row):
                continue
            matching += 1
            item = normalize_pair(row, path, idx)
            key = pair_hash(item)
            if key in seen:
                continue
            seen.add(key)
            new_unique += 1
            pairs.append(item)
        source_rows.append(
            {
                "source_file": str(path),
                "exists": True,
                "rows": len(rows),
                "matching_high_pairs": matching,
                "new_unique_high_pairs": new_unique,
            }
        )
    return pairs, source_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def source_category(row: dict[str, Any]) -> str:
    category = str(row.get("rejected_category", ""))
    source = str(row.get("rejected_source", ""))
    if category:
        return category
    if source.startswith("hard_synthetic"):
        return "hard_synthetic"
    if source.startswith("actual") or "sample" in source:
        return "actual_model_generation"
    return "unknown"


def dataset_qc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    false_failure = 0
    source_counts = Counter(source_category(row) for row in rows)
    for row in rows:
        chosen = target_from_message(row.get("chosen") or {})
        rejected = target_from_message(row.get("rejected") or {})
        c_score = score_of(chosen)
        r_score = score_of(rejected)
        if c_score is not None:
            chosen_scores.append(float(c_score))
        if r_score is not None:
            rejected_scores.append(float(r_score))
        if has_false_failure(rejected):
            false_failure += 1
    leakage_fields = ("r5h_source_file", "candidate_pool", "dataset_variant", "rejected_source")
    contains_dev_or_test_marker = any(
        any(marker in str(row.get(field, "")).lower() for marker in ("dev", "test")) for field in leakage_fields
        for row in rows
    )
    return {
        "dataset": DATASET_NAME,
        "pair_count": len(rows),
        "unique_source_sample_ids": len({str(row.get("source_sample_id", "")) for row in rows}),
        "actual_rejected_source_count": source_counts.get("actual_model_generation", 0),
        "synthetic_rejected_source_count": source_counts.get("hard_synthetic", 0),
        "unknown_rejected_source_count": source_counts.get("unknown", 0),
        "chosen_score_mean": mean(chosen_scores),
        "rejected_score_mean": mean(rejected_scores),
        "rejected_false_failure_rate": false_failure / len(rows) if rows else 0.0,
        "contains_low_to_high_pairs": any(str(row.get("risk_type")) == "low_to_high_score_risk" for row in rows),
        "contains_dev_or_test_marker": contains_dev_or_test_marker,
    }


def write_dpo_config(path: Path, run: dict[str, Any], dataset_dir: Path) -> None:
    output_dir = f"saves/edubench/qwen3-4b/{run['run_name']}_lora"
    config = {
        "model_name_or_path": MODEL_PATH,
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "adapter_name_or_path": run["init_adapter"],
        "ref_model": MODEL_PATH,
        "ref_model_adapters": run["init_adapter"],
        "dataset_dir": str(dataset_dir),
        "dataset": DATASET_NAME,
        "template": "qwen3_nothink",
        "cutoff_len": 4096,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "pref_loss": "sigmoid",
        "pref_beta": run["pref_beta"],
        "pref_ftx": 0.1,
        "output_dir": output_dir,
        "logging_steps": 5,
        "save_steps": 999999,
        "save_only_model": True,
        "plot_loss": True,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": run["learning_rate"],
        "bf16": True,
        "gradient_checkpointing": True,
        "max_steps": run["max_steps"],
    }
    lines = [f"{key}: {'true' if value is True else 'false' if value is False else value}" for key, value in config.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_configs(out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for run in RUN_CONFIGS:
        config_path = out_dir / "configs" / f"llamafactory_qwen3_4b_{run['run_name']}.yaml"
        write_dpo_config(config_path, run, out_dir)
        rows.append(
            {
                "run_name": run["run_name"],
                "init_family": run["init_family"],
                "init_adapter": run["init_adapter"],
                "dataset": DATASET_NAME,
                "max_steps": run["max_steps"],
                "pref_beta": run["pref_beta"],
                "learning_rate": run["learning_rate"],
                "output_dir": f"saves/edubench/qwen3-4b/{run['run_name']}_lora",
            }
        )
    write_csv(
        out_dir / "tables" / "r5h_run_matrix.csv",
        rows,
        ["run_name", "init_family", "init_adapter", "dataset", "max_steps", "pref_beta", "learning_rate", "output_dir"],
    )


def write_report(out_dir: Path, qc: dict[str, Any], source_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Exp19-R5H Two-Stage DPO Data QC",
        "",
        "R5H builds a stage-2 high-protection-only DPO dataset. It is intended to start from a",
        "low-risk adapter and lightly restore high-score protection.",
        "",
        "## Dataset",
        "",
        f"- dataset: `{DATASET_NAME}`",
        f"- pair_count: {qc['pair_count']}",
        f"- unique_source_sample_ids: {qc['unique_source_sample_ids']}",
        f"- chosen_score_mean: {float(qc['chosen_score_mean']):.4f}",
        f"- rejected_score_mean: {float(qc['rejected_score_mean']):.4f}",
        f"- rejected_false_failure_rate: {float(qc['rejected_false_failure_rate']):.4f}",
        f"- contains_low_to_high_pairs: `{qc['contains_low_to_high_pairs']}`",
        f"- contains_dev_or_test_marker: `{qc['contains_dev_or_test_marker']}`",
        "",
        "## Source Files",
        "",
        "| source | exists | rows | matching high-protection pairs | new unique pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in source_rows:
        lines.append(
            f"| `{row['source_file']}` | {row['exists']} | {row['rows']} | "
            f"{row['matching_high_pairs']} | {row['new_unique_high_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This preparation step does not read test.",
            "- No human rationale text is added to the prompt.",
            "- Full DPO JSON is written under gitignored `data/`.",
            "- R5H stage-2 data contains only high-protection pairs.",
        ]
    )
    write_text(out_dir / "reports" / "r5h_dataset_qc_report.md", "\n".join(lines))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    source_files = args.source_dpo_json or default_source_files(args.r5f2_dir, args.r5g_dir)
    pairs, source_rows = collect_high_pairs(source_files)
    if len(pairs) < args.min_pairs:
        raise ValueError(
            f"Only {len(pairs)} high-protection pairs found; expected at least {args.min_pairs}. "
            f"Check source DPO JSON files."
        )
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    write_json_array(out_dir / DATASET_FILE, pairs)
    dataset_info = {DATASET_NAME: dpo_dataset_entry(DATASET_FILE)}
    write_json(out_dir / "dataset_info.json", dataset_info)
    write_json(out_dir / "dataset_info_r5h_snippet.json", dataset_info)

    qc = dataset_qc(pairs)
    write_csv(
        out_dir / "tables" / "r5h_high_protection_qc.csv",
        [qc],
        [
            "dataset",
            "pair_count",
            "unique_source_sample_ids",
            "actual_rejected_source_count",
            "synthetic_rejected_source_count",
            "unknown_rejected_source_count",
            "chosen_score_mean",
            "rejected_score_mean",
            "rejected_false_failure_rate",
            "contains_low_to_high_pairs",
            "contains_dev_or_test_marker",
        ],
    )
    write_csv(
        out_dir / "tables" / "r5h_pair_source_counts.csv",
        source_rows,
        ["source_file", "exists", "rows", "matching_high_pairs", "new_unique_high_pairs"],
    )
    write_configs(out_dir)
    write_report(out_dir, qc, source_rows)
    return {
        "out_dir": str(out_dir),
        "dataset": DATASET_NAME,
        "pair_count": len(pairs),
        "configs": len(RUN_CONFIGS),
        "seed": args.seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp19-R5H two-stage DPO data/configs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--r5f2-dir", type=Path, default=DEFAULT_R5F2_DIR)
    parser.add_argument("--r5g-dir", type=Path, default=DEFAULT_R5G_DIR)
    parser.add_argument("--source-dpo-json", type=Path, action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pairs", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 keeps all high-protection pairs.")
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
