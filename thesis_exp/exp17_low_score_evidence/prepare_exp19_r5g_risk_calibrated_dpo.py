"""Prepare Exp19-R5G risk-calibrated DPO scout data/configs.

R5G follows the R5F2 finding that real low-to-high rejected responses can reduce
low-score overestimation, but risk-only DPO can over-correct and hurt high-score
protection. This script builds a small risk-calibration scout:

- Group A reuses the real-only low-to-high data with lighter DPO strength.
- Group B mixes real low-to-high pairs with high-protection pairs at 70/30,
  60/40, and 50/50 ratios.

The script does not train a model, does not read test, and does not use dev/D1
annotations as training labels. Full DPO JSON is written under gitignored data/.
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

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    MODEL_PATH,
    OPENAI_TAGS,
    dpo_dataset_entry,
    sha1,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_R5F2_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5f2_rejection_mining_seed42")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r5g_risk_calibrated_dpo_seed42")

R5F2_REAL_ONLY_FILE = Path("data/edubench_r5f2_real_only_small_dpo_train.json")
R5F2_MAIN_FILE = Path("data/edubench_r5f2_score_risk_main_dpo_train.json")

REAL_ONLY_DATASET = "edubench_r5g_real_only_dpo_train"
REAL_ONLY_FILE = "data/edubench_r5g_real_only_dpo_train.json"
RATIO_DATASETS = {
    "70_30": ("edubench_r5g_ratio_70_30_dpo_train", "data/edubench_r5g_ratio_70_30_dpo_train.json", 0.70, 0.30),
    "60_40": ("edubench_r5g_ratio_60_40_dpo_train", "data/edubench_r5g_ratio_60_40_dpo_train.json", 0.60, 0.40),
    "50_50": ("edubench_r5g_ratio_50_50_dpo_train", "data/edubench_r5g_ratio_50_50_dpo_train.json", 0.50, 0.50),
}

R2C_ADAPTER = "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora"
RUN_CONFIGS = [
    {
        "run_name": "r5g_a1_real_only_s25_b0p03_lr2em6",
        "dataset": REAL_ONLY_DATASET,
        "dataset_family": "real_only",
        "max_steps": 25,
        "pref_beta": 0.03,
        "learning_rate": "2.0e-6",
        "output_dir": "saves/edubench/qwen3-4b/r5g_a1_real_only_s25_b0p03_lr2em6_lora",
    },
    {
        "run_name": "r5g_a2_real_only_s50_b0p03_lr2em6",
        "dataset": REAL_ONLY_DATASET,
        "dataset_family": "real_only",
        "max_steps": 50,
        "pref_beta": 0.03,
        "learning_rate": "2.0e-6",
        "output_dir": "saves/edubench/qwen3-4b/r5g_a2_real_only_s50_b0p03_lr2em6_lora",
    },
    {
        "run_name": "r5g_a3_real_only_s50_b0p05_lr5em6",
        "dataset": REAL_ONLY_DATASET,
        "dataset_family": "real_only",
        "max_steps": 50,
        "pref_beta": 0.05,
        "learning_rate": "5.0e-6",
        "output_dir": "saves/edubench/qwen3-4b/r5g_a3_real_only_s50_b0p05_lr5em6_lora",
    },
    {
        "run_name": "r5g_b1_ratio70_30_s100_b0p03_lr5em6",
        "dataset": RATIO_DATASETS["70_30"][0],
        "dataset_family": "ratio_70_30",
        "max_steps": 100,
        "pref_beta": 0.03,
        "learning_rate": "5.0e-6",
        "output_dir": "saves/edubench/qwen3-4b/r5g_b1_ratio70_30_s100_b0p03_lr5em6_lora",
    },
    {
        "run_name": "r5g_b2_ratio60_40_s100_b0p03_lr5em6",
        "dataset": RATIO_DATASETS["60_40"][0],
        "dataset_family": "ratio_60_40",
        "max_steps": 100,
        "pref_beta": 0.03,
        "learning_rate": "5.0e-6",
        "output_dir": "saves/edubench/qwen3-4b/r5g_b2_ratio60_40_s100_b0p03_lr5em6_lora",
    },
    {
        "run_name": "r5g_b3_ratio50_50_s100_b0p03_lr5em6",
        "dataset": RATIO_DATASETS["50_50"][0],
        "dataset_family": "ratio_50_50",
        "max_steps": 100,
        "pref_beta": 0.03,
        "learning_rate": "5.0e-6",
        "output_dir": "saves/edubench/qwen3-4b/r5g_b3_ratio50_50_s100_b0p03_lr5em6_lora",
    },
]


def read_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing DPO JSON: {path}")
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


def target_score(row: dict[str, Any], key: str) -> int | None:
    try:
        payload = json.loads(assistant_content(row, key))
        score = payload.get("score")
        if score is None:
            return None
        return int(round(float(score)))
    except Exception:
        return None


def pair_hash(row: dict[str, Any]) -> str:
    payload = {
        "messages": row.get("messages", []),
        "chosen": assistant_content(row, "chosen"),
        "rejected": assistant_content(row, "rejected"),
        "risk_type": row.get("risk_type", ""),
        "source_sample_id": row.get("source_sample_id", ""),
        "rejected_source": row.get("rejected_source", ""),
    }
    return sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def tag_rows(rows: list[dict[str, Any]], dataset_variant: str, pair_role: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["r5g_dataset_variant"] = dataset_variant
        item["r5g_pair_role"] = pair_role
        item["r5g_pair_index"] = idx
        out.append(item)
    return out


def split_low_high(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    low_rows = [row for row in rows if row.get("risk_type") == "low_to_high_score_risk"]
    high_rows = [row for row in rows if row.get("risk_type") == "high_to_low_score_risk"]
    if not low_rows:
        raise ValueError("No low_to_high_score_risk rows found in R5F2 main data")
    if not high_rows:
        raise ValueError("No high_to_low_score_risk rows found in R5F2 main data")
    return low_rows, high_rows


def ratio_sample(
    low_rows: list[dict[str, Any]],
    high_rows: list[dict[str, Any]],
    low_ratio: float,
    high_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    low_count = len(low_rows)
    high_count = round(low_count * high_ratio / low_ratio)
    high_count = min(high_count, len(high_rows))
    sampled_high = rng.sample(high_rows, high_count)
    return list(low_rows), sampled_high


def count_unique(rows: list[dict[str, Any]], key: str) -> int:
    return len({str(row.get(key, "")) for row in rows})


def row_qc(dataset_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    risk_counts = Counter(str(row.get("risk_type", "")) for row in rows)
    rejected_categories = Counter(str(row.get("rejected_category", "")) for row in rows)
    hashes = [pair_hash(row) for row in rows]
    total = len(rows)
    low = risk_counts.get("low_to_high_score_risk", 0)
    high = risk_counts.get("high_to_low_score_risk", 0)
    return {
        "dataset": dataset_name,
        "n": total,
        "low_to_high_count": low,
        "high_to_low_count": high,
        "low_ratio": low / total if total else 0,
        "high_ratio": high / total if total else 0,
        "actual_rejected_count": rejected_categories.get("actual_model_generation", 0),
        "hard_synthetic_count": rejected_categories.get("hard_synthetic", 0),
        "unique_source_sample_ids": count_unique(rows, "source_sample_id"),
        "unique_rejected_sources": count_unique(rows, "rejected_source"),
        "unique_pair_hashes": len(set(hashes)),
        "duplicate_pair_rate": 1 - (len(set(hashes)) / total) if total else 0,
    }


def source_count_rows(dataset_name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (
            str(row.get("risk_type", "")),
            str(row.get("rejected_category", "")),
            str(row.get("candidate_pool", "")),
            str(row.get("rejected_source", "")),
        )
        for row in rows
    )
    return [
        {
            "dataset": dataset_name,
            "risk_type": risk_type,
            "rejected_category": rejected_category,
            "candidate_pool": candidate_pool,
            "rejected_source": rejected_source,
            "n": n,
        }
        for (risk_type, rejected_category, candidate_pool, rejected_source), n in sorted(counts.items())
    ]


def dataset_info_for(file_name: str) -> dict[str, Any]:
    return dpo_dataset_entry(file_name)


def write_dpo_config(path: Path, run: dict[str, Any], dataset_dir: Path) -> None:
    config = {
        "model_name_or_path": MODEL_PATH,
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "adapter_name_or_path": R2C_ADAPTER,
        "ref_model": MODEL_PATH,
        "ref_model_adapters": R2C_ADAPTER,
        "dataset_dir": str(dataset_dir),
        "dataset": run["dataset"],
        "template": "qwen3_nothink",
        "cutoff_len": 4096,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "pref_loss": "sigmoid",
        "pref_beta": run["pref_beta"],
        "pref_ftx": 0.1,
        "output_dir": run["output_dir"],
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
    lines: list[str] = []
    for key, value in config.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_configs(out_dir: Path) -> None:
    for run in RUN_CONFIGS:
        config_path = out_dir / "configs" / f"llamafactory_qwen3_4b_{run['run_name']}.yaml"
        write_dpo_config(config_path, run, out_dir)
    write_csv(
        out_dir / "tables" / "r5g_run_matrix.csv",
        RUN_CONFIGS,
        ["run_name", "dataset", "dataset_family", "max_steps", "pref_beta", "learning_rate", "output_dir"],
    )


def write_report(out_dir: Path, qc_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Exp19-R5G Risk-Calibrated DPO Data QC",
        "",
        "R5G constructs a small DPO scout after R5F2 showed that real low-to-high rejected responses reduce",
        "low-score overestimation but can over-correct and hurt high-score protection.",
        "",
        "## Datasets",
        "",
        "| dataset | n | low-to-high | high-to-low | actual rejected | hard synthetic | duplicate pair rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in qc_rows:
        lines.append(
            f"| `{row['dataset']}` | {row['n']} | {row['low_to_high_count']} | {row['high_to_low_count']} | "
            f"{row['actual_rejected_count']} | {row['hard_synthetic_count']} | {float(row['duplicate_pair_rate']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Scout Matrix",
            "",
            "- Group A: lighter real-only DPO from R2c, using steps/beta/lr sweeps.",
            "- Group B: ratio-calibrated low-risk/high-protection DPO from R2c at 70/30, 60/40, and 50/50.",
            "- No test split is read.",
            "- D1 annotations are not used for training labels.",
            "- Full DPO JSON remains under gitignored `data/`.",
            "",
            "## Source Summary",
            "",
            f"- source rows summarized: {len(source_rows)}",
        ]
    )
    write_text(out_dir / "reports" / "r5g_dataset_qc_report.md", "\n".join(lines))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    out_dir = args.out_dir
    r5f2_dir = args.r5f2_dir
    real_only = read_json_array(r5f2_dir / R5F2_REAL_ONLY_FILE)
    main = read_json_array(r5f2_dir / R5F2_MAIN_FILE)
    low_rows, high_rows = split_low_high(main)

    datasets: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    datasets[REAL_ONLY_DATASET] = (REAL_ONLY_FILE, tag_rows(real_only, "r5g_real_only", "low_to_high"))
    for ratio_name, (dataset_name, file_name, low_ratio, high_ratio) in RATIO_DATASETS.items():
        ratio_rng = random.Random(args.seed + int(ratio_name.split("_")[0]))
        lows, highs = ratio_sample(low_rows, high_rows, low_ratio, high_ratio, ratio_rng)
        rows = tag_rows(lows, f"r5g_ratio_{ratio_name}", "low_to_high") + tag_rows(
            highs, f"r5g_ratio_{ratio_name}", "high_to_low"
        )
        rng.shuffle(rows)
        datasets[dataset_name] = (file_name, rows)

    dataset_info: dict[str, Any] = {}
    qc_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for dataset_name, (file_name, rows) in datasets.items():
        write_json_array(out_dir / file_name, rows)
        dataset_info[dataset_name] = dataset_info_for(file_name)
        qc_rows.append(row_qc(dataset_name, rows))
        source_rows.extend(source_count_rows(dataset_name, rows))

    write_json(out_dir / "dataset_info.json", dataset_info)
    write_json(out_dir / "dataset_info_r5g_snippet.json", dataset_info)
    write_csv(
        out_dir / "tables" / "r5g_dataset_qc.csv",
        qc_rows,
        [
            "dataset",
            "n",
            "low_to_high_count",
            "high_to_low_count",
            "low_ratio",
            "high_ratio",
            "actual_rejected_count",
            "hard_synthetic_count",
            "unique_source_sample_ids",
            "unique_rejected_sources",
            "unique_pair_hashes",
            "duplicate_pair_rate",
        ],
    )
    write_csv(
        out_dir / "tables" / "r5g_pair_source_counts.csv",
        source_rows,
        ["dataset", "risk_type", "rejected_category", "candidate_pool", "rejected_source", "n"],
    )
    write_configs(out_dir)
    write_report(out_dir, qc_rows, source_rows)
    return {
        "out_dir": str(out_dir),
        "datasets": {name: len(rows) for name, (_file, rows) in datasets.items()},
        "configs": len(RUN_CONFIGS),
        "seed": args.seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp19-R5G risk-calibrated DPO data/configs.")
    parser.add_argument("--r5f2-dir", type=Path, default=DEFAULT_R5F2_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
