"""Prepare Exp23 R7 DPO scout configs.

Exp23 is a small training workflow, not a new data construction step. It uses
the already-reviewed R7D and Exp22 matched-control datasets to answer one
sanity question before formula-level ORC/SRC-DPO work:

Does chosen-side human rationale help ordinary DPO beyond an exactly matched
score-only control?

No dev/test labels are read here. Existing R7/Exp22 leakage reports are only
summarized as prior guardrail evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.utils.io import count_json_records, write_csv, write_json, write_text  # noqa: E402


DEFAULT_R7_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_r7_reason_recovered_real_dpo_seed42")
DEFAULT_EXP22_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp22_r7_matched_controls_seed42")
DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp23_r7_dpo_scout")
DEFAULT_MODEL = "/home/jpang/models/modelscope/Qwen/Qwen3-4B"
DEFAULT_INIT_ADAPTER = "saves/edubench/qwen3-4b/r2_clean_reason_score_balanced_lora"
DEFAULT_SAVE_ROOT = "saves/edubench/qwen3-4b"


RUNS = [
    {
        "run_name": "r7d_reason_real_s100_b0p03_lr5em6",
        "dataset_family": "human_reason_real_error",
        "dataset_dir_kind": "r7",
        "dataset": "edubench_r7d_strict_label_consistent_reason_real_dpo_train",
        "description": "R7D chosen has recovered human rationale plus gold score; rejected is real wrong score.",
    },
    {
        "run_name": "r7e_matched_score_only_s100_b0p03_lr5em6",
        "dataset_family": "matched_score_only_control",
        "dataset_dir_kind": "exp22",
        "dataset": "edubench_r7e_matched_score_only_strict_real_dpo_train",
        "description": "R7E uses the exact R7D pair pool but removes chosen rationale.",
    },
    {
        "run_name": "r7f_score_reason_consistency_s100_b0p03_lr5em6",
        "dataset_family": "score_reason_consistency_counterfactual",
        "dataset_dir_kind": "exp22",
        "dataset": "edubench_r7f_score_reason_consistency_dpo_train",
        "description": "R7F trains consistency between recovered human reason and final score.",
    },
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dataset_path(dataset_dir: Path, dataset_name: str) -> Path:
    info_path = dataset_dir / "dataset_info.json"
    info = read_json(info_path)
    if dataset_name not in info:
        raise SystemExit(f"dataset {dataset_name} missing from {info_path}")
    file_name = info[dataset_name].get("file_name")
    if not file_name:
        raise SystemExit(f"dataset {dataset_name} has no file_name in {info_path}")
    path = dataset_dir / file_name
    if not path.exists():
        raise SystemExit(f"dataset file missing for {dataset_name}: {path}")
    return path


def make_config(args: argparse.Namespace, run: dict[str, str], dataset_dir: Path) -> dict[str, Any]:
    return {
        "model_name_or_path": args.model_name_or_path,
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "adapter_name_or_path": args.init_adapter,
        "ref_model": args.model_name_or_path,
        "ref_model_adapters": args.init_adapter,
        "dataset_dir": str(dataset_dir),
        "template": "qwen3_nothink",
        "cutoff_len": 4096,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "pref_loss": "sigmoid",
        "pref_beta": args.pref_beta,
        "pref_ftx": args.pref_ftx,
        "logging_steps": 10,
        "save_steps": args.max_steps,
        "save_only_model": True,
        "plot_loss": True,
        "report_to": "none",
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "bf16": True,
        "gradient_checkpointing": True,
        "dataset": run["dataset"],
        "output_dir": f"{args.save_root}/exp23_{run['run_name']}",
    }


def write_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    config_dir = args.out_dir / "configs"
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        source_dir = args.r7_dir if run["dataset_dir_kind"] == "r7" else args.exp22_dir
        data_path = dataset_path(source_dir, run["dataset"])
        config = make_config(args, run, source_dir)
        config_path = config_dir / f"llamafactory_qwen3_4b_{run['run_name']}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        rows.append(
            {
                "run_name": run["run_name"],
                "dataset_family": run["dataset_family"],
                "dataset": run["dataset"],
                "dataset_dir": str(source_dir),
                "data_file": str(data_path),
                "pair_count": count_json_records(data_path),
                "init_adapter": args.init_adapter,
                "output_dir": config["output_dir"],
                "config_path": str(config_path),
                "max_steps": args.max_steps,
                "pref_beta": args.pref_beta,
                "pref_ftx": args.pref_ftx,
                "learning_rate": args.learning_rate,
                "description": run["description"],
            }
        )
    write_csv(args.out_dir / "tables" / "exp23_run_matrix.csv", rows)
    return rows


def summarize_leakage(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, path in [
        ("r7", args.r7_dir / "tables" / "r7_leakage_audit.csv"),
        ("exp22", args.exp22_dir / "tables" / "exp22_leakage_audit.csv"),
    ]:
        for row in read_csv_if_exists(path):
            out = {"source": source}
            out.update(row)
            rows.append(out)
    write_csv(args.out_dir / "tables" / "exp23_prior_leakage_guardrails.csv", rows)
    return rows


def write_report(args: argparse.Namespace, run_rows: list[dict[str, Any]], leakage_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Exp23 R7 DPO Scout Plan",
        "",
        "Exp23 trains ordinary DPO scouts to check whether recovered human rationale is useful before",
        "introducing ORC-DPO or SRC-DPO formula changes.",
        "",
        "## Runs",
        "",
        "| run | family | pairs | purpose |",
        "|---|---|---:|---|",
    ]
    for row in run_rows:
        lines.append(
            f"| `{row['run_name']}` | `{row['dataset_family']}` | {row['pair_count']} | {row['description']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- R7D vs R7E is the key fair comparison because they share the same source pair pool.",
            "- R7F is not a natural real-error dataset; it is a score-reason consistency auxiliary scout.",
            "- This step uses ordinary DPO only. It should not be described as the final algorithmic method.",
            "",
            "## Training Defaults",
            "",
            f"- init adapter: `{args.init_adapter}`",
            f"- max_steps: {args.max_steps}",
            f"- pref_beta: {args.pref_beta}",
            f"- pref_ftx: {args.pref_ftx}",
            f"- learning_rate: {args.learning_rate}",
            f"- per_device_train_batch_size: {args.per_device_train_batch_size}",
            f"- gradient_accumulation_steps: {args.gradient_accumulation_steps}",
            "",
            "## Guardrails",
            "",
            "- Training datasets are train-only DPO data.",
            "- This preparation script does not read dev/test labels.",
            "- Existing R7/Exp22 leakage audits are copied only as prior guardrail evidence.",
            "- Do not submit checkpoints, raw predictions, logs, full generated outputs, numpy arrays, or model weights.",
            "",
            "## Prior Leakage Summary",
            "",
        ]
    )
    if leakage_rows:
        lines.extend(
            [
                "| source | dataset | pairs | dev sample overlap | dev question overlap | test sample overlap | test question overlap | reason in prompt | pass |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in leakage_rows:
            dataset = row.get("dataset_variant") or row.get("dataset_name") or ""
            lines.append(
                f"| `{row.get('source', '')}` | `{dataset}` | {row.get('pairs', '')} | "
                f"{row.get('dev_sample_id_overlap', '')} | {row.get('dev_question_key_overlap', '')} | "
                f"{row.get('test_sample_id_overlap', '')} | {row.get('test_question_key_overlap', '')} | "
                f"{row.get('human_reason_in_prompt_count', '')} | {row.get('leakage_pass', '')} |"
            )
    else:
        lines.append("No prior leakage rows found.")
    write_text(args.out_dir / "reports" / "exp23_r7_dpo_scout_plan.md", "\n".join(lines))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = write_configs(args)
    leakage_rows = summarize_leakage(args)
    decision = {
        "status": "READY_FOR_SCOUT_TRAINING",
        "primary_comparison": "r7d_reason_real_s100_b0p03_lr5em6 vs r7e_matched_score_only_s100_b0p03_lr5em6",
        "auxiliary_run": "r7f_score_reason_consistency_s100_b0p03_lr5em6",
        "requires_gpu_for_training": True,
        "test_read": False,
        "dev_labels_used_for_training": False,
        "human_reason_in_prompt": False,
        "next_command": "./thesis_exp/scripts/run_exp23_r7_dpo_scout_train.sh",
    }
    write_json(args.out_dir / "decision" / "exp23_r7_dpo_scout_prepare_decision.json", decision)
    write_report(args, run_rows, leakage_rows)
    return {
        "out_dir": str(args.out_dir),
        "runs": len(run_rows),
        "pair_counts": {row["run_name"]: row["pair_count"] for row in run_rows},
        "decision": decision["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp23 R7 DPO scout configs.")
    parser.add_argument("--r7-dir", type=Path, default=DEFAULT_R7_DIR)
    parser.add_argument("--exp22-dir", type=Path, default=DEFAULT_EXP22_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--init-adapter", default=DEFAULT_INIT_ADAPTER)
    parser.add_argument("--save-root", default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--pref-beta", type=float, default=0.03)
    parser.add_argument("--pref-ftx", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
