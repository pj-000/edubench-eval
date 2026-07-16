"""Aggregate existing Exp46 teacher optimization histories and adaptation config."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from thesis_exp.exp47_label2_identifiability.common import (
    EXPECTED_FOLDS,
    EXP44_RUN_ROOT,
    EXP46_PUBLIC_ROOT,
    EXP46_RUN_ROOT,
    ROOT,
    ensure_dirs,
    sanitize_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--exp44-run-root", type=Path, default=EXP44_RUN_ROOT)
    parser.add_argument("--exp46-run-root", type=Path, default=EXP46_RUN_ROOT)
    return parser.parse_args()


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {"history_status": "MISSING_PRIVATE_HISTORY"}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    history_rows = []
    parameter_rows = []
    teacher_by_epoch: dict[int, list[dict]] = defaultdict(list)
    for model, root, variant in (
        ("M0_0.6B_E4", args.exp44_run_root, "C0_E4_baseline"),
        ("M1_4B_teacher", args.exp46_run_root, "T1_4B_teacher"),
    ):
        for fold in EXPECTED_FOLDS:
            summary_path = root / f"groupcv/{variant}/seed_42/fold_{fold}/run_summary.json"
            summary = read_summary(summary_path)
            history = summary.get("history", [])
            if not history:
                history_rows.append({"model": model, "record_type": "fold_epoch", "fold": fold, "epoch": "", "history_status": "MISSING_PRIVATE_HISTORY"})
            for row in history:
                output = {
                    "model": model,
                    "record_type": "fold_epoch",
                    "fold": fold,
                    "epoch": row.get("epoch"),
                    "history_status": "AVAILABLE",
                    "global_step": row.get("global_step"),
                    "learning_rate": row.get("learning_rate"),
                    "train_total_loss": row.get("train_loss"),
                    "train_human_loss": row.get("train_human"),
                    "train_human_distribution_loss": row.get("train_human_distribution", row.get("train_distribution")),
                    "train_ordinal_loss": row.get("train_human_ordinal", row.get("train_ordinal")),
                }
                history_rows.append(output)
                if model == "M1_4B_teacher":
                    teacher_by_epoch[int(row["epoch"])].append(output)
            identity = summary.get("run_identity", {}).get("config", {})
            counts = summary.get("parameter_counts", {})
            parameter_rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "history_status": "AVAILABLE" if history else "MISSING_PRIVATE_HISTORY",
                    "model_mode": summary.get("model_mode", "unknown"),
                    "total_parameters": counts.get("total"),
                    "trainable_parameters": counts.get("trainable"),
                    "trainable_fraction": counts.get("trainable", 0) / counts.get("total", 1) if counts else None,
                    "peak_gpu_memory_mib": summary.get("peak_gpu_memory_mib"),
                    "epochs": identity.get("epochs"),
                    "fixed_final_epoch": summary.get("fixed_final_epoch"),
                    "global_steps": summary.get("global_step"),
                    "learning_rate": identity.get("learning_rate"),
                    "batch_size": identity.get("batch_size"),
                    "gradient_accumulation": identity.get("gradient_accumulation"),
                    "max_length": identity.get("max_length"),
                    "nan_count": summary.get("nan_count"),
                    "oom_count": summary.get("oom_count"),
                    "dev_access_count": summary.get("dev_access_count", 0),
                    "test_access_count": summary.get("test_access_count", 0),
                }
            )

    for epoch, rows in sorted(teacher_by_epoch.items()):
        aggregate = {"model": "M1_4B_teacher", "record_type": "epoch_fold_aggregate", "fold": "all", "epoch": epoch, "history_status": "AVAILABLE"}
        for key in ("global_step", "learning_rate", "train_total_loss", "train_human_loss", "train_human_distribution_loss", "train_ordinal_loss"):
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            aggregate[f"{key}_mean"] = float(np.mean(values)) if values else None
            aggregate[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        history_rows.append(aggregate)

    method_lock = json.loads((EXP46_PUBLIC_ROOT / "configs/exp46a_method_lock.json").read_text(encoding="utf-8"))
    for row in parameter_rows:
        if row["model"] == "M1_4B_teacher":
            lora = method_lock["teacher"]["lora"]
            row.update(
                {
                    "lora_rank": lora["rank"],
                    "lora_alpha": lora["alpha"],
                    "lora_dropout": lora["dropout"],
                    "lora_target_modules": "|".join(lora["target_modules"]),
                }
            )
    write_csv(args.out_dir / "tables/exp47_teacher_optimization_history.csv", sanitize_rows(history_rows))
    write_csv(args.out_dir / "tables/exp47_teacher_trainable_parameter_audit.csv", sanitize_rows(parameter_rows))
    print(json.dumps({"status": "OPTIMIZATION_AUDITED", "teacher_history_folds": sum(row["model"] == "M1_4B_teacher" and row["history_status"] == "AVAILABLE" for row in parameter_rows), "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
