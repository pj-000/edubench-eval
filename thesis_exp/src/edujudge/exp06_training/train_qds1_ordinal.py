"""Train Exp6 QD-S1 human + synthetic low-score ordinary ordinal model."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp04.train_objective import (
    ObjectiveTrainConfig,
    grouped_metrics,
    train,
)
from thesis_exp.src.edujudge.exp06_training import (
    QD_S1_CHECKPOINT_DIR,
    QD_S1_DATASET_DIR,
    QD_S1_RUN_DIR,
    QD_S1_RUN_ID,
    QUESTION_SPLIT_DIR,
    ensure_exp06_training_dirs,
)
from thesis_exp.src.edujudge.exp06_training.collect_qds1_results import collect
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_text


FORBIDDEN_TEXT_MARKERS = [
    "rationale_for_label",
    "expected_failure_against_rubric",
    "error_type",
    "label_source",
]
API_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{12,}")
CHECKPOINT_EXTENSIONS = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}


def _failures_to_text(rows: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {row['check_name']}: {row['status']} ({row.get('details', '')})" for row in rows if row["status"] != "PASS")


def _read_split(split: str) -> list[dict[str, Any]]:
    return read_jsonl(QD_S1_DATASET_DIR / f"{split}.jsonl")


def _split_keys(split: str, field: str) -> set[str]:
    return {str(row.get(field)) for row in read_jsonl(QUESTION_SPLIT_DIR / f"{split}.jsonl") if row.get(field)}


def _tracked_weight_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return sorted(path for path in result.stdout.splitlines() if Path(path).suffix.lower() in CHECKPOINT_EXTENSIONS)


def _api_key_hits(path: Path) -> list[str]:
    if not path.exists():
        return []
    hits: list[str] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file() or candidate.stat().st_size > 20_000_000:
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if API_KEY_PATTERN.search(text):
            hits.append(relpath(candidate))
    return sorted(hits)


def dataset_sanity_rows(max_train_samples: int | None, max_eval_samples: int | None, formal: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check_name: str, passed: bool, details: Any = "") -> None:
        rows.append({"check_name": check_name, "status": "PASS" if passed else "FAIL", "details": details})

    train_rows = _read_split("train")
    dev_rows = _read_split("dev")
    test_rows = _read_split("test")
    train_source_counts = {}
    for row in train_rows:
        train_source_counts[row.get("source_type")] = train_source_counts.get(row.get("source_type"), 0) + 1

    add("QD-S1 train/dev/test rows = 3710/1107/1103", [len(train_rows), len(dev_rows), len(test_rows)] == [3710, 1107, 1103], [len(train_rows), len(dev_rows), len(test_rows)])
    add("dev synthetic rows = 0", sum(1 for row in dev_rows if row.get("source_type") == "synthetic") == 0)
    add("test synthetic rows = 0", sum(1 for row in test_rows if row.get("source_type") == "synthetic") == 0)
    add("train synthetic rows = 384", train_source_counts.get("synthetic", 0) == 384, train_source_counts.get("synthetic", 0))
    add("train human rows = 3326", train_source_counts.get("human", 0) == 3326, train_source_counts.get("human", 0))

    dev_test_q = _split_keys("dev", "question_key") | _split_keys("test", "question_key")
    dev_test_t = _split_keys("dev", "triple_key") | _split_keys("test", "triple_key")
    synthetic_rows = [row for row in train_rows if row.get("source_type") == "synthetic"]
    q_overlap = sorted({str(row.get("source_question_key")) for row in synthetic_rows if str(row.get("source_question_key")) in dev_test_q})
    t_overlap = sorted({str(row.get("source_triple_key")) for row in synthetic_rows if str(row.get("source_triple_key")) in dev_test_t})
    add("synthetic source question overlap with dev/test = 0", not q_overlap, len(q_overlap))
    add("synthetic source triple overlap with dev/test = 0", not t_overlap, len(t_overlap))

    marker_hits = {marker: 0 for marker in FORBIDDEN_TEXT_MARKERS}
    for row in train_rows:
        text = str(row.get("text") or "")
        for marker in FORBIDDEN_TEXT_MARKERS:
            if marker in text:
                marker_hits[marker] += 1
    add("train text excludes rationale/error_type/label_source markers", all(value == 0 for value in marker_hits.values()), marker_hits)

    add("formal run has no max sample limits", not formal or (max_train_samples is None and max_eval_samples is None), f"max_train_samples={max_train_samples}, max_eval_samples={max_eval_samples}")
    add("synthetic label provenance pseudo_label", all(row.get("label_provenance") == "pseudo_label" for row in synthetic_rows))
    add("dev/test human label provenance human_score", all(row.get("label_provenance") == "human_score" for row in dev_rows + test_rows))
    add("no tracked checkpoint/weight files", not _tracked_weight_files(), ",".join(_tracked_weight_files()))
    return rows


def require_cuda_if_requested() -> None:
    if os.environ.get("REQUIRE_CUDA", "0") != "1":
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - server environment check
        raise RuntimeError(f"REQUIRE_CUDA=1 but torch import failed: {type(exc).__name__}: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("REQUIRE_CUDA=1 but torch.cuda.is_available() is false.")


def write_language_metrics(run_dir: Path) -> None:
    tables_dir = run_dir / "tables"
    rows: list[dict[str, Any]] = []
    for split in ["dev", "test"]:
        predictions_path = run_dir / "predictions" / f"predictions_{split}.jsonl"
        if predictions_path.exists():
            rows.extend(grouped_metrics(read_jsonl(predictions_path), split, "language", "ordinal"))
    if rows:
        write_csv(tables_dir / "language_level_metrics.csv", rows)


def copy_dev_history_to_logs(run_dir: Path) -> None:
    src = run_dir / "tables" / "dev_metrics_history.csv"
    dst = run_dir / "logs" / "dev_metrics_history.csv"
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def postprocess_outputs(run_dir: Path, strict: bool = True) -> None:
    write_language_metrics(run_dir)
    copy_dev_history_to_logs(run_dir)
    api_hits = _api_key_hits(run_dir)
    if api_hits and strict:
        raise RuntimeError(f"API key-like strings found in outputs: {api_hits}")
    collect(run_dir)


def train_qds1(args: argparse.Namespace) -> None:
    ensure_exp06_training_dirs()
    formal = os.environ.get("FORMAL_RUN", "0") == "1"
    smoke = bool(args.smoke)
    if smoke and formal:
        raise RuntimeError("Smoke run must use FORMAL_RUN=0.")
    if not smoke and not formal and not args.postprocess_only and not args.preflight_only:
        raise RuntimeError("Formal QD-S1 training requires FORMAL_RUN=1.")

    sanity_rows = dataset_sanity_rows(args.max_train_samples, args.max_eval_samples, formal)
    sanity_path = args.output_dir / "tables" / ("smoke_preflight_qds1.csv" if smoke else "preflight_qds1.csv")
    write_csv(sanity_path, sanity_rows)
    failures = [row for row in sanity_rows if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"QD-S1 preflight failed:\n{_failures_to_text(failures)}")
    if args.preflight_only:
        write_text(
            args.output_dir / "qds1_preflight_status.md",
            "\n".join(
                [
                    "# QD-S1 Preflight Status",
                    "",
                    "Status: PASS",
                    f"Formal mode: {formal}",
                    f"Smoke mode: {smoke}",
                    "Training executed: no",
                    "API called: no",
                    "Synthetic generated: no",
                ]
            ),
        )
        return
    if args.postprocess_only:
        postprocess_outputs(args.output_dir)
        return

    require_cuda_if_requested()
    config = ObjectiveTrainConfig(
        objective_id=QD_S1_RUN_ID,
        objective_type="ordinal",
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_output_dir=args.checkpoint_output_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        eval_only=args.eval_only,
        checkpoint_dir=args.checkpoint_dir,
        regression_loss="none",
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
        num_workers=args.num_workers,
        log_steps=args.log_steps,
        progress_bar=args.progress_bar,
    )
    train(config)
    postprocess_outputs(args.output_dir)
    write_text(
        args.output_dir / "qds1_training_status.md",
        "\n".join(
            [
                "# QD-S1 Training Status",
                "",
                "Status: completed",
                f"Smoke: {smoke}",
                "API called: no",
                "Synthetic generated: no",
                "Loss: ordinary ordinal BCEWithLogitsLoss",
                "Synthetic labels: pseudo_label metadata only",
            ]
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Exp6 QD-S1 ordinary ordinal model.")
    parser.add_argument("--model_name_or_path", default="/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--data_dir", type=Path, default=QD_S1_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=QD_S1_RUN_DIR)
    parser.add_argument("--checkpoint_output_dir", type=Path, default=QD_S1_CHECKPOINT_DIR)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--selection_metric", default="MAE_label")
    parser.add_argument("--selection_mode", choices=["min", "max"], default="min")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--postprocess_only", action="store_true")
    parser.add_argument("--no_progress_bar", action="store_false", dest="progress_bar")
    parser.set_defaults(progress_bar=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_qds1(args)
    print(f"QD-S1 output: {relpath(args.output_dir)}")


if __name__ == "__main__":
    main()
