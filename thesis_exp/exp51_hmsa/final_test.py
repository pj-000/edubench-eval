"""One-shot frozen test campaign for paired B0 and Exp51 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp49_cphce.build_targets import aggregate_text_hash, load_split
from thesis_exp.exp49_cphce.metric_contract import compute_metrics
from thesis_exp.exp51_hmsa import FORMAL_SEEDS, OUTPUT_ROOT, baseline_run_dir, checkpoint_dir, run_output_dir
from thesis_exp.exp51_hmsa.formal_gate import verify_formal_lock
from thesis_exp.exp51_hmsa.train import evaluate_dual, load_dual_model
from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    TrainConfig,
    evaluate,
    load_model_and_tokenizer,
    make_dataloader,
    set_seed,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_jsonl, write_text


MODEL_PATH = "/home/share/models/modelscope/Qwen/Qwen3-Reranker-0.6B"
FINAL_LOCK_PATH = Path(__file__).resolve().parents[1] / "configs" / "exp51_hmsa" / "final_test_lock.json"
FINAL_OUTPUT_ROOT = OUTPUT_ROOT / "final_test"
CAMPAIGN_STATE_PATH = FINAL_OUTPUT_ROOT / "campaign_state.json"
FORMAL_DECISION_PATH = OUTPUT_ROOT / "decision" / "formal_decision.json"
ARMS = ("b0", "exp51")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_path(arm: str, seed: int) -> Path:
    if arm == "b0":
        return Path(__file__).resolve().parents[2] / "thesis_exp" / "artifacts" / "exp49_cphce" / "b0_hard_ce" / f"seed_{seed}" / "best" / "state_dict.pt"
    if arm == "exp51":
        return checkpoint_dir(seed) / "state_dict.pt"
    raise ValueError(f"Unknown arm: {arm}")


def result_key(arm: str, seed: int) -> str:
    return f"{arm}/seed_{seed}"


def result_dir(arm: str, seed: int, split: str) -> Path:
    root = FINAL_OUTPUT_ROOT if split == "test" else OUTPUT_ROOT / "audit" / "final_test_dev_smoke"
    return root / arm / f"seed_{seed}"


def verify_final_lock(*, include_checkpoints: bool) -> dict[str, Any]:
    lock = read_json(FINAL_LOCK_PATH)
    repo_root = Path(__file__).resolve().parents[2]
    for relative, expected in lock["files"].items():
        actual = sha256_file(repo_root / relative)
        if actual != expected:
            raise ValueError(f"Final-test source lock mismatch: {relative}: {actual} != {expected}")
    if include_checkpoints:
        for key, frozen in lock["checkpoints"].items():
            path = repo_root / frozen["path"]
            if path.stat().st_size != int(frozen["size_bytes"]):
                raise ValueError(f"Checkpoint size mismatch: {key}")
            actual = sha256_file(path)
            if actual != frozen["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {key}: {actual} != {frozen['sha256']}")
    lock["manifest_sha256"] = sha256_file(FINAL_LOCK_PATH)
    return lock


def verify_formal_authorization() -> dict[str, Any]:
    verify_formal_lock()
    decision = read_json(FORMAL_DECISION_PATH)
    if decision.get("status") != "EXP51_FORMAL_PASS" or not decision.get("passed"):
        raise PermissionError("Exp51 final test requires EXP51_FORMAL_PASS")
    if int(decision.get("test_access_count", -1)) != 0:
        raise PermissionError("Formal dev decision has unexpected test access")
    return decision


def begin_campaign() -> dict[str, Any]:
    lock = verify_final_lock(include_checkpoints=True)
    decision = verify_formal_authorization()
    if lock.get("status") != "EXP51_FINAL_TEST_LOCKED" or int(lock.get("test_access_before_campaign", -1)) != 0:
        raise PermissionError("Final-test lock is not in the pre-campaign state")
    if CAMPAIGN_STATE_PATH.exists():
        raise PermissionError(f"Final-test campaign state already exists: {CAMPAIGN_STATE_PATH}")
    state = {
        "status": "TEST_IN_PROGRESS",
        "test_access_count": 1,
        "final_lock_manifest_sha256": lock["manifest_sha256"],
        "formal_decision_sha256": sha256_file(FORMAL_DECISION_PATH),
        "formal_status": decision["status"],
        "test_anchor": None,
    }
    write_json(CAMPAIGN_STATE_PATH, state)
    return state


def require_in_progress() -> dict[str, Any]:
    state = read_json(CAMPAIGN_STATE_PATH)
    if state.get("status") != "TEST_IN_PROGRESS" or int(state.get("test_access_count", -1)) != 1:
        raise PermissionError("Final-test campaign is not in the one-shot in-progress state")
    lock = verify_final_lock(include_checkpoints=False)
    if state["final_lock_manifest_sha256"] != lock["manifest_sha256"]:
        raise ValueError("Final-test lock changed after campaign authorization")
    return state


def anchor_test() -> dict[str, Any]:
    state = require_in_progress()
    if state.get("test_anchor") is not None:
        raise PermissionError("Test split has already been anchored for this campaign")
    test_path = Path(__file__).resolve().parents[2] / "thesis_exp" / "data" / "splits" / "paper_like_triple_seed42" / "test.jsonl"
    digest = hashlib.sha256()
    row_count = 0
    with test_path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            row_count += int(bool(line.strip()))
    if row_count != 2218:
        raise ValueError(f"Unexpected test row count: {row_count}")
    state["test_anchor"] = {
        "path": str(test_path.relative_to(Path(__file__).resolve().parents[2])),
        "sha256": digest.hexdigest(),
        "size_bytes": test_path.stat().st_size,
        "rows": row_count,
    }
    write_json(CAMPAIGN_STATE_PATH, state)
    return state


def eval_config(seed: int, output_dir: Path, max_eval_samples: int | None) -> TrainConfig:
    return TrainConfig(
        model_name_or_path=MODEL_PATH,
        data_dir=Path("EXP51_FINAL_TEST_IN_MEMORY"),
        output_dir=output_dir,
        checkpoint_output_dir=output_dir,
        max_length=2048,
        num_train_epochs=10.0,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32,
        max_grad_norm=1.0,
        seed=seed,
        bf16="auto",
        fp16=False,
        gradient_checkpointing=False,
        trust_remote_code=False,
        local_files_only=True,
        max_train_samples=None,
        max_eval_samples=max_eval_samples,
        eval_only=True,
        checkpoint_dir=None,
        num_workers=0,
        log_steps=20,
        progress_bar=False,
        evaluate_test=False,
    )


def verify_one_checkpoint(arm: str, seed: int, lock: dict[str, Any]) -> Path:
    key = result_key(arm, seed)
    frozen = lock["checkpoints"][key]
    path = checkpoint_path(arm, seed)
    if str(path.relative_to(Path(__file__).resolve().parents[2])) != frozen["path"]:
        raise ValueError(f"Checkpoint path mismatch: {key}")
    if path.stat().st_size != int(frozen["size_bytes"]) or sha256_file(path) != frozen["sha256"]:
        raise ValueError(f"Frozen checkpoint verification failed: {key}")
    return path


def evaluate_checkpoint(arm: str, seed: int, *, split: str, max_eval_samples: int | None = None) -> dict[str, Any]:
    if arm not in ARMS or seed not in FORMAL_SEEDS:
        raise ValueError(f"Unknown frozen run: {arm} seed{seed}")
    if split not in ("dev", "test"):
        raise ValueError(f"Unsupported split: {split}")
    if split == "test":
        if max_eval_samples is not None:
            raise PermissionError("Partial final-test evaluation is forbidden")
        state = require_in_progress()
        if state.get("test_anchor") is None:
            raise PermissionError("Test split must be anchored after one-shot authorization")
        lock = verify_final_lock(include_checkpoints=False)
        checkpoint = verify_one_checkpoint(arm, seed, lock)
        selected_epoch = int(lock["checkpoints"][result_key(arm, seed)]["selected_epoch"])
        final_lock_manifest = lock["manifest_sha256"]
    else:
        verify_formal_authorization()
        checkpoint = checkpoint_path(arm, seed)
        summary_path = (baseline_run_dir(seed) if arm == "b0" else run_output_dir(seed)) / "run_summary.json"
        selected_epoch = int(read_json(summary_path)["selected_epoch"])
        final_lock_manifest = None
    output_dir = result_dir(arm, seed, split)
    if split == "test" and output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite final-test output: {output_dir}")
    write_dir = output_dir if split == "dev" else output_dir.parent / f".{output_dir.name}.staging.{os.getpid()}"
    if write_dir.exists():
        raise FileExistsError(f"Evaluation staging directory already exists: {write_dir}")
    config = eval_config(seed, output_dir, max_eval_samples)
    set_seed(seed)
    if arm == "b0":
        model, tokenizer, model_mode = load_model_and_tokenizer(config)
    else:
        model, tokenizer, model_mode, _ = load_dual_model(config)
    import torch

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = load_split(split, allow_test=split == "test")
    if max_eval_samples is not None:
        rows = rows[:max_eval_samples]
    loader = make_dataloader(rows, tokenizer, config, split, shuffle=False)
    result = evaluate(model, loader, device, split) if arm == "b0" else evaluate_dual(model, loader, device, split)
    metrics = {
        "arm": arm,
        "seed": seed,
        "split": split,
        "model_mode": model_mode,
        "selected_epoch": selected_epoch,
        "inference": "raw_logit_argmax" if arm == "b0" else "hard_head_raw_logit_argmax",
        "final_lock_manifest_sha256": final_lock_manifest,
        "checkpoint_sha256": sha256_file(checkpoint),
        "input_text_hash": aggregate_text_hash(rows),
        "test_file_sha256": state["test_anchor"]["sha256"] if split == "test" else None,
        "test_access_count": 1 if split == "test" else 0,
        **compute_metrics(result.predictions),
    }
    if arm == "b0":
        metrics["eval_hard_ce_loss"] = float(result.metrics["loss"])
    else:
        for key in ("eval_hard_ce_loss", "soft_aux_ce", "soft_aux_jsd", "soft_aux_rps", "soft_aux_entropy", "soft_aux_argmax_exact"):
            metrics[key] = float(result.metrics[key])
    write_dir.mkdir(parents=True, exist_ok=True)
    write_json(write_dir / f"{split}_metrics.json", metrics)
    write_csv(write_dir / f"{split}_metrics.csv", [metrics])
    write_jsonl(write_dir / f"predictions_{split}.jsonl", result.predictions)
    if split == "test":
        write_dir.replace(output_dir)
    return metrics


def paired_test_deltas(seed: int) -> np.ndarray:
    def rows(arm: str) -> dict[str, dict[str, Any]]:
        path = result_dir(arm, seed, "test") / "predictions_test.jsonl"
        return {str(row["record_id"]): row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}

    b0, exp51 = rows("b0"), rows("exp51")
    if set(b0) != set(exp51) or len(b0) != 2218:
        raise ValueError(f"Final-test prediction ID mismatch for seed {seed}")
    return np.asarray(
        [abs(float(exp51[key]["pred_label_5"]) - float(b0[key]["human_mean_5"])) - abs(float(b0[key]["pred_label_5"]) - float(b0[key]["human_mean_5"])) for key in sorted(b0)],
        dtype=float,
    )


def summarize_test() -> dict[str, Any]:
    state = require_in_progress()
    metrics = {(arm, seed): read_json(result_dir(arm, seed, "test") / "test_metrics.json") for arm in ARMS for seed in FORMAL_SEEDS}
    keys = (
        "Exact_rounded",
        "MAE_human_mean",
        "Bias_human_mean",
        "Kendall_human_mean",
        "L2H_count",
        "Recall_4_rounded",
        "Recall_5_rounded",
    )
    per_seed: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        b0, exp51 = metrics[("b0", seed)], metrics[("exp51", seed)]
        if int(b0["n"]) != 2218 or int(exp51["n"]) != 2218 or b0["input_text_hash"] != exp51["input_text_hash"]:
            raise ValueError(f"Final-test contract mismatch for seed {seed}")
        row: dict[str, Any] = {"seed": seed}
        for key in keys:
            row[f"b0_{key}"] = b0[key]
            row[f"exp51_{key}"] = exp51[key]
        row.update(
            {
                "mae_improvement": float(b0["MAE_human_mean"]) - float(exp51["MAE_human_mean"]),
                "delta_exact": float(exp51["Exact_rounded"]) - float(b0["Exact_rounded"]),
                "delta_kendall": float(exp51["Kendall_human_mean"]) - float(b0["Kendall_human_mean"]),
                "delta_l2h_count": int(exp51["L2H_count"]) - int(b0["L2H_count"]),
                "abs_bias_improvement": abs(float(b0["Bias_human_mean"])) - abs(float(exp51["Bias_human_mean"])),
            }
        )
        per_seed.append(row)
    numeric_keys = [key for key, value in per_seed[0].items() if key != "seed" and isinstance(value, (int, float))]
    means = {key: float(np.mean([float(row[key]) for row in per_seed])) for key in numeric_keys}
    standard_deviations = {key: float(np.std([float(row[key]) for row in per_seed], ddof=1)) for key in numeric_keys}
    deltas = np.stack([paired_test_deltas(seed) for seed in FORMAL_SEEDS], axis=0).mean(axis=0)
    rng = np.random.default_rng(42)
    bootstrap = np.empty(10_000)
    for index in range(len(bootstrap)):
        bootstrap[index] = float(deltas[rng.integers(0, len(deltas), len(deltas))].mean())
    summary = {
        "status": "EXP51_FINAL_TEST_COMPLETED",
        "test_access_count": 1,
        "test_anchor": state["test_anchor"],
        "per_seed": per_seed,
        "mean": means,
        "standard_deviation": standard_deviations,
        "paired_bootstrap": {
            "mean_delta_mae": float(deltas.mean()),
            "ci_low": float(np.quantile(bootstrap, 0.025)),
            "ci_high": float(np.quantile(bootstrap, 0.975)),
            "resamples": len(bootstrap),
        },
        "finite": all(math.isfinite(float(value)) for row in per_seed for value in row.values() if isinstance(value, (int, float))),
    }
    write_json(FINAL_OUTPUT_ROOT / "final_test_summary.json", summary)
    write_csv(FINAL_OUTPUT_ROOT / "final_test_comparison.csv", per_seed)
    write_text(
        FINAL_OUTPUT_ROOT / "final_test_summary.md",
        "# Exp51 one-shot final test\n\n"
        f"- Mean MAE improvement: {means['mae_improvement']:.6f}\n"
        f"- Mean Exact delta: {means['delta_exact']:.6f}\n"
        f"- Mean Kendall delta: {means['delta_kendall']:.6f}\n"
        "- Test campaign access count: 1\n",
    )
    return summary


def mark_complete() -> dict[str, Any]:
    state = require_in_progress()
    summary = read_json(FINAL_OUTPUT_ROOT / "final_test_summary.json")
    if summary.get("status") != "EXP51_FINAL_TEST_COMPLETED" or int(summary.get("test_access_count", -1)) != 1:
        raise PermissionError("Final-test summary is incomplete")
    state["status"] = "TEST_COMPLETED"
    state["summary_sha256"] = sha256_file(FINAL_OUTPUT_ROOT / "final_test_summary.json")
    write_json(CAMPAIGN_STATE_PATH, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", action="store_true")
    action.add_argument("--begin", action="store_true")
    action.add_argument("--anchor-test", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--mark-complete", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--max-eval-samples", type=int)
    args = parser.parse_args()
    if args.verify:
        output = verify_final_lock(include_checkpoints=True)
    elif args.begin:
        output = begin_campaign()
    elif args.anchor_test:
        output = anchor_test()
    elif args.evaluate:
        if args.arm is None or args.seed is None:
            parser.error("--evaluate requires --arm and --seed")
        output = evaluate_checkpoint(args.arm, args.seed, split=args.split, max_eval_samples=args.max_eval_samples)
    elif args.summarize:
        output = summarize_test()
    else:
        output = mark_complete()
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
