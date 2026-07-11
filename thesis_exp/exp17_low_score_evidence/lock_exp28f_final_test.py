"""Freeze the Exp28 final test campaign without reading the test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_DEV_DECISION = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_reranker_multiseed_dev/"
    "decision/exp28e_multiseed_dev_decision.json"
)
DEFAULT_BOOTSTRAP_DECISION = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28f_paper_dev_statistical_lock/"
    "decision/exp28f_bootstrap_decision.json"
)
DEFAULT_DATA_ROOT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/private/datasets"
)
DEFAULT_CHECKPOINT_ROOT = Path("thesis_exp/artifacts/exp28e_paper_reranker_ce")
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28f_paper_dev_statistical_lock"
)
LOCKED_VARIANTS = ("b0_original_human", "b2_selective_dual_teacher")
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.dev_decision, args.bootstrap_decision):
        if not path.exists():
            raise FileNotFoundError(path)
    dev = json.loads(args.dev_decision.read_text(encoding="utf-8"))
    bootstrap = json.loads(args.bootstrap_decision.read_text(encoding="utf-8"))
    if dev.get("status") != "READY_FOR_BOOTSTRAP_AND_FINAL_DEV_LOCK":
        raise ValueError("Multiseed dev decision does not authorize final locking")
    if bootstrap.get("status") != "READY_FOR_FINAL_DEV_LOCK":
        raise ValueError("Bootstrap decision does not authorize final locking")
    if dev.get("recommended_variant") != "b2_selective_dual_teacher":
        raise ValueError("Unexpected selected main variant")

    registry = []
    for variant in LOCKED_VARIANTS:
        train_path = args.data_root / variant / "train.jsonl"
        dev_path = args.data_root / variant / "dev.jsonl"
        if not train_path.exists() or not dev_path.exists():
            raise FileNotFoundError(f"Missing locked dataset for {variant}")
        if (args.data_root / variant / "test.jsonl").exists():
            raise ValueError("Test dataset was materialized before final lock")
        for seed in SEEDS:
            checkpoint = args.checkpoint_root / variant / f"seed_{seed}" / "best"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            state_dict = checkpoint / "state_dict.pt"
            training_config = checkpoint / "training_config.json"
            dev_metrics = checkpoint / "dev_metrics.json"
            for required in (state_dict, training_config, dev_metrics):
                if not required.exists():
                    raise FileNotFoundError(required)
            checkpoint_config = json.loads(training_config.read_text(encoding="utf-8"))
            if Path(str(checkpoint_config.get("data_dir"))).as_posix() != (
                args.data_root / variant
            ).as_posix():
                raise ValueError(f"Checkpoint data_dir does not match locked variant: {checkpoint}")
            if int(checkpoint_config.get("seed", -1)) != seed:
                raise ValueError(f"Checkpoint seed does not match locked seed: {checkpoint}")
            registry.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "train_sha256": sha256(train_path),
                    "dev_sha256": sha256(dev_path),
                    "checkpoint": str(checkpoint),
                    "checkpoint_state_sha256": sha256(state_dict),
                    "checkpoint_dev_metrics_sha256": sha256(dev_metrics),
                }
            )
    manifest = {
        "status": "READY_FOR_ONE_SHOT_FINAL_TEST",
        "paper_split": "paper_like_triple_seed42",
        "test_rows_expected": 2218,
        "locked_variants": list(LOCKED_VARIANTS),
        "seeds": list(SEEDS),
        "checkpoint_selection": "validation_accuracy",
        "student_input": "question+answer+evaluation_dimension",
        "loss": "ordinary_cross_entropy",
        "test_access_before_lock": 0,
        "test_open_authorized": True,
        "registry": registry,
    }
    path = args.out_dir / "configs" / "exp28f_final_test_lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-decision", type=Path, default=DEFAULT_DEV_DECISION)
    parser.add_argument("--bootstrap-decision", type=Path, default=DEFAULT_BOOTSTRAP_DECISION)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    lock(parse_args())
