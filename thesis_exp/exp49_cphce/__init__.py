"""Locked Exp49 CPHCE experiment paths and constants."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP49_ROOT = REPO_ROOT / "thesis_exp" / "exp49_cphce"
SPLIT_ROOT = REPO_ROOT / "thesis_exp" / "data" / "splits" / "paper_like_triple_seed42"
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp49_cphce"
ARTIFACT_ROOT = REPO_ROOT / "thesis_exp" / "artifacts" / "exp49_cphce"
CONFIG_ROOT = REPO_ROOT / "thesis_exp" / "configs" / "exp49_cphce"

EXPECTED_ROWS = {"train": 2654, "dev": 664, "test": 2218}
VARIANTS = ("b0_hard_ce", "m1_human_soft")
FORMAL_SEEDS = (42, 43, 44)
START_COMMIT = "de00041f6c91ffc2acc1f57fe13c93c4678477d5"


def split_path(split: str) -> Path:
    if split not in EXPECTED_ROWS:
        raise ValueError(f"Unknown split: {split}")
    return SPLIT_ROOT / f"{split}.jsonl"


def run_output_dir(variant: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / variant / f"seed_{seed}"


def checkpoint_dir(variant: str, seed: int) -> Path:
    return ARTIFACT_ROOT / variant / f"seed_{seed}" / "best"
