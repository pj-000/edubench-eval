"""Within-label shuffled-soft mechanism control for the HMSA paper."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp55_within_label_shuffle"
ARTIFACT_ROOT = REPO_ROOT / "thesis_exp" / "artifacts" / "exp55_within_label_shuffle"

VARIANT = "within_label_shuffled_soft"
MODEL_SEEDS = (42, 43, 44)
SHUFFLE_SEED = 20260730


def run_output_dir(seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / VARIANT / f"seed_{seed}"


def checkpoint_dir(seed: int) -> Path:
    return ARTIFACT_ROOT / VARIANT / f"seed_{seed}" / "best"
