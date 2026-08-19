"""Locked paths and constants for Exp50 CAHS-0.5."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp50_cahs"
ARTIFACT_ROOT = REPO_ROOT / "thesis_exp" / "artifacts" / "exp50_cahs"
CONFIG_ROOT = REPO_ROOT / "thesis_exp" / "configs" / "exp50_cahs"
EXP49_OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp49_cphce"
EXP49_ARTIFACT_ROOT = REPO_ROOT / "thesis_exp" / "artifacts" / "exp49_cphce"

VARIANT = "c1_cahs_0p5"
ALPHA = 0.5
FORMAL_SEEDS = (42, 43, 44)
LOCKED_BASELINE_COMMIT = "9ad6190277950439a3a5b9b7188c14b3943b7433"


def run_output_dir(seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / VARIANT / f"seed_{seed}"


def checkpoint_dir(seed: int) -> Path:
    return ARTIFACT_ROOT / VARIANT / f"seed_{seed}" / "best"


def baseline_run_dir(seed: int) -> Path:
    return EXP49_OUTPUT_ROOT / "runs" / "b0_hard_ce" / f"seed_{seed}"


def baseline_checkpoint_dir(seed: int) -> Path:
    return EXP49_ARTIFACT_ROOT / "b0_hard_ce" / f"seed_{seed}" / "best"
