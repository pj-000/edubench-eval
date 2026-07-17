"""Locked paths and constants for Exp51 hard-main/soft-auxiliary MTL."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp51_hmsa"
ARTIFACT_ROOT = REPO_ROOT / "thesis_exp" / "artifacts" / "exp51_hmsa"
EXP49_OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp49_cphce"

VARIANT = "hmsa_lambda1"
AUX_WEIGHT = 1.0
FORMAL_SEEDS = (42, 43, 44)
BASELINE_RESULT_COMMIT = "9ad6190277950439a3a5b9b7188c14b3943b7433"
EXP50_RESULT_COMMIT = "d71169a"
SEED42_INITIAL_HEAD_HASH = "d7c922a1956118af437c52189ffc993465277a72698cef38290e47404247f9e2"


def run_output_dir(seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / VARIANT / f"seed_{seed}"


def checkpoint_dir(seed: int) -> Path:
    return ARTIFACT_ROOT / VARIANT / f"seed_{seed}" / "best"


def baseline_run_dir(seed: int) -> Path:
    return EXP49_OUTPUT_ROOT / "runs" / "b0_hard_ce" / f"seed_{seed}"
