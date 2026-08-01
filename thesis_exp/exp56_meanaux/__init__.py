"""Locked constants and paths for the post-hoc MeanAux dev control."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp56_meanaux"
ARTIFACT_ROOT = REPO_ROOT / "thesis_exp" / "artifacts" / "exp56_meanaux"
EXP49_OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp49_cphce"
EXP51_OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp51_hmsa"

VARIANT = "hard_main_mean_aux_lambda1"
AUX_WEIGHT = 1.0
SMOOTH_L1_BETA = 1.0
FORMAL_SEEDS = (42, 43, 44)
SEED42_INITIAL_HEAD_HASH = (
    "d7c922a1956118af437c52189ffc993465277a72698cef38290e47404247f9e2"
)


def run_output_dir(seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / VARIANT / f"seed_{seed}"


def checkpoint_dir(seed: int) -> Path:
    return ARTIFACT_ROOT / VARIANT / f"seed_{seed}" / "best"


def baseline_run_dir(seed: int) -> Path:
    return EXP49_OUTPUT_ROOT / "runs" / "b0_hard_ce" / f"seed_{seed}"


def hmsa_run_dir(seed: int) -> Path:
    return EXP51_OUTPUT_ROOT / "runs" / "hmsa_lambda1" / f"seed_{seed}"
