"""Exp64: optimizer-state-aware attribution of multi-rater residual utility."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "thesis_exp" / "configs" / "exp64_optimizer_state_residual"
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp64_optimizer_state_residual"
ARTIFACT_ROOT = REPO_ROOT / "thesis_exp" / "artifacts" / "exp64_optimizer_state_residual"

SEEDS = (72, 73, 74, 75, 76)
STAGE_EPOCHS = (2, 5, 8)
ARMS = (
    "blocked",
    "full_residual",
    "parallel_only",
    "orthogonal_only",
    "sign_flipped_residual",
)

