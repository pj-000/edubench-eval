"""Exp63: same-state, same-norm, same-clipping counterfactual updates."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "thesis_exp" / "configs" / "exp63_same_state_counterfactual"
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp63_same_state_counterfactual"
ARTIFACT_ROOT = REPO_ROOT / "thesis_exp" / "artifacts" / "exp63_same_state_counterfactual"
PROTOCOL_PATH = CONFIG_ROOT / "protocol.json"
SOURCE_LOCK_PATH = CONFIG_ROOT / "source_lock.json"

SEEDS = (67, 68, 69, 70, 71)
STAGE_EPOCHS = (2, 5, 8)
ARMS = ("blocked", "full_residual", "parallel_only", "orthogonal_only")

