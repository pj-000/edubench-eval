"""Exp59 residual-geometry ablation under standard global clipping."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "thesis_exp" / "configs" / "exp59_residual_geometry"
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp59_residual_geometry"
PROTOCOL_PATH = CONFIG_ROOT / "protocol.json"
SOURCE_LOCK_PATH = CONFIG_ROOT / "source_lock.json"
