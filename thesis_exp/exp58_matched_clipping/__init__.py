"""Exp58 common-scale-matched residual mechanism control."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "thesis_exp" / "configs" / "exp58_matched_clipping"
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp58_matched_clipping"
PROTOCOL_PATH = CONFIG_ROOT / "protocol.json"
SOURCE_LOCK_PATH = CONFIG_ROOT / "source_lock.json"
