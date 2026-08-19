"""Exp54: rubric-aligned, reliability-gated rationale SFT."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar0_alignment"
