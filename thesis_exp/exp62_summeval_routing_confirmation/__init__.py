"""Exp62: independent SummEval confirmation of residual-gradient routing."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "thesis_exp/outputs/exp62_summeval_routing_confirmation"
CONFIG_ROOT = EXPERIMENT_ROOT / "configs"
SPLIT_MANIFEST = OUTPUT_ROOT / "stage0/split_manifest.jsonl"

DIMENSIONS = ("coherence", "fluency")
LABELS = (1, 2, 3, 4, 5)
NUM_RATERS = 3
SEEDS = (62, 63, 64, 65, 66)
VARIANTS = (
    "direct_residual_blocked",
    "routed_hmsa",
    "orthogonal_only",
    "parallel_only",
)

