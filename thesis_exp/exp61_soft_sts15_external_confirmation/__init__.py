"""Exp61: external residual-alignment confirmation on Soft-STS-15.

The held-out test split is intentionally absent from all training constants.
It can only be opened later by the separately frozen one-shot analysis entry
point after all nine epoch-10 checkpoints exist.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP61_ROOT = ROOT / "thesis_exp/exp61_soft_sts15_external_confirmation"
OUTPUT_ROOT = ROOT / "thesis_exp/outputs/exp61_soft_sts15_external_confirmation"
STAGE0_PROTOCOL = EXP61_ROOT / "configs/stage0_protocol.json"
FROZEN_PROTOCOL = EXP61_ROOT / "configs/protocol.json"
SPLIT_MANIFEST = OUTPUT_ROOT / "data/split_manifest.jsonl"
MAPPING_PATH = OUTPUT_ROOT / "audit/train_maximum_mismatch_mapping.jsonl"
MAPPING_AUDIT_PATH = OUTPUT_ROOT / "audit/train_maximum_mismatch_mapping_audit.json"

NUM_CLASSES = 6
NUM_PUBLISHED_RATINGS = 5
LABELS = tuple(range(NUM_CLASSES))
TRAINING_SPLITS = ("train", "dev")
VARIANTS = (
    "quantized_mean_only",
    "aligned_orthogonal_only",
    "matched_shuffled_orthogonal_only",
)
SEEDS = (61, 62, 63)
