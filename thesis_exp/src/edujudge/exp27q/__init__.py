"""Locked Exp27Q V3-Safe16 directional-safety sensitivity experiment."""

from pathlib import Path


VARIANT = "v3_safe16_original_low_anchor"
OUTPUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27q_safe16_multiseed_seed42_44"
)
RUN_ROOT = Path("thesis_exp/runs/exp27q_safe16")
ARTIFACT_ROOT = Path("thesis_exp/artifacts/exp27q_safe16")
EXP27O_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42"
)
EXP27P_RUN_ROOT = Path("thesis_exp/runs/exp27p_soft_target_reranker")
SAFE16_DATASET = OUTPUT_DIR / "private/data/exp27q_v3_safe16_original_low_anchor_train.jsonl"

