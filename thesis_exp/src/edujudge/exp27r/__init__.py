"""Exp27R frozen one-shot final test campaign."""

from pathlib import Path


VARIANTS = (
    "v0_original_unweighted",
    "v1_original_label_matched_weight",
    "v2_selective_hard_relabel",
    "v3_selective_soft_audit",
    "v3_safe16_original_low_anchor",
)
SEEDS = (42, 43, 44)
COMPARISONS = (
    ("v0_original_unweighted", "v1_original_label_matched_weight", "weight_exclusion_effect"),
    ("v1_original_label_matched_weight", "v2_selective_hard_relabel", "hard_relabel_effect"),
    ("v2_selective_hard_relabel", "v3_selective_soft_audit", "soft_target_effect"),
    ("v3_selective_soft_audit", "v3_safe16_original_low_anchor", "directional_safety_effect"),
    ("v0_original_unweighted", "v3_safe16_original_low_anchor", "descriptive_end_to_end"),
)
OUTPUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27r_final_test_campaign_seed42_44"
)
EXP27P_RUN_ROOT = Path("thesis_exp/runs/exp27p_soft_target_reranker")
EXP27Q_RUN_ROOT = Path("thesis_exp/runs/exp27q_safe16")


def run_root(variant: str) -> Path:
    return EXP27Q_RUN_ROOT if variant == "v3_safe16_original_low_anchor" else EXP27P_RUN_ROOT

