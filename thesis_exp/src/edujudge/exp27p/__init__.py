"""Exp27P shared soft-target Qwen3-Reranker pilot."""

from pathlib import Path


OUTPUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27p_soft_target_reranker_seed42"
)
EXP27O_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27o_361_in_place_pilot_seed42"
)
RUN_ROOT = Path("thesis_exp/runs/exp27p_soft_target_reranker")
ARTIFACT_ROOT = Path("thesis_exp/artifacts/exp27p_soft_target_reranker")
VARIANTS = (
    "v0_original_unweighted",
    "v1_original_label_matched_weight",
    "v2_selective_hard_relabel",
    "v3_selective_soft_audit",
)
PAIRWISE_COMPARISONS = (
    ("v0_original_unweighted", "v1_original_label_matched_weight", "weight_and_exclusion_effect"),
    ("v1_original_label_matched_weight", "v2_selective_hard_relabel", "hard_relabel_effect"),
    ("v2_selective_hard_relabel", "v3_selective_soft_audit", "soft_target_effect"),
    ("v0_original_unweighted", "v3_selective_soft_audit", "end_to_end_effect"),
)
