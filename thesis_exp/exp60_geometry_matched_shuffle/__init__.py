"""Exp60 sample-alignment specificity audit for residual gradient geometry."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "thesis_exp" / "configs" / "exp60_geometry_matched_shuffle"
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp60_geometry_matched_shuffle"
PROTOCOL_PATH = CONFIG_ROOT / "protocol_draft.json"
SOURCE_LOCK_PATH = CONFIG_ROOT / "source_lock.json"
MAPPING_PATH = OUTPUT_ROOT / "audit" / "max_mismatch_mapping.jsonl"
MAPPING_AUDIT_PATH = OUTPUT_ROOT / "audit" / "max_mismatch_mapping_audit.json"
REAL_PREFLIGHT_DECISION_PATH = OUTPUT_ROOT / "decision" / "real_model_preflight_decision.json"
