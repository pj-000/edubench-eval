"""CBRD mechanism audit: consensus--boundary residual decomposition.

This package is intentionally train/dev only.  It starts with a source and
mathematical audit before any new model run is permitted.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "thesis_exp" / "data" / "splits" / "paper_like_triple_seed42"
OUTPUT_ROOT = REPO_ROOT / "thesis_exp" / "outputs" / "exp57_cbrd"
CONFIG_ROOT = REPO_ROOT / "thesis_exp" / "configs" / "exp57_cbrd"

# The original HMSA sources are kept by the repository's main history.  They
# are not ancestors of the current paper-review branch, hence these immutable
# hashes are part of the CBRD source-closure contract.
LEGACY_RESULT_COMMIT = "fa72bd46864a8fc202015fb18507046bf9dd6bcf"
LEGACY_HMSA_SOURCE_COMMIT = "193d64e51ddc7c0ffd8bd4c4b5d4468dafa91162"
LEGACY_FORMAL_PROTOCOL_COMMIT = "f21fca2f383169f1f9969ca22e538194d889ab3b"
LEGACY_EXP49_COMMIT = "d7fdf0348fedb005f5dcdf9bf44ff4f33e598800"
LEGACY_EXP50_COMMIT = "d71169a27429cc7935d445eca41ef5429bb7b8d3"

FORMAL_SEEDS = (42, 43, 44)
SHUFFLE_SEED = 20260730
LABELS = (1, 2, 3, 4, 5)
