"""Question-disjoint Exp6 baseline utilities."""

from __future__ import annotations

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP06_QD_NAME = "exp06_question_disjoint_baselines"
EXP06_QD_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP06_QD_NAME
EXP06_QD_DATASETS_DIR = EXP06_QD_OUTPUT_DIR / "datasets"
EXP06_QD_RUNS_DIR = EXP06_QD_OUTPUT_DIR / "runs"
EXP06_QD_TABLES_DIR = EXP06_QD_OUTPUT_DIR / "tables"
EXP06_QD_REPORTS_DIR = EXP06_QD_OUTPUT_DIR / "reports"
EXP06_QD_LOGS_DIR = EXP06_QD_OUTPUT_DIR / "logs"

EXP06_QD_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / EXP06_QD_NAME
EXP06_QD_CHECKPOINTS_DIR = EXP06_QD_ARTIFACTS_DIR / "checkpoints"

QUESTION_SPLIT_DIR = THESIS_DIR / "data" / "splits" / "question_seed42"
A4_QD_DATASET_DIR = EXP06_QD_DATASETS_DIR / "A4_question_answer_metric_rubric_metadata_question_seed42"

QD_B0_RUN_ID = "QD-B0_human_only_ordinary_ordinal"
QD_B1_RUN_ID = "QD-B1_human_only_L1_weighted_ordinal"

EXPECTED_QD_SPLIT_ROWS = {"train": 3326, "dev": 1107, "test": 1103}


def ensure_exp06_qd_dirs() -> None:
    for path in [
        EXP06_QD_OUTPUT_DIR,
        EXP06_QD_DATASETS_DIR,
        EXP06_QD_RUNS_DIR,
        EXP06_QD_TABLES_DIR,
        EXP06_QD_REPORTS_DIR,
        EXP06_QD_LOGS_DIR,
        EXP06_QD_CHECKPOINTS_DIR,
        A4_QD_DATASET_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
