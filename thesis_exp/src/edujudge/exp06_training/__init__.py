"""Exp6 synthetic-augmented training utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP06_SYNTH_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp06_synthetic_low_score"
EXP06_TRAINING_DATASETS_DIR = EXP06_SYNTH_OUTPUT_DIR / "training_datasets"
EXP06_TRAINING_RUNS_DIR = EXP06_SYNTH_OUTPUT_DIR / "training_runs"
EXP06_TRAINING_TABLES_DIR = EXP06_TRAINING_RUNS_DIR / "tables"
EXP06_TRAINING_LOGS_DIR = EXP06_TRAINING_RUNS_DIR / "logs"
EXP06_TRAINING_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / "exp06_synthetic_low_score" / "training_runs"

QD_S1_RUN_ID = "QD-S1_human_plus_synthetic_ordinal"
QD_S1_DATASET_DIR = EXP06_TRAINING_DATASETS_DIR / QD_S1_RUN_ID
QD_S1_RUN_DIR = EXP06_TRAINING_RUNS_DIR / QD_S1_RUN_ID
QD_S1_CHECKPOINT_DIR = EXP06_TRAINING_ARTIFACTS_DIR / "checkpoints" / QD_S1_RUN_ID

QUESTION_SPLIT_DIR = THESIS_DIR / "data" / "splits" / "question_seed42"
QD_BASELINE_OUTPUT_DIR = THESIS_DIR / "outputs" / "exp06_question_disjoint_baselines"
QD_BASELINE_RUNS_DIR = QD_BASELINE_OUTPUT_DIR / "runs"
QD_BASELINE_TABLES_DIR = QD_BASELINE_OUTPUT_DIR / "tables"


def ensure_exp06_training_dirs() -> None:
    for path in [
        EXP06_TRAINING_RUNS_DIR,
        EXP06_TRAINING_TABLES_DIR,
        EXP06_TRAINING_LOGS_DIR,
        EXP06_TRAINING_ARTIFACTS_DIR,
        QD_S1_RUN_DIR,
        QD_S1_CHECKPOINT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
