"""Exp10 QD-PR2 module ablation utilities."""

from __future__ import annotations

from pathlib import Path

from thesis_exp.src.edujudge.utils.io import THESIS_DIR


EXP10_NAME = "exp10_qdpr2_module_ablation"
EXP10_OUTPUT_DIR = THESIS_DIR / "outputs" / EXP10_NAME
EXP10_TABLES_DIR = EXP10_OUTPUT_DIR / "tables"
EXP10_REPORTS_DIR = EXP10_OUTPUT_DIR / "reports"
EXP10_RUNS_DIR = EXP10_OUTPUT_DIR / "runs"
EXP10_LOGS_DIR = EXP10_OUTPUT_DIR / "logs"
EXP10_CONFIG_SNAPSHOT_DIR = EXP10_OUTPUT_DIR / "configs"
EXP10_ARTIFACTS_DIR = THESIS_DIR / "artifacts" / EXP10_NAME
EXP10_CHECKPOINTS_DIR = EXP10_ARTIFACTS_DIR / "checkpoints"
EXP10_CONFIG_DIR = THESIS_DIR / "configs" / EXP10_NAME

ABLATION_ORDER = [
    "full_qdpr2",
    "no_pair",
    "no_pair_same_pair_batches",
    "no_anchor",
    "no_mono",
    "point_only",
    "no_point_diagnostic",
]

PRIMARY_ABLATIONS = [
    "full_qdpr2",
    "no_pair_same_pair_batches",
    "no_anchor",
    "no_mono",
    "point_only",
]

ABLATION_LAMBDAS = {
    "full_qdpr2": {"lambda_point": 1.0, "lambda_pair": 0.05, "lambda_anchor": 0.5, "lambda_mono": 0.1},
    "no_pair": {"lambda_point": 1.0, "lambda_pair": 0.0, "lambda_anchor": 0.5, "lambda_mono": 0.1},
    "no_pair_same_pair_batches": {"lambda_point": 1.0, "lambda_pair": 0.0, "lambda_anchor": 0.5, "lambda_mono": 0.1},
    "no_anchor": {"lambda_point": 1.0, "lambda_pair": 0.05, "lambda_anchor": 0.0, "lambda_mono": 0.1},
    "no_mono": {"lambda_point": 1.0, "lambda_pair": 0.05, "lambda_anchor": 0.5, "lambda_mono": 0.0},
    "point_only": {"lambda_point": 1.0, "lambda_pair": 0.0, "lambda_anchor": 0.0, "lambda_mono": 0.0},
    "no_point_diagnostic": {"lambda_point": 0.0, "lambda_pair": 0.05, "lambda_anchor": 0.5, "lambda_mono": 0.1},
}

DISPLAY_NAMES = {
    "full_qdpr2": "full_qdpr2",
    "no_pair": "no_pair_pointwise_loader",
    "no_pair_same_pair_batches": "no_pair_same_pair_batches",
    "no_anchor": "no_anchor",
    "no_mono": "no_mono",
    "point_only": "point_only",
    "no_point_diagnostic": "no_point_diagnostic",
}

FORCE_PAIR_TRAINING = {
    "full_qdpr2": False,
    "no_pair": False,
    "no_pair_same_pair_batches": True,
    "no_anchor": False,
    "no_mono": False,
    "point_only": False,
    "no_point_diagnostic": False,
}

DATALOADER_MODES = {
    name: "pair" if lambdas["lambda_pair"] != 0.0 or FORCE_PAIR_TRAINING[name] else "pointwise"
    for name, lambdas in ABLATION_LAMBDAS.items()
}

STRICT_MODULE_ABLATION = {
    "full_qdpr2": False,
    "no_pair": False,
    "no_pair_same_pair_batches": True,
    "no_anchor": True,
    "no_mono": True,
    "point_only": False,
    "no_point_diagnostic": False,
}

REMOVED_MODULES = {
    "full_qdpr2": "none",
    "no_pair": "L_pair",
    "no_pair_same_pair_batches": "L_pair",
    "no_anchor": "L_anchor",
    "no_mono": "L_mono",
    "point_only": "L_pair+L_anchor+L_mono",
    "no_point_diagnostic": "L_point",
}

ABLATION_INTERPRETATIONS = {
    "full_qdpr2": "Reference QD-PR2 objective with all planned loss terms active.",
    "no_pair": "Auxiliary diagnostic: removes L_pair and switches to the pointwise loader.",
    "no_pair_same_pair_batches": "Primary L_pair evidence: removes the L_pair gradient contribution under identical pair-batch exposure.",
    "no_anchor": "Strict module ablation for L_anchor while retaining pair-batch exposure.",
    "no_mono": "Strict module ablation for L_mono while retaining pair-batch exposure.",
    "point_only": "Reduced pointwise-only baseline with pair, anchor, and monotonic terms disabled.",
    "no_point_diagnostic": "Diagnostic run without direct pointwise supervision.",
}


def active_losses_for_lambdas(lambdas: dict[str, float]) -> str:
    names = []
    if float(lambdas["lambda_point"]) != 0.0:
        names.append("L_point")
    if float(lambdas["lambda_pair"]) != 0.0:
        names.append("L_pair")
    if float(lambdas["lambda_anchor"]) != 0.0:
        names.append("L_anchor")
    if float(lambdas["lambda_mono"]) != 0.0:
        names.append("L_mono")
    return " + ".join(names) if names else "none"


def use_pair_training_for(ablation_name: str) -> bool:
    lambdas = ABLATION_LAMBDAS[ablation_name]
    return bool(float(lambdas["lambda_pair"]) != 0.0 or FORCE_PAIR_TRAINING[ablation_name])


def ablation_metadata(ablation_name: str) -> dict[str, object]:
    lambdas = ABLATION_LAMBDAS[ablation_name]
    return {
        "ablation_name": ablation_name,
        "display_name": DISPLAY_NAMES[ablation_name],
        "dataloader_mode": DATALOADER_MODES[ablation_name],
        "force_pair_training": FORCE_PAIR_TRAINING[ablation_name],
        "use_pair_training": use_pair_training_for(ablation_name),
        "strict_module_ablation": STRICT_MODULE_ABLATION[ablation_name],
        "removed_module": REMOVED_MODULES[ablation_name],
        "active_losses": active_losses_for_lambdas(lambdas),
        "interpretation": ABLATION_INTERPRETATIONS[ablation_name],
        **lambdas,
    }


def exp10_run_id(ablation_name: str) -> str:
    return f"EXP10_{ablation_name}"


def exp10_run_dir(ablation_name: str) -> Path:
    return EXP10_RUNS_DIR / ablation_name


def exp10_checkpoint_dir(ablation_name: str) -> Path:
    return EXP10_CHECKPOINTS_DIR / ablation_name


def ensure_exp10_dirs() -> None:
    for path in [
        EXP10_OUTPUT_DIR,
        EXP10_TABLES_DIR,
        EXP10_REPORTS_DIR,
        EXP10_RUNS_DIR,
        EXP10_LOGS_DIR,
        EXP10_CONFIG_SNAPSHOT_DIR,
        EXP10_ARTIFACTS_DIR,
        EXP10_CHECKPOINTS_DIR,
        EXP10_CONFIG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for name in ABLATION_ORDER:
        (exp10_run_dir(name) / "tables").mkdir(parents=True, exist_ok=True)
        (exp10_run_dir(name) / "logs").mkdir(parents=True, exist_ok=True)
        (exp10_run_dir(name) / "predictions").mkdir(parents=True, exist_ok=True)
        (exp10_run_dir(name) / "arrays").mkdir(parents=True, exist_ok=True)
