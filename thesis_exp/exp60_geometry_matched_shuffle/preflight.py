"""CPU-only, no-update preflight for the Exp60 draft protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp60_geometry_matched_shuffle import (
    MAPPING_AUDIT_PATH,
    OUTPUT_ROOT,
    PROTOCOL_PATH,
)
from thesis_exp.exp60_geometry_matched_shuffle.geometry import (
    match_shuffled_orthogonal,
    select_component,
)


OUTPUT_PATH = OUTPUT_ROOT / "audit" / "cpu_no_update_preflight.json"


def _random_vector_dict(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "backbone.lower": rng.normal(size=19).astype(np.float32),
        "backbone.upper": rng.normal(size=11).astype(np.float32),
        "hard_head.weight": rng.normal(size=5).astype(np.float32),
        "soft_head.weight": rng.normal(size=5).astype(np.float32),
    }


def _backbone_only(values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value if name.startswith("backbone.") else np.zeros_like(value)
        for name, value in values.items()
    }


def run() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING_AUDIT_PATH.read_text(encoding="utf-8"))
    rng = np.random.default_rng(60)
    audits: list[dict[str, Any]] = []
    head_support_pass = True
    for case in range(64):
        common = _random_vector_dict(rng)
        aligned_residual = _backbone_only(_random_vector_dict(rng))
        shuffled_residual = _backbone_only(_random_vector_dict(rng))
        aligned, shuffled, audit = match_shuffled_orthogonal(
            common,
            aligned_residual,
            shuffled_residual,
            max_norm=1.0,
        )
        for variant in (
            "consensus_only",
            "aligned_orthogonal_only",
            "matched_shuffled_orthogonal_only",
        ):
            selected = select_component(variant, common, aligned, shuffled)
            for name in ("hard_head.weight", "soft_head.weight"):
                head_support_pass &= float(np.linalg.norm(selected[name])) <= 1e-12
        audits.append({"case": case, **audit.as_dict()})

    zero_guard_pass = False
    try:
        match_shuffled_orthogonal(
            {
                "backbone.a": np.asarray([1.0, 0.0]),
                "hard_head.weight": np.asarray([1.0]),
            },
            {
                "backbone.a": np.asarray([0.0, 1.0]),
                "hard_head.weight": np.zeros(1),
            },
            {
                "backbone.a": np.asarray([2.0, 0.0]),
                "hard_head.weight": np.zeros(1),
            },
        )
    except RuntimeError as error:
        zero_guard_pass = str(error) == (
            "EXP60_STOP_SHUFFLED_ORTHOGONAL_ZERO_WHILE_ALIGNED_NONZERO"
        )

    maxima = {
        "aligned_normalized_orthogonality_error": max(
            row["aligned_normalized_orthogonality_error"] for row in audits
        ),
        "shuffled_normalized_orthogonality_error": max(
            row["shuffled_normalized_orthogonality_error"] for row in audits
        ),
        "component_norm_relative_error": max(
            row["component_norm_relative_error"] for row in audits
        ),
        "preclip_total_norm_relative_error": max(
            row["preclip_total_norm_relative_error"] for row in audits
        ),
        "clip_coefficient_relative_error": max(
            row["clip_coefficient_relative_error"] for row in audits
        ),
    }
    checks = {
        "protocol_remains_draft_and_training_forbidden": protocol["status"]
        == "EXP60_DRAFT_FOR_INDEPENDENT_REVIEW_NOT_AUTHORIZED_FOR_TRAINING",
        "mapping_audit_pass": mapping["status"]
        == "EXP60_MAXIMUM_MISMATCH_MAPPING_PASS",
        "mapping_maximum_mismatch_every_label": mapping["checks"][
            "maximum_mismatch_achieved_in_every_label"
        ],
        "aligned_orthogonality_at_most_1e_6": maxima[
            "aligned_normalized_orthogonality_error"
        ]
        <= 1e-6,
        "shuffled_orthogonality_at_most_1e_6": maxima[
            "shuffled_normalized_orthogonality_error"
        ]
        <= 1e-6,
        "component_norm_match_at_most_1e_4": maxima[
            "component_norm_relative_error"
        ]
        <= 1e-4,
        "preclip_total_norm_match_at_most_1e_4": maxima[
            "preclip_total_norm_relative_error"
        ]
        <= 1e-4,
        "clip_coefficient_match_at_most_1e_4": maxima[
            "clip_coefficient_relative_error"
        ]
        <= 1e-4,
        "residual_components_have_zero_head_support": head_support_pass,
        "zero_shuffled_nonzero_aligned_forces_stop": zero_guard_pass,
        "no_optimizer_step": True,
        "no_model_loaded": True,
        "no_test_access": True,
    }
    report = {
        "status": "EXP60_CPU_NO_UPDATE_PREFLIGHT_PASS"
        if all(checks.values())
        else "EXP60_CPU_NO_UPDATE_PREFLIGHT_FAIL",
        "cases": len(audits),
        "rng_seed": 60,
        "maxima": maxima,
        "mapping_sha256": mapping["mapping_sha256"],
        "checks": checks,
        "scope": "generic FP32/FP64 geometry only; real Qwen BF16 preflight remains required before formal training",
        "allowed_splits": ["train"],
        "test_access_count": 0,
    }
    write_json(OUTPUT_PATH, report)
    return report


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
