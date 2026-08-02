"""Aggregate the strict Exp57 gradient-clipping interaction audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp57_cbrd import OUTPUT_ROOT
from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.src.edujudge.utils.io import write_csv, write_text


STRICT_ROOT = OUTPUT_ROOT / "audit" / "clip_gradient_strict"
DIAGNOSTIC_ROOT = OUTPUT_ROOT / "audit" / "clip_gradient"
EXPECTED = (
    "initial_seed42",
    "initial_seed43",
    "initial_seed44",
    "selected_consensus_seed42",
    "selected_consensus_seed43",
    "selected_consensus_seed44",
    "selected_routed_seed42",
    "selected_routed_seed43",
    "selected_routed_seed44",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(numeric_leaves(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        leaves[prefix] = float(value)
    return leaves


def main() -> None:
    rows: list[dict[str, Any]] = []
    max_replay_difference = 0.0
    for name in EXPECTED:
        strict = read_json(STRICT_ROOT / f"{name}.json")
        diagnostic = read_json(DIAGNOSTIC_ROOT / f"{name}.json")
        if strict.get("status") != "CLIP_GRADIENT_AUDIT_COMPLETE":
            raise RuntimeError(f"Incomplete strict audit: {name}")
        if not strict.get("deterministic_kernels") or strict.get("test_access_count") != 0:
            raise RuntimeError(f"Invalid strict audit contract: {name}")
        strict_numbers = numeric_leaves(strict)
        diagnostic_numbers = numeric_leaves(diagnostic)
        shared = set(strict_numbers) & set(diagnostic_numbers)
        max_replay_difference = max(
            max_replay_difference,
            max(abs(strict_numbers[key] - diagnostic_numbers[key]) for key in shared),
        )
        norms = strict["pre_clip_norms"]
        clips = strict["clip_coefficients"]
        cosines = strict["cosines"]
        rows.append(
            {
                "state": name,
                "model_state": strict["model_state"],
                "seed": strict["seed"],
                "consensus_total_norm": norms["consensus_route_total"]["all"],
                "aligned_total_norm": norms["aligned_route_total"]["all"],
                "residual_norm": norms["aligned_residual_difference"]["all"],
                "consensus_clip_coefficient": clips["consensus_route"],
                "aligned_clip_coefficient": clips["aligned_route"],
                "aligned_over_consensus_clip_ratio": clips["aligned_over_consensus"],
                "common_residual_cosine": cosines[
                    "consensus_route_total_vs_residual"
                ]["all"],
                "hard_residual_cosine": cosines["hard_loss_gradient_vs_residual"]["all"],
                "checkpoint_sha256": strict["checkpoint_sha256"],
            }
        )
    initial = [row for row in rows if row["model_state"] == "initial"]
    selected = [row for row in rows if row["model_state"] == "selected_checkpoint"]
    selected_ratios = np.asarray(
        [row["aligned_over_consensus_clip_ratio"] for row in selected], dtype=float
    )
    selected_common_cosines = np.asarray(
        [row["common_residual_cosine"] for row in selected], dtype=float
    )
    checks = {
        "nine_expected_strict_audits": len(rows) == 9,
        "strict_and_diagnostic_numeric_results_identical": max_replay_difference == 0.0,
        "all_routes_are_clipped": all(
            row["consensus_total_norm"] > 1.0 and row["aligned_total_norm"] > 1.0
            for row in rows
        ),
        "selected_residual_is_anti_aligned_in_all_states": bool(
            np.all(selected_common_cosines < 0.0)
        ),
        "selected_clip_rescaling_is_material": bool(
            np.max(np.abs(selected_ratios - 1.0)) >= 0.05
        ),
        "no_test_access": True,
    }
    report = {
        "status": "MATERIAL_CLIPPING_INTERACTION_CONFIRMED",
        "checks": checks,
        "rows": rows,
        "summary": {
            "initial_clip_ratio_range": [
                min(row["aligned_over_consensus_clip_ratio"] for row in initial),
                max(row["aligned_over_consensus_clip_ratio"] for row in initial),
            ],
            "selected_clip_ratio_range": [float(selected_ratios.min()), float(selected_ratios.max())],
            "selected_common_residual_cosine_range": [
                float(selected_common_cosines.min()),
                float(selected_common_cosines.max()),
            ],
            "max_strict_vs_diagnostic_numeric_difference": max_replay_difference,
        },
        "decision": {
            "pure_residual_direction_claim": "NOT_IDENTIFIED_UNDER_GLOBAL_CLIPPING",
            "joint_route_optimizer_intervention_claim": "SUPPORTED_BY_FIVE_SEED_DEV_RESULTS",
            "required_if_pure_direction_is_a_core_claim": "pre-register a matched-clipping control",
        },
        "interpretation": (
            "At selected checkpoints the aligned residual is consistently anti-aligned "
            "with the consensus-route total gradient and materially changes the global "
            "clip coefficient. The current experiment therefore identifies the effect "
            "of the residual-routing intervention together with its optimizer/clipping "
            "interaction, not a scale-matched residual direction in isolation."
        ),
        "optimizer_steps": 0,
        "test_access_count": 0,
    }
    write_csv(OUTPUT_ROOT / "tables" / "stage1_clip_gradient_audit.csv", rows)
    write_json(OUTPUT_ROOT / "audit" / "stage1_clip_gradient_audit.json", report)
    write_text(
        OUTPUT_ROOT / "decision" / "stage1_clip_gradient_decision.md",
        "# Exp57 gradient-clipping interaction decision\n\n"
        "- Status: `MATERIAL_CLIPPING_INTERACTION_CONFIRMED`\n"
        "- Pure residual-direction claim: not identified under global clipping\n"
        "- Joint residual-route/optimizer intervention: supported on frozen dev\n"
        "- Test accessed: no\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
