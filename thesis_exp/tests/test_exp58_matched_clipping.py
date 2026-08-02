"""CPU tests for the frozen Exp58 matched-clipping geometry."""

from __future__ import annotations

import json
import unittest

import numpy as np

from thesis_exp.exp58_matched_clipping import PROTOCOL_PATH
from thesis_exp.exp58_matched_clipping.matched_update import (
    clip_coefficient,
    compose_matched_gradients,
    largest_safe_beta,
    summarize_components,
)


class MatchedClippingTest(unittest.TestCase):
    def test_protocol_closes_search_and_test_access(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            protocol["status"],
            "EXP58_MATCHED_CLIPPING_PROTOCOL_V2_FROZEN_BEFORE_FORMAL_TRAINING",
        )
        self.assertEqual(protocol["allowed_splits"], ["train", "dev"])
        self.assertEqual(protocol["test_access_count"], 0)
        self.assertEqual(protocol["new_training_runs"]["run_count"], 5)
        self.assertFalse(protocol["fixed_training"]["hyperparameter_search"])
        gates = protocol["implementation_gates_before_training"]
        self.assertEqual(
            gates["bf16_runtime_gate"][
                "standard_routed_post_clip_reconstruction_relative_error_at_most"
            ],
            1.5 * (2.0 ** -7),
        )
        self.assertEqual(
            gates["fp32_identity_gate"][
                "paired_trainer_vs_independent_construction_relative_error_at_most"
            ],
            1e-4,
        )

    def test_clip_coefficient_matches_frozen_rule(self) -> None:
        self.assertEqual(clip_coefficient(0.5), 1.0)
        self.assertAlmostEqual(clip_coefficient(2.0), 1.0 / 2.000001)

    def test_preferred_beta_is_kept_when_safe(self) -> None:
        beta, capped = largest_safe_beta(
            common_sq=4.0,
            residual_sq=1.0,
            common_residual_dot=-1.5,
            alpha_common=0.5,
            alpha_routed=0.6,
        )
        self.assertEqual(beta, 0.6)
        self.assertFalse(capped)

    def test_beta_is_capped_at_positive_quadratic_root(self) -> None:
        beta, capped = largest_safe_beta(
            common_sq=4.0,
            residual_sq=4.0,
            common_residual_dot=0.0,
            alpha_common=0.5,
            alpha_routed=0.5,
        )
        self.assertTrue(capped)
        self.assertAlmostEqual(beta, 0.0, places=7)

    def test_composed_update_matches_common_scale_and_safe_norm(self) -> None:
        common = {"backbone": np.asarray([3.0, 4.0]), "head": np.asarray([2.0])}
        residual = {"backbone": np.asarray([-1.0, 0.0]), "head": np.asarray([0.0])}
        scalars = summarize_components(common, residual)
        matched = compose_matched_gradients(common, residual, scalars)
        norm = np.linalg.norm(np.concatenate([value.ravel() for value in matched.values()]))
        self.assertLessEqual(norm, 1.000001)
        self.assertTrue(np.array_equal(residual["head"], np.zeros(1)))

    def test_zero_residual_reduces_to_consensus_clipping(self) -> None:
        common = {"p": np.asarray([3.0, 4.0])}
        residual = {"p": np.zeros(2)}
        scalars = summarize_components(common, residual)
        matched = compose_matched_gradients(common, residual, scalars)
        self.assertEqual(scalars.beta, 0.0)
        self.assertFalse(scalars.beta_cap_active)
        self.assertTrue(np.allclose(matched["p"], common["p"] * scalars.alpha_common))

    def test_protocol_freezes_the_reviewed_scientific_gate(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        gate = protocol["scientific_go_no_go"]
        self.assertEqual(gate["favorable_mae_seeds_required"], 4)
        self.assertEqual(gate["mean_delta_mae_at_most"], -0.005)
        self.assertEqual(gate["mean_delta_exact_at_least"], -0.003)
        self.assertEqual(gate["mean_delta_kendall_at_least"], -0.005)
        self.assertEqual(gate["five_seed_question_cluster_bootstrap_ci_upper_below"], 0.0)


if __name__ == "__main__":
    unittest.main()
