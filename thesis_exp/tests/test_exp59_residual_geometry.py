"""CPU geometry and protocol tests for Exp59."""

from __future__ import annotations

import json
import unittest

import numpy as np

from thesis_exp.exp59_residual_geometry import PROTOCOL_PATH
from thesis_exp.exp59_residual_geometry.geometry import (
    compose_preclip_gradient,
    decompose_residual,
)
from thesis_exp.exp59_residual_geometry.endpoint_parity import run as endpoint_parity


class ResidualGeometryTest(unittest.TestCase):
    def test_numerical_amendment_preserves_protocol_and_threshold(self) -> None:
        amendment_path = (
            PROTOCOL_PATH.parent / "implementation_amendment_v2.json"
        )
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        self.assertEqual(
            amendment["protocol_sha256_unchanged"],
            "42a96ca25f6a40619da011c34905387e0c67da0567fee6a27744724fba257e85",
        )
        self.assertEqual(amendment["trigger"]["frozen_limit"], 1e-6)
        self.assertFalse(amendment["performance_used_to_authorize_change"])
        self.assertEqual(amendment["test_access_count"], 0)

    def test_protocol_freezes_primary_and_no_search(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            protocol["status"],
            "EXP59_RESIDUAL_GEOMETRY_PROTOCOL_FROZEN_BEFORE_REAL_MODEL_RESULTS",
        )
        self.assertEqual(protocol["allowed_splits"], ["train", "dev"])
        self.assertEqual(protocol["test_access_count"], 0)
        self.assertEqual(protocol["new_runs"]["run_count"], 10)
        self.assertEqual(
            protocol["primary_comparison"]["delta"],
            "orthogonal_only minus consensus_only",
        )
        self.assertFalse(protocol["fixed_training"]["hyperparameter_search"])

    def test_parallel_plus_orthogonal_reconstructs_residual(self) -> None:
        common = {
            "backbone.a": np.asarray([3.0, 4.0]),
            "hard_head.weight": np.asarray([2.0]),
            "soft_head.weight": np.asarray([-1.0]),
        }
        residual = {
            "backbone.a": np.asarray([2.0, -1.0]),
            "hard_head.weight": np.asarray([0.0]),
            "soft_head.weight": np.asarray([0.0]),
        }
        parallel, orthogonal, audit = decompose_residual(common, residual)
        self.assertLessEqual(audit.reconstruction_relative_error, 1e-12)
        self.assertLessEqual(audit.normalized_orthogonality_error, 1e-6)
        for name in residual:
            self.assertTrue(np.allclose(parallel[name] + orthogonal[name], residual[name]))

    def test_components_have_zero_head_support(self) -> None:
        common = {
            "backbone.a": np.asarray([1.0, 0.0]),
            "hard_head.weight": np.asarray([4.0]),
            "soft_head.weight": np.asarray([5.0]),
        }
        residual = {
            "backbone.a": np.asarray([1.0, 2.0]),
            "hard_head.weight": np.asarray([9.0]),
            "soft_head.weight": np.asarray([8.0]),
        }
        parallel, orthogonal, _ = decompose_residual(common, residual)
        for name in ("hard_head.weight", "soft_head.weight"):
            self.assertEqual(float(np.linalg.norm(parallel[name])), 0.0)
            self.assertEqual(float(np.linalg.norm(orthogonal[name])), 0.0)

    def test_zero_common_assigns_all_residual_to_orthogonal(self) -> None:
        common = {"backbone.a": np.zeros(2), "hard_head.weight": np.ones(1)}
        residual = {"backbone.a": np.asarray([2.0, -3.0]), "hard_head.weight": np.zeros(1)}
        parallel, orthogonal, audit = decompose_residual(common, residual)
        self.assertEqual(float(np.linalg.norm(parallel["backbone.a"])), 0.0)
        self.assertTrue(np.array_equal(orthogonal["backbone.a"], residual["backbone.a"]))
        self.assertEqual(audit.projection_coefficient, 0.0)

    def test_full_route_reconstruction(self) -> None:
        common = {"backbone.a": np.asarray([3.0, 4.0]), "hard_head.weight": np.asarray([2.0])}
        residual = {"backbone.a": np.asarray([-1.0, 2.0]), "hard_head.weight": np.zeros(1)}
        parallel, orthogonal, _ = decompose_residual(common, residual)
        reconstructed = compose_preclip_gradient(
            compose_preclip_gradient(common, parallel), orthogonal
        )
        full = compose_preclip_gradient(common, residual)
        for name in full:
            self.assertTrue(np.allclose(reconstructed[name], full[name]))

    def test_existing_endpoint_parity(self) -> None:
        report = endpoint_parity()
        self.assertEqual(report["status"], "EXP59_ENDPOINT_PARITY_PASS")
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["test_access_count"], 0)


if __name__ == "__main__":
    unittest.main()
