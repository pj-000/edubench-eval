"""CPU-only tests for the frozen Exp57 Stage 1 contract."""

from __future__ import annotations

import json
import unittest

from thesis_exp.exp57_cbrd import CONFIG_ROOT
from thesis_exp.exp57_cbrd.data_audit import load_rows
from thesis_exp.exp57_cbrd.losses import VARIANTS
from thesis_exp.exp57_cbrd.metrics import add_boundary_diagnostics


class CbrdStage1Test(unittest.TestCase):
    def test_test_split_is_rejected(self) -> None:
        with self.assertRaises(PermissionError):
            load_rows("test")

    def test_protocol_freezes_six_scientific_variants(self) -> None:
        protocol = json.loads((CONFIG_ROOT / "stage1_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(
            set(protocol["new_variants"]),
            {
                "dual_hard",
                "consensus_only",
                "routed_hmsa",
                "residual_only",
                "sign_flipped",
                "shuffled_residual",
            },
        )
        self.assertEqual(protocol["primary_comparison"], "routed_hmsa minus consensus_only")
        self.assertEqual(protocol["fixed_training"]["gradient_accumulation_steps"], 32)
        self.assertFalse(protocol["fixed_training"]["hyperparameter_search"])
        self.assertEqual(protocol["test_access_count"], 0)

    def test_model_and_loss_route_names_are_closed(self) -> None:
        self.assertEqual(
            set(VARIANTS),
            {
                "ordinary_hmsa",
                "routed_hmsa",
                "dual_hard",
                "consensus_only",
                "residual_only",
                "sign_flipped",
                "shuffled_residual",
                "detached_soft",
            },
        )

    def test_boundary_diagnostic_reports_all_registered_strata(self) -> None:
        rows = [
            {"boundary_state": "down", "minority_neighbor_advantage": -0.5},
            {"boundary_state": "up", "minority_neighbor_advantage": 0.25},
            {"boundary_state": "zero", "minority_neighbor_advantage": None},
        ]
        metrics = add_boundary_diagnostics({}, rows)
        self.assertEqual(metrics["boundary_advantage_down_mean"], -0.5)
        self.assertEqual(metrics["boundary_advantage_up_mean"], 0.25)
        self.assertEqual(metrics["boundary_advantage_pooled_mean"], -0.125)
        self.assertEqual(metrics["boundary_advantage_pooled_n"], 2)


if __name__ == "__main__":
    unittest.main()
