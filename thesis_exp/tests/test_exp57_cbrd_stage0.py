"""CPU-only regression tests for CBRD Stage 0 target and shuffle contracts."""

from __future__ import annotations

import unittest

from thesis_exp.exp57_cbrd.data_audit import audit_split, load_rows, shuffled_residual_audit, source_closure_audit
from thesis_exp.exp57_cbrd.method import STATE_DOWN, STATE_UP, STATE_ZERO, describe_target


class CbrdStage0Test(unittest.TestCase):
    def test_canonical_relation_examples(self) -> None:
        self.assertEqual(describe_target(4, (0, 0, 1 / 3, 2 / 3, 0))["state"], STATE_DOWN)
        self.assertEqual(describe_target(4, (0, 0, 0, 2 / 3, 1 / 3))["state"], STATE_UP)
        self.assertEqual(describe_target(4, (0, 0, 0, 1, 0))["state"], STATE_ZERO)

    def test_train_dev_support_is_valid(self) -> None:
        for split, expected_rows in (("train", 2654), ("dev", 664)):
            report = audit_split(split)
            self.assertEqual(report["rows"], expected_rows)
            self.assertTrue(report["checks"]["no_invalid_rows"])
            self.assertEqual(report["distinct_target_vectors"], 13)
            self.assertEqual(report["distinct_residual_vectors"], 9)

    def test_shuffle_mapping_reproduces_exp55(self) -> None:
        report = shuffled_residual_audit(load_rows("train"))
        self.assertTrue(report["checks"]["mapping_matches_exp55"])
        self.assertEqual(report["effective_target_changes"], 1490)

    def test_legacy_source_closure(self) -> None:
        report = source_closure_audit()
        self.assertTrue(report["checks"]["all_historical_blobs_match"])
        self.assertTrue(report["checks"]["all_live_shared_dependencies_match"])


if __name__ == "__main__":
    unittest.main()
