"""CPU-only mapping and geometry tests for the Exp60 draft."""

from __future__ import annotations

import itertools
import json
import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

from thesis_exp.exp60_geometry_matched_shuffle.analyze_confirmation import (
    paired_rows,
    question_cluster_bootstrap,
)
from thesis_exp.exp60_geometry_matched_shuffle import PROTOCOL_PATH
from thesis_exp.exp60_geometry_matched_shuffle.geometry import (
    match_shuffled_orthogonal,
    select_component,
)
from thesis_exp.exp60_geometry_matched_shuffle.mapping import (
    build_maximum_mismatch_mapping,
    theoretical_maximum_changes,
)
from thesis_exp.exp60_geometry_matched_shuffle.preflight import run as run_preflight
from thesis_exp.exp60_geometry_matched_shuffle.train import (
    assert_formal_config_matches_protocol,
    assert_gpu_slot_assignment,
    verify_contract,
)


class Exp60MappingTest(unittest.TestCase):
    def test_frozen_real_mapping_reaches_declared_maximum(self) -> None:
        from thesis_exp.exp57_cbrd.data_audit import model_rows

        _, audit = build_maximum_mismatch_mapping(model_rows("train"))
        self.assertEqual(audit["rows"], 2654)
        self.assertEqual(audit["effective_target_changes"], 2512)
        self.assertAlmostEqual(audit["effective_change_rate"], 2512 / 2654)
        self.assertEqual(audit["self_assignments"], 0)
        self.assertTrue(all(audit["checks"].values()))

    def test_theoretical_bound_matches_bruteforce_small_multisets(self) -> None:
        for values in ([0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 2]):
            expected = max(
                sum(left != right for left, right in zip(values, permutation))
                for permutation in set(itertools.permutations(values))
            )
            self.assertEqual(theoretical_maximum_changes(Counter(values)), expected)

    def test_deterministic_rotation_reaches_maximum(self) -> None:
        rows = []
        scores = (
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 2),
        )
        for index, human_scores in enumerate(scores):
            rows.append(
                {
                    "record_id": f"r{index}",
                    "label_5": 1,
                    "human_1_5": human_scores[0],
                    "human_2_5": human_scores[1],
                    "human_3_5": human_scores[2],
                }
            )
        mapping, audit = build_maximum_mismatch_mapping(rows)
        self.assertEqual(audit["effective_target_changes"], 2)
        self.assertEqual(len(mapping), 4)
        self.assertTrue(all(audit["checks"].values()))


class Exp60GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.common = {
            "backbone.a": np.asarray([3.0, 4.0, 1.0]),
            "hard_head.weight": np.asarray([2.0]),
            "soft_head.weight": np.asarray([-1.0]),
        }
        self.aligned = {
            "backbone.a": np.asarray([2.0, -1.0, 3.0]),
            "hard_head.weight": np.zeros(1),
            "soft_head.weight": np.zeros(1),
        }
        self.shuffled = {
            "backbone.a": np.asarray([-4.0, 2.0, 1.0]),
            "hard_head.weight": np.zeros(1),
            "soft_head.weight": np.zeros(1),
        }

    def test_geometry_match_equalizes_norm_total_norm_and_clip(self) -> None:
        aligned, shuffled, audit = match_shuffled_orthogonal(
            self.common, self.aligned, self.shuffled
        )
        self.assertLessEqual(audit.aligned_normalized_orthogonality_error, 1e-6)
        self.assertLessEqual(audit.shuffled_normalized_orthogonality_error, 1e-6)
        self.assertLessEqual(audit.component_norm_relative_error, 1e-12)
        self.assertLessEqual(audit.preclip_total_norm_relative_error, 1e-12)
        self.assertLessEqual(audit.clip_coefficient_relative_error, 1e-12)
        for name in ("hard_head.weight", "soft_head.weight"):
            self.assertEqual(float(np.linalg.norm(aligned[name])), 0.0)
            self.assertEqual(float(np.linalg.norm(shuffled[name])), 0.0)

    def test_all_arms_preserve_head_updates(self) -> None:
        aligned, shuffled, _ = match_shuffled_orthogonal(
            self.common, self.aligned, self.shuffled
        )
        for variant in (
            "consensus_only",
            "aligned_orthogonal_only",
            "matched_shuffled_orthogonal_only",
        ):
            component = select_component(variant, self.common, aligned, shuffled)
            for name in ("hard_head.weight", "soft_head.weight"):
                self.assertEqual(float(np.linalg.norm(component[name])), 0.0)

    def test_protocol_is_draft_and_forbids_training(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            protocol["status"],
            "EXP60_DRAFT_FOR_INDEPENDENT_REVIEW_NOT_AUTHORIZED_FOR_TRAINING",
        )
        self.assertEqual(protocol["allowed_splits"], ["train", "dev"])
        self.assertEqual(protocol["formal_runs"]["run_count"], 9)
        self.assertIn(
            "formal GPU training before independent review and source lock",
            protocol["prohibited"],
        )
        with self.assertRaisesRegex(RuntimeError, "protocol is not frozen"):
            verify_contract()

    def test_formal_config_assertion_rejects_subsampling_and_disabled_bf16(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        fixed = protocol["fixed_training"]
        base = dict(
            model_name_or_path=protocol["model_name_or_path"],
            num_train_epochs=float(fixed["epochs"]),
            learning_rate=float(fixed["learning_rate"]),
            weight_decay=float(fixed["weight_decay"]),
            warmup_ratio=float(fixed["warmup_ratio"]),
            per_device_train_batch_size=int(fixed["micro_batch_size"]),
            per_device_eval_batch_size=int(fixed["eval_batch_size"]),
            gradient_accumulation_steps=int(fixed["gradient_accumulation_steps"]),
            max_grad_norm=float(fixed["max_grad_norm"]),
            max_length=int(fixed["max_length"]),
            bf16="true",
            fp16=False,
            gradient_checkpointing=True,
            max_train_samples=None,
            max_eval_samples=None,
            num_workers=0,
            local_files_only=True,
            trust_remote_code=False,
            eval_only=False,
            evaluate_test=False,
            seed=47,
        )
        assert_formal_config_matches_protocol(
            SimpleNamespace(**base), "consensus_only", protocol
        )
        for update in ({"max_train_samples": 8}, {"bf16": "false"}):
            with self.assertRaisesRegex(RuntimeError, "formal configuration mismatch"):
                assert_formal_config_matches_protocol(
                    SimpleNamespace(**{**base, **update}), "consensus_only", protocol
                )

    def test_latin_square_assignment_is_enforced(self) -> None:
        assert_gpu_slot_assignment(47, "aligned_orthogonal_only", 1)
        with self.assertRaisesRegex(RuntimeError, "Latin-square mismatch"):
            assert_gpu_slot_assignment(47, "aligned_orthogonal_only", 0)

    def test_cpu_no_update_preflight_passes(self) -> None:
        report = run_preflight()
        self.assertEqual(report["status"], "EXP60_CPU_NO_UPDATE_PREFLIGHT_PASS")
        self.assertTrue(all(report["checks"].values()))


class Exp60AnalysisTest(unittest.TestCase):
    def test_paired_mae_sign_is_treatment_minus_control(self) -> None:
        treatment = [
            {
                "record_id": "r1",
                "question_key": "q1",
                "human_mean_5": 2.0,
                "pred_label_5": 2,
            }
        ]
        control = [
            {
                "record_id": "r1",
                "question_key": "q1",
                "human_mean_5": 2.0,
                "pred_label_5": 3,
            }
        ]
        self.assertEqual(paired_rows(treatment, control)[0]["delta_MAE"], -1.0)

    def test_cluster_bootstrap_is_deterministic_and_conditional_on_three_seeds(self) -> None:
        per_seed = []
        for _ in range(3):
            per_seed.append(
                [
                    {"record_id": "r1", "question_key": "q1", "delta_MAE": -1.0},
                    {"record_id": "r2", "question_key": "q2", "delta_MAE": -0.5},
                ]
            )
        first = question_cluster_bootstrap(per_seed)
        second = question_cluster_bootstrap(per_seed)
        self.assertEqual(first, second)
        self.assertEqual(first["point_estimate"], -0.75)
        self.assertLess(first["ci_95"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
