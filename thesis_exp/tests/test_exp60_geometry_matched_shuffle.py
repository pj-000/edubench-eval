"""CPU-only mapping and geometry tests for the Exp60 draft."""

from __future__ import annotations

import itertools
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from thesis_exp.exp60_geometry_matched_shuffle.analyze_confirmation import (
    paired_rows,
    question_cluster_bootstrap,
)
from thesis_exp.exp60_geometry_matched_shuffle import (
    MAPPING_AUDIT_PATH,
    MAPPING_PATH,
    PREFLIGHT_SOURCE_LOCK_PATH,
    PROTOCOL_PATH,
)
from thesis_exp.exp60_geometry_matched_shuffle.contract import (
    FORMAL_MANDATORY_FILES,
    FORMAL_SOURCE_LOCK_SCHEMA_VERSION,
    manifest_sha256,
    normalized_scientific_protocol_sha256,
    stable_gpu_identity,
    validate_formal_source_lock,
    verify_preflight_source_lock,
)
from thesis_exp.exp60_geometry_matched_shuffle.geometry import (
    match_shuffled_orthogonal,
    select_component,
)
from thesis_exp.exp60_geometry_matched_shuffle.finalize_real_preflight import (
    treatment_separation_by_seed,
)
from thesis_exp.exp60_geometry_matched_shuffle.mapping import (
    build_maximum_mismatch_mapping,
    mapping_sha256,
    mapping_target_lookup,
    theoretical_maximum_changes,
)
from thesis_exp.exp60_geometry_matched_shuffle.preflight import run as run_preflight
from thesis_exp.exp60_geometry_matched_shuffle.train import (
    assert_formal_config_matches_protocol,
    assert_gpu_slot_assignment,
    file_manifest,
    verify_contract,
)


class Exp60MappingTest(unittest.TestCase):
    def test_actual_mapping_file_hash_and_train_coverage_are_canonical(self) -> None:
        from thesis_exp.exp57_cbrd.data_audit import model_rows

        mapping = [
            json.loads(line)
            for line in MAPPING_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        audit = json.loads(MAPPING_AUDIT_PATH.read_text(encoding="utf-8"))
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        actual = mapping_sha256(mapping)
        self.assertEqual(actual, audit["mapping_sha256"])
        self.assertEqual(actual, protocol["mapping"]["canonical_sha256"])
        self.assertEqual(
            set(mapping_target_lookup(mapping)),
            {str(row["record_id"]) for row in model_rows("train")},
        )

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
        self.assertEqual(
            set(protocol["formal_runs"]["real_preflight_gpu_schedule"].values()),
            {0, 1, 2},
        )
        self.assertEqual(
            protocol["implementation_gates_before_training"][
                "preflight_treatment_component_activity_ratio_at_least"
            ],
            1e-6,
        )
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

    def test_model_manifest_excludes_only_unreadable_modelscope_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".msc").write_text("private metadata", encoding="utf-8")
            (root / "config.json").write_text("{}", encoding="utf-8")
            manifest = file_manifest(root)
        self.assertNotIn(".msc", manifest["files"])
        self.assertIn("config.json", manifest["files"])

    def test_latin_square_assignment_is_enforced(self) -> None:
        assert_gpu_slot_assignment(47, "aligned_orthogonal_only", 1)
        with self.assertRaisesRegex(RuntimeError, "Latin-square mismatch"):
            assert_gpu_slot_assignment(47, "aligned_orthogonal_only", 0)

    def test_cpu_no_update_preflight_passes(self) -> None:
        report = run_preflight()
        self.assertEqual(report["status"], "EXP60_CPU_NO_UPDATE_PREFLIGHT_PASS")
        self.assertTrue(all(report["checks"].values()))

    def test_normalized_protocol_allows_only_freeze_fields(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        baseline = normalized_scientific_protocol_sha256(protocol)
        allowed = copy.deepcopy(protocol)
        allowed["status"] = "EXP60_PROTOCOL_FROZEN_BEFORE_FORMAL_RESULTS"
        allowed["formal_runs"]["physical_gpu_bindings"] = {
            "gpu_slot_0": "4",
            "gpu_slot_1": "6",
            "gpu_slot_2": "7",
        }
        allowed["formal_freeze_timestamp"] = "future"
        self.assertEqual(baseline, normalized_scientific_protocol_sha256(allowed))
        changed = copy.deepcopy(protocol)
        changed["fixed_training"]["learning_rate"] = 3e-5
        self.assertNotEqual(baseline, normalized_scientific_protocol_sha256(changed))

    def test_preflight_source_lock_is_no_training_authority_and_verifies(self) -> None:
        self.assertTrue(PREFLIGHT_SOURCE_LOCK_PATH.is_file())
        lock = json.loads(PREFLIGHT_SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            lock["status"], "EXP60_PREFLIGHT_SOURCE_LOCK_NO_TRAINING_AUTHORITY"
        )
        self.assertEqual(lock["optimizer_steps"], 0)
        self.assertEqual(lock["test_access_count"], 0)
        binding = verify_preflight_source_lock()
        self.assertEqual(
            binding["normalized_scientific_protocol_sha256"],
            lock["normalized_scientific_protocol_sha256"],
        )

    def test_gpu_identity_fails_without_stable_uuid(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda _device: SimpleNamespace(
                    uuid=None, name="fake", total_memory=24
                )
            )
        )
        command = SimpleNamespace(returncode=1, stdout="")
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4"}, clear=False), patch(
            "thesis_exp.exp60_geometry_matched_shuffle.contract.subprocess.run",
            return_value=command,
        ):
            with self.assertRaisesRegex(RuntimeError, "STABLE_GPU_IDENTITY_UNAVAILABLE"):
                stable_gpu_identity(fake_torch, "cuda")

    def test_gpu_identity_requires_uuid_sources_to_agree_and_rejects_mig(self) -> None:
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                get_device_properties=lambda _device: SimpleNamespace(
                    uuid="GPU-aaaaaaaa", name="fake", total_memory=24
                )
            )
        )
        disagree = SimpleNamespace(
            returncode=0,
            stdout="GPU-bbbbbbbb, 00000000:01:00.0, Disabled\n",
        )
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4"}, clear=False), patch(
            "thesis_exp.exp60_geometry_matched_shuffle.contract.subprocess.run",
            return_value=disagree,
        ):
            with self.assertRaisesRegex(RuntimeError, "UUID_SOURCES_DISAGREE"):
                stable_gpu_identity(fake_torch, "cuda")
        mig = SimpleNamespace(
            returncode=0,
            stdout="GPU-aaaaaaaa, 00000000:01:00.0, Enabled\n",
        )
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4"}, clear=False), patch(
            "thesis_exp.exp60_geometry_matched_shuffle.contract.subprocess.run",
            return_value=mig,
        ):
            with self.assertRaisesRegex(RuntimeError, "MIG_ENVIRONMENT_NOT_AUTHORIZED"):
                stable_gpu_identity(fake_torch, "cuda")

    def test_formal_source_lock_rejects_empty_and_incomplete_manifests(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        decision = {"status": "EXP60_REAL_MODEL_PREFLIGHT_ALL_SEEDS_PASS"}
        normalized = normalized_scientific_protocol_sha256(protocol)
        base = {
            "schema_version": FORMAL_SOURCE_LOCK_SCHEMA_VERSION,
            "status": "EXP60_FORMAL_SOURCE_LOCK",
            "protocol_sha256": "protocol-sha",
            "real_model_preflight_decision_sha256": "decision-sha",
            "normalized_scientific_protocol_sha256": normalized,
            "mandatory_file_manifest_sha256": manifest_sha256(FORMAL_MANDATORY_FILES),
            "contains_frozen_analysis": True,
            "physical_gpu_bindings_equal_preflight_devices": True,
            "allowed_splits": ["train", "dev"],
            "test_access_count": 0,
            "files": {},
            "file_count": 0,
        }
        with patch(
            "thesis_exp.exp60_geometry_matched_shuffle.contract.sha256_file",
            side_effect=["protocol-sha", "decision-sha"],
        ):
            with self.assertRaisesRegex(RuntimeError, "file manifest is empty"):
                validate_formal_source_lock(base, protocol, decision)
        incomplete = copy.deepcopy(base)
        incomplete["files"] = {FORMAL_MANDATORY_FILES[0]: "sha"}
        incomplete["file_count"] = 1
        with patch(
            "thesis_exp.exp60_geometry_matched_shuffle.contract.sha256_file",
            side_effect=["protocol-sha", "decision-sha"],
        ):
            with self.assertRaisesRegex(RuntimeError, "misses mandatory files"):
                validate_formal_source_lock(incomplete, protocol, decision)


class Exp60AnalysisTest(unittest.TestCase):
    def test_treatment_separation_is_nondegenerate_and_required_per_seed(self) -> None:
        good = {
            "storage_component_cosines": [0.98, 0.999],
            "storage_component_relative_distances": [0.2, 0.01],
            "storage_component_activity_ratios": [1e-4, 2e-4],
        }
        observations = {seed: copy.deepcopy(good) for seed in (47, 48, 49)}
        passed = treatment_separation_by_seed(
            observations, cosine_max=0.99, distance_min=0.1, activity_min=1e-6
        )
        self.assertTrue(all(passed.values()))
        observations[48]["storage_component_activity_ratios"] = [0.0, 0.0]
        observations[49]["storage_component_cosines"] = [0.999, 0.999]
        failed = treatment_separation_by_seed(
            observations, cosine_max=0.99, distance_min=0.1, activity_min=1e-6
        )
        self.assertTrue(failed["47"])
        self.assertFalse(failed["48"])
        self.assertFalse(failed["49"])

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
