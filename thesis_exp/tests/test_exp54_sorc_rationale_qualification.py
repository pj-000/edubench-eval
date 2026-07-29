from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from thesis_exp.exp54_rar_sft.audit_sorc_rationale_qualification import (
    validate_candidate_boundary,
    validate_result_identity,
    validate_source_hashes,
)
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file
from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.build_sorc_rationale_qualification import (
    EXPECTED_PAIRS,
    EXPECTED_PRESENTATIONS_PER_STAGE,
    QUOTAS,
    build_package,
    label_band,
    select_pairs,
    stable_hash,
)


def fixtures() -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    pairs: list[dict] = []
    labels = [1] * 30 + [2] * 30 + [3] * 45 + [4] * 75 + [5] * 75
    for position, label in enumerate(labels):
        record_id = f"record-{position:04d}"
        metric_id = f"M{position % 12:02d}"
        language = "en" if position % 2 == 0 else "zh"
        train.append(
            {
                "record_id": record_id,
                "label_5": label,
                "metric_id": metric_id,
                "language": language,
                "question": f"Question {position}",
                "answer": f"Answer {position}",
                "metric_canonical": f"Metric {metric_id}",
                "rubric": ["1: weak", "5: strong"],
            }
        )
        chosen = f"Aligned rationale {position}"
        rejected = f"Shuffled rationale {position}"
        pairs.append(
            {
                "record_id": record_id,
                "pair_hash": stable_hash("pair", position),
                "pair_type": "rationale_alignment",
                "pair_source": "rationale_control",
                "score_loss_mask": "off",
                "rationale_loss_mask": "rationale_content_tokens_only",
                "gold_label": label,
                "metric_id": metric_id,
                "language": language,
                "chosen": {"score": label, "rationale": chosen},
                "rejected": {"score": label, "rationale": rejected},
                "chosen_rationale_sha256": stable_hash_bytes(chosen),
                "rejected_rationale_sha256": stable_hash_bytes(rejected),
            }
        )
    return train, pairs


def stable_hash_bytes(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SORCRationaleQualificationTest(unittest.TestCase):
    def test_prompt_defines_every_required_contract_field(self) -> None:
        root = REPO_ROOT / "thesis_exp/exp54_rar_sft"
        for stage in ("score_blind", "score_visible"):
            prompt = (
                root / f"prompts/rationale_audit_{stage}_v2.txt"
            ).read_text(encoding="utf-8")
            schema = json.loads(
                (
                    root
                    / f"schemas/rationale_audit_{stage}_v2.schema.json"
                ).read_text(encoding="utf-8")
            )
            for field in schema["required"]:
                with self.subTest(stage=stage, field=field):
                    self.assertIn(field, prompt)
            normalized_prompt = " ".join(prompt.split())
            self.assertIn(
                "Copy the supplied presentation_id exactly",
                normalized_prompt,
            )
            self.assertIn("overall_preference", prompt)
            self.assertIn("brief_reason", prompt)

    def test_qualification_rule_is_fixed_before_execution(self) -> None:
        path = (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/configs/"
            "sorc_rationale_qualification_rule_v1.json"
        )
        rule = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(rule["primary_stage"], "score_blind")
        self.assertEqual(rule["primary_field"], "overall_preference")
        self.assertFalse(rule["row_level_filtering_or_selection_allowed"])
        self.assertFalse(rule["rationale_blind_qualification_completed"])
        self.assertFalse(rule["p3_preference_training_allowed"])

    def test_fail_closed_rejects_premature_p3_flags(self) -> None:
        base = {
            "rationale_blind_qualification_completed": False,
            "p3_preference_training_allowed": False,
            "preference_training_allowed": False,
            "dev_accessed": False,
            "test_accessed": False,
        }
        validate_candidate_boundary(base)
        for field in (
            "rationale_blind_qualification_completed",
            "p3_preference_training_allowed",
        ):
            with self.subTest(field=field):
                changed = {**base, field: True}
                with self.assertRaises(ValueError):
                    validate_candidate_boundary(changed)

    def test_source_hash_binding_detects_one_byte_change(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("original", encoding="utf-8")
            locked = {"prompt": sha256_file(path)}
            validate_source_hashes(locked, {"prompt": path})
            path.write_text("original!", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_source_hashes(locked, {"prompt": path})

    def test_selector_is_deterministic_balanced_and_complete(self) -> None:
        _, pairs = fixtures()
        selected = select_pairs(pairs)
        reverse_selected = select_pairs(list(reversed(pairs)))
        self.assertEqual(
            [row["pair_hash"] for row in selected],
            [row["pair_hash"] for row in reverse_selected],
        )
        self.assertEqual(len(selected), EXPECTED_PAIRS)
        self.assertEqual(
            {
                band: sum(
                    label_band(int(row["gold_label"])) == band
                    for row in selected
                )
                for band in QUOTAS
            },
            QUOTAS,
        )
        self.assertEqual(len({row["metric_id"] for row in selected}), 12)
        self.assertEqual({row["language"] for row in selected}, {"en", "zh"})

    def test_package_has_two_inverse_orientations_per_pair(self) -> None:
        train, pairs = fixtures()
        package = build_package(train_rows=train, source_pairs=pairs)
        self.assertEqual(len(package["sample_manifest"]), EXPECTED_PAIRS)
        self.assertEqual(
            len(package["score_blind_tasks"]),
            EXPECTED_PRESENTATIONS_PER_STAGE,
        )
        self.assertEqual(
            len(package["score_visible_tasks"]),
            EXPECTED_PRESENTATIONS_PER_STAGE,
        )
        keys_by_pair: dict[str, list[dict]] = {}
        for row in package["answer_key"]:
            keys_by_pair.setdefault(row["pair_id"], []).append(row)
        for rows in keys_by_pair.values():
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                {row["a_source"] for row in rows},
                {"R3_ALIGNED", "R2_SHUFFLED"},
            )
            self.assertEqual(
                {row["stage"] for row in rows},
                {"score_blind", "score_visible"},
            )
        blind = package["score_blind_tasks"][0]
        visible = package["score_visible_tasks"][0]
        self.assertNotIn("score", blind["candidate_a"])
        self.assertIn("score", visible["candidate_a"])
        self.assertEqual(blind["stage"], "score_blind")
        self.assertEqual(visible["stage"], "score_visible")
        blind_ids = {
            row["presentation_id"]
            for row in package["score_blind_tasks"]
        }
        visible_ids = {
            row["presentation_id"]
            for row in package["score_visible_tasks"]
        }
        self.assertTrue(blind_ids.isdisjoint(visible_ids))
        self.assertEqual(
            len(blind_ids | visible_ids),
            2 * EXPECTED_PRESENTATIONS_PER_STAGE,
        )

    def test_cross_stage_identity_cannot_be_substituted(self) -> None:
        train, pairs = fixtures()
        package = build_package(train_rows=train, source_pairs=pairs)
        blind = package["score_blind_tasks"][0]
        visible = package["score_visible_tasks"][0]
        self.assertNotEqual(
            blind["presentation_id"], visible["presentation_id"]
        )
        self.assertNotEqual(blind["stage"], visible["stage"])
        blind_key = next(
            row
            for row in package["answer_key"]
            if row["presentation_id"] == blind["presentation_id"]
        )
        valid_result = {
            "presentation_id": blind["presentation_id"],
            "stage": "score_blind",
            "parsed_judgment": {
                "presentation_id": blind["presentation_id"],
                "stage": "score_blind",
            },
        }
        validate_result_identity(
            result=valid_result,
            task=blind,
            answer_key=blind_key,
        )
        swapped = {
            **valid_result,
            "stage": "score_visible",
            "parsed_judgment": {
                **valid_result["parsed_judgment"],
                "stage": "score_visible",
            },
        }
        with self.assertRaises(ValueError):
            validate_result_identity(
                result=swapped,
                task=blind,
                answer_key=blind_key,
            )

    def test_source_pair_must_change_only_rationale(self) -> None:
        _, pairs = fixtures()
        pairs[0]["rejected"]["score"] = 5
        with self.assertRaises(ValueError):
            select_pairs(pairs)


if __name__ == "__main__":
    unittest.main()
