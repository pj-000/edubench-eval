from __future__ import annotations

import json
import unittest

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_sorc_dpo_pairs import raw_source_key
from thesis_exp.exp54_rar_sft.build_sorc_dpo_pairs import (
    ScoreEvidence,
    build_hybrid_score_pairs,
    build_matched_synthetic_pairs,
    build_rationale_pairs,
    choose_adjacent,
    choose_severe_l2h,
    odpo_offset,
)


def train_fixture() -> list[dict]:
    counts = {1: 24, 2: 52, 3: 251, 4: 946, 5: 1381}
    rows = []
    position = 0
    for label, count in counts.items():
        for _ in range(count):
            rows.append(
                {
                    "record_id": f"record-{position:04d}",
                    "label_5": label,
                    "metric_id": f"M{position % 12:02d}",
                    "language": "en" if position % 2 == 0 else "zh",
                }
            )
            position += 1
    return rows


def evidence_row(record_id: str, gold: int, score: int, seed: int) -> dict:
    return {
        "record_id": record_id,
        "gold_label": gold,
        "metric_id": "M",
        "language": "en",
        "generator_arm": "R3",
        "generator_seed": seed,
        "generator_epoch": 3,
        "generator_adapter_sha256": str(seed) * 64,
        "generation_mode": "greedy",
        "rollout_seed": None,
        "parse_success": True,
        "generated_score": score,
        "generated_rationale": "Evidence without a score.",
        "forced_completion": False,
        "signed_error": score - gold,
        "absolute_error": abs(score - gold),
        "error_class": "fixture",
        "severe_low_to_high": gold <= 2 and score >= 4,
        "severe_high_to_low": gold >= 4 and score <= 2,
    }


def value(record_id: str, gold: int, score: int, seeds=(42,)) -> ScoreEvidence:
    return ScoreEvidence(
        score=score,
        rows=tuple(
            evidence_row(record_id, gold, score, seed) for seed in seeds
        ),
    )


class SORCDPOPairTest(unittest.TestCase):
    def test_identical_raw_output_hashes_are_seed_scoped(self) -> None:
        self.assertNotEqual(
            raw_source_key("a" * 64, 42),
            raw_source_key("a" * 64, 43),
        )
        with self.assertRaises(ValueError):
            raw_source_key("a" * 64, 45)

    def test_protocol_freezes_training_boundary(self) -> None:
        path = (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/configs/"
            "sorc_dpo_pair_protocol_v1.json"
        )
        protocol = json.loads(path.read_text())
        self.assertEqual(protocol["method_name"], "SORC-DPO")
        self.assertFalse(protocol["stochastic_rollout_allowed"])
        self.assertFalse(protocol["preference_training_allowed"])
        self.assertFalse(protocol["dev_accessed"])
        self.assertFalse(protocol["test_accessed"])

    def test_odpo_offset_is_not_beta_scaled(self) -> None:
        self.assertAlmostEqual(odpo_offset(2, 3), 0.125)
        self.assertAlmostEqual(odpo_offset(2, 4), 0.75)
        self.assertAlmostEqual(odpo_offset(1, 5), 1.0)
        self.assertAlmostEqual(odpo_offset(5, 2), 0.375)

    def test_actual_error_selection_is_deterministic(self) -> None:
        record_id = "r"
        evidence = {
            3: value(record_id, 2, 3, (42, 43)),
            4: value(record_id, 2, 4, (42, 43)),
            5: value(record_id, 2, 5, (44,)),
        }
        self.assertEqual(choose_adjacent(record_id, 2, evidence).score, 3)
        self.assertEqual(
            choose_severe_l2h(record_id, 2, evidence).score, 4
        )

    def test_hybrid_blocks_use_record_units_and_minimal_backfill(self) -> None:
        rows = train_fixture()
        grouped = {str(row["record_id"]): {} for row in rows}
        low = [row for row in rows if row["label_5"] in {1, 2}]
        for row in low[:53]:
            record_id = str(row["record_id"])
            grouped[record_id][4] = value(
                record_id, int(row["label_5"]), 4
            )
        label3 = [row for row in rows if row["label_5"] == 3]
        for row in label3[:100]:
            record_id = str(row["record_id"])
            grouped[record_id][4] = value(record_id, 3, 4)
        label4 = [row for row in rows if row["label_5"] == 4]
        for row in label4[:39]:
            record_id = str(row["record_id"])
            grouped[record_id][3] = value(record_id, 4, 3)
        label5 = [row for row in rows if row["label_5"] == 5]
        for row in label5[:217]:
            record_id = str(row["record_id"])
            grouped[record_id][4] = value(record_id, 5, 4)

        hybrid = build_hybrid_score_pairs(rows, grouped, {})
        by_type = {}
        for pair in hybrid:
            by_type.setdefault(pair["pair_type"], []).append(pair)
            self.assertEqual(
                pair["chosen"]["rationale"],
                pair["rejected"]["rationale"],
            )
        self.assertEqual(len(by_type["adjacent_score"]), 356)
        self.assertEqual(len(by_type["severe_l2h"]), 76)
        self.assertEqual(len(by_type["h2l_guard"]), 76)
        self.assertEqual(
            sum(
                pair["pair_source"] == "actual_controlled"
                for pair in by_type["severe_l2h"]
            ),
            53,
        )
        self.assertEqual(
            sum(
                pair["pair_source"] == "synthetic_backfill"
                for pair in by_type["h2l_guard"]
            ),
            13,
        )
        for block, pairs in by_type.items():
            with self.subTest(block=block):
                self.assertEqual(
                    len(pairs),
                    len({pair["record_id"] for pair in pairs}),
                )

        synthetic = build_matched_synthetic_pairs(hybrid, rows, {})
        self.assertEqual(len(synthetic), len(hybrid))
        self.assertEqual(
            {
                (pair["pair_type"], pair["record_id"])
                for pair in synthetic
            },
            {
                (pair["pair_type"], pair["record_id"])
                for pair in hybrid
            },
        )
        self.assertEqual(
            {pair["pair_source"] for pair in synthetic},
            {"synthetic_control"},
        )

    def test_rationale_pair_changes_only_rationale(self) -> None:
        train = [
            {
                "record_id": "r",
                "label_5": 2,
                "metric_id": "M",
                "language": "en",
            }
        ]
        shared = {
            "base_event_id": "e",
            "record_id": "r",
            "score_target": 2,
            "rationale_active": True,
            "truncated": False,
        }
        r2 = {
            **shared,
            "rationale": "A relevant but mismatched rationale.",
            "arm_selected_reference_id": "r2-ref",
            "arm_rationale_source_event_id": "donor-e",
        }
        r3 = {
            **shared,
            "rationale": "The answer omits the required derivation.",
            "arm_selected_reference_id": "r3-ref",
            "arm_rationale_source_event_id": "e",
        }
        pairs = build_rationale_pairs(train, {"e": (r2, r3)})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            pairs[0]["chosen"]["score"], pairs[0]["rejected"]["score"]
        )
        self.assertNotEqual(
            pairs[0]["chosen"]["rationale"],
            pairs[0]["rejected"]["rationale"],
        )
        self.assertEqual(pairs[0]["score_loss_mask"], "off")


if __name__ == "__main__":
    unittest.main()
