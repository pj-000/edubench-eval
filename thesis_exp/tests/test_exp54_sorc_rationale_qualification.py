from __future__ import annotations

import unittest

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
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["a_source"] for row in rows},
                {"R3_ALIGNED", "R2_SHUFFLED"},
            )
        blind = package["score_blind_tasks"][0]
        visible = package["score_visible_tasks"][0]
        self.assertNotIn("score", blind["candidate_a"])
        self.assertIn("score", visible["candidate_a"])

    def test_source_pair_must_change_only_rationale(self) -> None:
        _, pairs = fixtures()
        pairs[0]["rejected"]["score"] = 5
        with self.assertRaises(ValueError):
            select_pairs(pairs)


if __name__ == "__main__":
    unittest.main()
