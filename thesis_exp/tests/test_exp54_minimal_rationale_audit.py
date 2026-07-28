from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.build_minimal_rationale_audit import (
    select_rows,
)


def synthetic_dev() -> list[dict[str, object]]:
    rows = []
    metrics = [f"M{index:02d}" for index in range(12)]
    for index in range(664):
        if index < 6:
            label = 1
        elif index < 20:
            label = 2
        elif index < 140:
            label = 3
        elif index < 500:
            label = 4
        else:
            label = 5
        rows.append(
            {
                "record_id": f"record-{index:04d}",
                "label_5": label,
                "metric_id": metrics[index % len(metrics)],
                "language": "en" if index % 2 == 0 else "zh",
                "generated_rationale": f"must-not-be-read-{index}",
                "forced_completion": index % 7 == 0,
            }
        )
    return rows


class MinimalRationaleAuditTest(unittest.TestCase):
    def test_selector_fixes_size_bands_and_coverage(self) -> None:
        selected = select_rows(synthetic_dev())
        counts = Counter(int(row["label_5"]) for row in selected)
        self.assertEqual(len(selected), 40)
        self.assertEqual(counts[1] + counts[2], 20)
        self.assertEqual(counts[3], 8)
        self.assertEqual(counts[4] + counts[5], 12)
        self.assertEqual(len({row["metric_id"] for row in selected}), 12)
        self.assertEqual({row["language"] for row in selected}, {"en", "zh"})

    def test_selector_is_independent_of_model_output_fields(self) -> None:
        left = synthetic_dev()
        right = synthetic_dev()
        for row in right:
            row["generated_rationale"] = "completely changed"
            row["forced_completion"] = not bool(row["forced_completion"])
            row["arm"] = "R3"
        self.assertEqual(select_rows(left), select_rows(right))

    def test_selector_rejects_wrong_low_score_inventory(self) -> None:
        rows = synthetic_dev()
        rows[0]["label_5"] = 3
        with self.assertRaises(ValueError):
            select_rows(rows)

    def test_judgment_schemas_have_exact_preference_enums(self) -> None:
        schema_root = REPO_ROOT / "thesis_exp/exp54_rar_sft/schemas"
        for name in (
            "rationale_audit_score_blind_v2.schema.json",
            "rationale_audit_score_visible_v2.schema.json",
        ):
            schema = json.loads((schema_root / name).read_text())
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(
                schema["properties"]["overall_preference"]["enum"],
                ["A", "B", "tie"],
            )


if __name__ == "__main__":
    unittest.main()
