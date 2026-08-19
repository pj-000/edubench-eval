from __future__ import annotations

import json
import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    ERROR_CLASSES,
    SEEDS,
    aggregate_failure_rows,
    classify_error,
    make_failure_row,
)
from thesis_exp.exp54_rar_sft.run_actual_failure_bank_vllm import (
    execute,
    vllm_token_prompts,
)


class ActualFailureBankTest(unittest.TestCase):
    def test_error_classes_cover_ordinal_directions(self) -> None:
        cases = {
            (2, 2): "correct",
            (2, 3): "adjacent_overestimate",
            (3, 2): "adjacent_underestimate",
            (2, 4): "severe_low_to_high",
            (5, 2): "severe_high_to_low",
            (1, 3): "other_overestimate",
            (5, 3): "other_underestimate",
        }
        for (gold, predicted), expected in cases.items():
            with self.subTest(gold=gold, predicted=predicted):
                self.assertEqual(
                    classify_error(
                        gold, predicted, parse_success=True
                    )["error_class"],
                    expected,
                )

    def test_invalid_output_is_separate_from_scoring_error(self) -> None:
        result = classify_error(2, None, parse_success=False)
        self.assertEqual(result["error_class"], "invalid_output")
        self.assertFalse(result["severe_low_to_high"])
        with self.assertRaises(ValueError):
            classify_error(2, 4, parse_success=False)

    def test_failure_row_records_actual_severe_l2h(self) -> None:
        source = {
            "record_id": "sample-a",
            "label_5": 2,
            "metric_id": "M1",
            "language": "en",
        }
        row = make_failure_row(
            source=source,
            row_position=0,
            generator_seed=42,
            adapter_sha256="a" * 64,
            generation_mode="greedy",
            rollout_seed=None,
            prediction={"score": 4, "rationale": "Fluent but wrong."},
            forced_completion=False,
        )
        self.assertEqual(row["error_class"], "severe_low_to_high")
        self.assertEqual(row["signed_error"], 2)
        self.assertTrue(row["severe_low_to_high"])

    def test_aggregate_counts_three_seed_inventory(self) -> None:
        rows = []
        for row_position in range(2654):
            gold = 2 if row_position < 76 else 5
            for seed in SEEDS:
                predicted = 4 if row_position == 0 else gold
                rows.append(
                    make_failure_row(
                        source={
                            "record_id": f"record-{row_position}",
                            "label_5": gold,
                            "metric_id": f"M{row_position % 12}",
                            "language": (
                                "en" if row_position % 2 == 0 else "zh"
                            ),
                        },
                        row_position=row_position,
                        generator_seed=seed,
                        adapter_sha256="a" * 64,
                        generation_mode="greedy",
                        rollout_seed=None,
                        prediction={
                            "score": predicted,
                            "rationale": "Reason",
                        },
                        forced_completion=False,
                    )
                )
        report = aggregate_failure_rows(rows)
        self.assertEqual(report["rows"], 7962)
        self.assertEqual(
            report["error_class_counts"]["severe_low_to_high"], 3
        )
        self.assertEqual(report["records_with_actual_severe_l2h"], 1)
        self.assertEqual(
            sum(report["error_class_counts"].values()), 7962
        )

    def test_schema_error_enum_matches_runtime(self) -> None:
        schema_path = (
            REPO_ROOT
            / "thesis_exp/exp54_rar_sft/schemas/"
            "actual_failure_bank_row_v1.schema.json"
        )
        schema = json.loads(schema_path.read_text())
        self.assertEqual(
            tuple(schema["properties"]["error_class"]["enum"]),
            ERROR_CLASSES,
        )

    def test_gpu_execution_requires_explicit_user_confirmation(self) -> None:
        args = argparse.Namespace(
            seed=42,
            max_num_seqs=128,
            gpu_memory_utilization=0.85,
        )
        with patch.dict(
            "os.environ",
            {"CUDA_VISIBLE_DEVICES": "7"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "EXP54_GPU_USER_CONFIRMED"
            ):
                execute(args)

    def test_vllm_011_token_prompt_contract(self) -> None:
        prepared = [
            {"token_ids": [1, 2, 3]},
            {"token_ids": (4, 5)},
        ]
        self.assertEqual(
            vllm_token_prompts(prepared),
            [
                {"prompt_token_ids": [1, 2, 3]},
                {"prompt_token_ids": [4, 5]},
            ],
        )


if __name__ == "__main__":
    unittest.main()
