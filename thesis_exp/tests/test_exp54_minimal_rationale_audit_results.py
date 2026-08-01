from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from thesis_exp.exp54_rar_sft.collect_minimal_rationale_audit_results import (
    STAGE_DIMENSIONS,
    bootstrap_interval,
    collect_exploratory_agent_primary_only,
    collect_results,
    map_ab_to_r3,
    pair_verdicts,
)


def synthetic_inputs() -> tuple[
    list[dict[str, object]],
    dict[str, list[dict[str, object]]],
]:
    answer_key: list[dict[str, object]] = []
    judgments = {"score_blind": [], "score_visible": []}
    for comparison, comparator in (("primary", "R2"), ("secondary", "R1")):
        for seed in (42, 43, 44):
            for record_index in range(40):
                pair_id = f"{comparison}-s{seed}-r{record_index:02d}"
                for orientation in (0, 1):
                    presentation_id = (
                        f"{comparison}-s{seed}-r{record_index:02d}-o"
                        f"{orientation}"
                    )
                    a_arm, b_arm = (
                        ("R3", comparator)
                        if orientation == 0
                        else (comparator, "R3")
                    )
                    answer_key.append(
                        {
                            "presentation_id": presentation_id,
                            "pair_id": pair_id,
                            "comparison": comparison,
                            "orientation": orientation,
                            "record_id": f"private-record-{record_index:02d}",
                            "row_position": record_index,
                            "label_5": 1 + record_index % 5,
                            "seed": seed,
                            "candidate_a_arm": a_arm,
                            "candidate_b_arm": b_arm,
                            "candidate_a_forced": (
                                a_arm == "R3" and record_index % 7 == 0
                            ),
                            "candidate_b_forced": (
                                b_arm == "R3" and record_index % 7 == 0
                            ),
                        }
                    )
                    winner = "A" if a_arm == "R3" else "B"
                    for stage, dimensions in STAGE_DIMENSIONS.items():
                        judgments[stage].append(
                            {
                                "presentation_id": presentation_id,
                                **{
                                    dimension: winner
                                    for dimension in dimensions
                                },
                                "brief_reason": (
                                    f"private-reason-{stage}-{pair_id}-"
                                    f"{orientation}"
                                ),
                            }
                        )
    return answer_key, judgments


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class MinimalRationaleAuditResultsTest(unittest.TestCase):
    def test_ab_mapping_uses_arm_identity(self) -> None:
        self.assertEqual(
            map_ab_to_r3(
                "A",
                candidate_a_arm="R3",
                candidate_b_arm="R2",
                comparator_arm="R2",
            ),
            "win",
        )
        self.assertEqual(
            map_ab_to_r3(
                "B",
                candidate_a_arm="R2",
                candidate_b_arm="R3",
                comparator_arm="R2",
            ),
            "win",
        )
        self.assertEqual(
            map_ab_to_r3(
                "A",
                candidate_a_arm="R1",
                candidate_b_arm="R3",
                comparator_arm="R1",
            ),
            "loss",
        )

    def test_orientation_disagreement_becomes_tie(self) -> None:
        answer_key, judgments = synthetic_inputs()
        target = next(
            row
            for row in judgments["score_visible"]
            if row["presentation_id"] == "primary-s42-r00-o1"
        )
        target["overall_preference"] = "A"
        pairs = pair_verdicts(
            answer_key,
            judgments["score_visible"],
            stage="score_visible",
        )
        selected = next(
            row
            for row in pairs
            if row["record_id"] == "private-record-00"
            and row["seed"] == 42
            and row["comparison"] == "primary"
        )
        self.assertEqual(selected["verdicts"]["overall_preference"], "tie")
        self.assertEqual(
            selected["verdicts"][
                "overall_scoring_justification_usefulness"
            ],
            "win",
        )

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        answer_key, judgments = synthetic_inputs()
        pairs = pair_verdicts(
            answer_key,
            judgments["score_visible"],
            stage="score_visible",
        )
        primary = [
            row for row in pairs if row["comparison"] == "primary"
        ]
        first = bootstrap_interval(
            primary,
            dimension="overall_preference",
            replicates=257,
            seed=20260728,
        )
        second = bootstrap_interval(
            primary,
            dimension="overall_preference",
            replicates=257,
            seed=20260728,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["valid_replicates"], 257)
        self.assertEqual(
            first["tie_adjusted_preference"],
            [1.0, 1.0],
        )

    def test_public_outputs_exclude_private_fields_and_values(self) -> None:
        answer_key, judgments = synthetic_inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            answer_path = root / "answer_key.jsonl"
            write_jsonl(answer_path, answer_key)
            paths = {}
            for evaluator in ("judge_one", "judge_two"):
                for stage in STAGE_DIMENSIONS:
                    path = root / f"{evaluator}_{stage}.jsonl"
                    write_jsonl(path, judgments[stage])
                    paths[(evaluator, stage)] = path
            output = root / "public"
            report = collect_results(
                answer_key_path=answer_path,
                judgment_paths=paths,
                output_dir=output,
                replicates=31,
            )
            self.assertEqual(
                report["status"],
                "MINIMAL_RATIONALE_AUDIT_RESULTS_COLLECTED",
            )
            for path in output.iterdir():
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("record_id", text)
                self.assertNotIn("presentation_id", text)
                self.assertNotIn("brief_reason", text)
                self.assertNotIn("private-record-00", text)
                self.assertNotIn("private-reason-", text)

    def test_exploratory_primary_only_requires_240_and_limits_claims(
        self,
    ) -> None:
        answer_key, judgments = synthetic_inputs()
        primary_ids = {
            str(row["presentation_id"])
            for row in answer_key
            if row["comparison"] == "primary"
        }
        primary_judgments = [
            row
            for row in judgments["score_visible"]
            if str(row["presentation_id"]) in primary_ids
        ]
        self.assertEqual(len(primary_judgments), 240)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            answer_path = root / "answer_key.jsonl"
            write_jsonl(answer_path, answer_key)
            paths = {}
            for evaluator in ("codex_agent_one", "codex_agent_two"):
                path = root / f"{evaluator}.jsonl"
                write_jsonl(path, primary_judgments)
                paths[evaluator] = path
            output = root / "exploratory_public"
            report = collect_exploratory_agent_primary_only(
                answer_key_path=answer_path,
                judgment_paths=paths,
                output_dir=output,
                replicates=31,
            )
            self.assertEqual(
                report["mode"],
                "exploratory_agent_primary_only",
            )
            self.assertFalse(
                report["evaluator_family_independence_satisfied"]
            )
            self.assertFalse(
                report["formal_preregistered_two_family_audit_complete"]
            )
            self.assertFalse(
                report["orientation_judgments_context_isolated"]
            )
            self.assertFalse(
                report["orientation_bias_diagnostic_interpretable"]
            )
            self.assertEqual(
                report["claim_scope"],
                "Codex-agent exploratory primary score-visible preference",
            )
            self.assertEqual(
                {
                    (
                        row["comparison"],
                        row["comparison_role"],
                        row["stage"],
                    )
                    for row in report["aggregates"]
                },
                {("R3_vs_R2", "primary", "score_visible")},
            )
            self.assertTrue(report["cross_evaluator_agreement"])
            self.assertEqual(
                {
                    row["pairs"]
                    for row in report["cross_evaluator_agreement"]
                },
                {120},
            )
            for path in output.iterdir():
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("private-record-00", text)
                self.assertNotIn("private-reason-", text)

            truncated = root / "truncated.jsonl"
            write_jsonl(truncated, primary_judgments[:-1])
            with self.assertRaisesRegex(ValueError, "must be 240"):
                collect_exploratory_agent_primary_only(
                    answer_key_path=answer_path,
                    judgment_paths={
                        "codex_agent_one": truncated,
                        "codex_agent_two": paths["codex_agent_two"],
                    },
                    output_dir=root / "must_not_complete",
                    replicates=7,
                )


if __name__ == "__main__":
    unittest.main()
