"""Build the private P2-vs-P3 rationale blind-audit package.

The 40-row selector is inherited from the frozen RAR-SFT audit and reads only
dev metadata.  P2/P3 outputs are joined after selection.  Both A/B orientations
are emitted for score-blind and score-visible stages.  Private tasks and the
answer key must not be committed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.build_minimal_rationale_audit import (
    compact_json,
    read_jsonl,
    select_rows,
    sha256_bytes,
    stable_hash,
    write_json,
    write_jsonl,
)


SEEDS = (42, 43, 44)
TARGET_ARM = "P3_JOINT_SORC"
COMPARATOR_ARM = "P2_SORC_SCORE"
EXPECTED_DEV_ROWS = 664
DEFAULT_DEV = (
    REPO_ROOT / "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl"
)
DEFAULT_DEV_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_lr5e6_followup/dev"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_lr5e6_followup/rationale_blind_audit"
)


def _prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_DEV_ROWS:
        raise ValueError(f"{path}: expected {EXPECTED_DEV_ROWS} predictions")
    output = {}
    for row_position, row in enumerate(rows):
        record_id = str(row["record_id"])
        prediction = row.get("prediction")
        if (
            not record_id
            or record_id in output
            or row.get("parse_success") is not True
            or not isinstance(prediction, dict)
            or not isinstance(prediction.get("score"), int)
            or not 1 <= int(prediction["score"]) <= 5
            or not isinstance(prediction.get("rationale"), str)
            or not str(prediction["rationale"]).strip()
        ):
            raise ValueError(f"{path}:{row_position}: malformed prediction")
        output[record_id] = {
            "row_position": row_position,
            "score": int(prediction["score"]),
            "rationale": str(prediction["rationale"]),
            "forced_completion": bool(row.get("forced_completion")),
        }
    return output


def _prediction_path(root: Path, arm: str, seed: int) -> Path:
    return root / arm.lower() / f"seed_{seed}/predictions.jsonl"


def build_package(
    *,
    dev_path: Path,
    dev_root: Path,
    output: Path,
) -> dict[str, Any]:
    dev_rows = read_jsonl(dev_path)
    if len(dev_rows) != EXPECTED_DEV_ROWS:
        raise ValueError("dev row inventory differs")
    selected = select_rows(dev_rows)
    selected_ids = [str(row["record_id"]) for row in selected]
    dev_by_id = {str(row["record_id"]): row for row in dev_rows}
    expected_order = [str(row["record_id"]) for row in dev_rows]

    predictions = {}
    prediction_hashes = {}
    for arm in (TARGET_ARM, COMPARATOR_ARM):
        for seed in SEEDS:
            path = _prediction_path(dev_root, arm, seed)
            index = _prediction_index(path)
            if list(index) != expected_order:
                raise ValueError(f"{arm}/{seed}: prediction order differs")
            predictions[(arm, seed)] = index
            prediction_hashes[f"{arm}|seed={seed}"] = sha256_bytes(
                path.read_bytes()
            )

    sample_manifest = []
    for selected_row in selected:
        context = dev_by_id[str(selected_row["record_id"])]
        sample_manifest.append(
            {
                **selected_row,
                "question": str(context["question"]),
                "answer": str(context["answer"]),
                "rubric": context["rubric"],
            }
        )

    score_blind_tasks = []
    score_visible_tasks = []
    answer_key = []
    forced_counts = Counter()
    for seed in SEEDS:
        for record_id in selected_ids:
            context = dev_by_id[record_id]
            pair_id = stable_hash(
                "exp54-sorc-dpo-rationale-pair-v1",
                seed,
                record_id,
            )
            base_swapped = int(pair_id[:2], 16) % 2 == 1
            for orientation in (0, 1):
                swapped = base_swapped ^ bool(orientation)
                arm_order = (
                    (COMPARATOR_ARM, TARGET_ARM)
                    if swapped
                    else (TARGET_ARM, COMPARATOR_ARM)
                )
                arm_a, arm_b = arm_order
                candidate_a = predictions[(arm_a, seed)][record_id]
                candidate_b = predictions[(arm_b, seed)][record_id]
                presentation_id = stable_hash(
                    pair_id, "orientation", orientation
                )
                common = {
                    "presentation_id": presentation_id,
                    "question": str(context["question"]),
                    "answer": str(context["answer"]),
                    "metric_id": str(context["metric_id"]),
                    "metric": str(
                        context.get("metric_canonical")
                        or context.get("metric_raw")
                        or context["metric_id"]
                    ),
                    "rubric": context["rubric"],
                }
                score_blind_tasks.append(
                    {
                        **common,
                        "stage": "score_blind",
                        "candidate_a": {
                            "rationale": candidate_a["rationale"]
                        },
                        "candidate_b": {
                            "rationale": candidate_b["rationale"]
                        },
                    }
                )
                score_visible_tasks.append(
                    {
                        **common,
                        "stage": "score_visible",
                        "candidate_a": {
                            "score": candidate_a["score"],
                            "rationale": candidate_a["rationale"],
                        },
                        "candidate_b": {
                            "score": candidate_b["score"],
                            "rationale": candidate_b["rationale"],
                        },
                    }
                )
                answer_key.append(
                    {
                        "presentation_id": presentation_id,
                        "pair_id": pair_id,
                        "orientation": orientation,
                        "record_id": record_id,
                        "row_position": int(
                            predictions[(arm_a, seed)][record_id][
                                "row_position"
                            ]
                        ),
                        "label_5": int(context["label_5"]),
                        "seed": seed,
                        "candidate_a_arm": arm_a,
                        "candidate_b_arm": arm_b,
                        "candidate_a_forced": candidate_a[
                            "forced_completion"
                        ],
                        "candidate_b_forced": candidate_b[
                            "forced_completion"
                        ],
                    }
                )
            for arm in (TARGET_ARM, COMPARATOR_ARM):
                forced_counts[
                    (
                        arm,
                        predictions[(arm, seed)][record_id][
                            "forced_completion"
                        ],
                    )
                ] += 1

    if (
        len(score_blind_tasks) != 240
        or len(score_visible_tasks) != 240
        or len(answer_key) != 240
    ):
        raise ValueError("audit presentation inventory differs")
    private = output / "private"
    bindings = {
        "sample_manifest": write_jsonl(
            private / "sample_manifest.jsonl", sample_manifest
        ),
        "score_blind_tasks": write_jsonl(
            private / "score_blind_tasks.jsonl", score_blind_tasks
        ),
        "score_visible_tasks": write_jsonl(
            private / "score_visible_tasks.jsonl", score_visible_tasks
        ),
        "answer_key": write_jsonl(
            private / "answer_key.jsonl", answer_key
        ),
    }
    report = {
        "schema_version": "exp54-sorc-dpo-rationale-audit-package-v1",
        "status": "SORC_DPO_RATIONALE_AUDIT_PACKAGE_READY",
        "comparison": {
            "target": TARGET_ARM,
            "comparator": COMPARATOR_ARM,
            "estimand": (
                "incremental visible rationale-alignment value of the P3 "
                "rationale preference block over P2 score-only risk DPO"
            ),
        },
        "sample": {
            "unique_records": 40,
            "seeds": list(SEEDS),
            "pair_instances": 120,
            "all_label_1_2_rows_included": True,
            "selection_reused_from_frozen_rar_sft_audit": True,
            "selection_reads_model_outputs": False,
        },
        "tasks": {
            "score_blind_presentations": len(score_blind_tasks),
            "score_visible_presentations": len(score_visible_tasks),
            "orientations_per_pair": 2,
            "evaluator_agents_required_per_stage": 2,
        },
        "forced_completion_summary": {
            f"{arm}|forced={str(forced).lower()}": count
            for (arm, forced), count in sorted(forced_counts.items())
        },
        "private_bindings": bindings,
        "prediction_source_hashes": prediction_hashes,
        "private_row_level_content_committed": False,
        "audit_completed": False,
        "dev_regenerated": False,
        "test_accessed": False,
    }
    write_json(output / "candidate_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--dev-root", type=Path, default=DEFAULT_DEV_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_package(
        dev_path=args.dev,
        dev_root=args.dev_root,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "sample": report["sample"],
                "tasks": report["tasks"],
                "dev_regenerated": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
