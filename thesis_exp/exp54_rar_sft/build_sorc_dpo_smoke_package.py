"""Build a deterministic private train-only SORC-DPO smoke package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import read_jsonl, sha256_file
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import reject_eval_path


RAR_ROOT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
TRAINING_ROOT = RAR_ROOT / "preference_training_candidate"
DEFAULT_FROZEN_LOCK = TRAINING_ROOT / "preference_training_frozen_lock.json"
DEFAULT_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/sorc_dpo_smoke_plan_v1.json"
)
DEFAULT_OUTPUT = RAR_ROOT / "preference_smoke"
SOURCE_MANIFESTS = {
    "P1_FIELD_DPO": TRAINING_ROOT / "private/p1_field_dpo.jsonl",
    "P2_SORC_SCORE": TRAINING_ROOT / "private/p2_sorc_score.jsonl",
    "P3_JOINT_SORC": TRAINING_ROOT / "private/p3_joint_sorc.jsonl",
    "P1_SYN_SEED42": TRAINING_ROOT / "private/p1_syn_seed42.jsonl",
}
SCORE_TYPES = ("adjacent_score", "severe_l2h", "h2l_guard")


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def vector_sha256(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(compact_json(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")
    os.replace(temporary, path)


def _selection_hash(
    row: dict[str, Any],
    *,
    namespace: str,
) -> str:
    payload = "|".join(
        [
            namespace,
            str(row["record_id"]),
            str(row["pair_type"]),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["record_id"]), str(row["pair_type"])


def _index(rows: list[dict[str, Any]], *, label: str) -> dict[tuple[str, str], dict[str, Any]]:
    output = {_key(row): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label}: duplicate record/block key")
    return output


def _selected_score_keys(
    p1_rows: list[dict[str, Any]],
    plan: dict[str, Any],
) -> list[tuple[str, str]]:
    namespace = str(plan["selection"]["namespace"])
    quotas = {
        str(name): int(count)
        for name, count in plan["selection"]["score_block_quotas"].items()
    }
    if set(quotas) != set(SCORE_TYPES):
        raise ValueError("smoke score quotas do not cover the exact blocks")
    output = []
    for pair_type in SCORE_TYPES:
        candidates = [
            row for row in p1_rows if str(row["pair_type"]) == pair_type
        ]
        candidates.sort(
            key=lambda row: (
                _selection_hash(row, namespace=namespace),
                _key(row),
            )
        )
        if len(candidates) < quotas[pair_type]:
            raise ValueError(f"{pair_type}: insufficient smoke candidates")
        output.extend(_key(row) for row in candidates[: quotas[pair_type]])
    if len(output) != len(set(output)):
        raise ValueError("selected smoke score keys are not unique")
    return output


def _select_rationale(
    p3_rows: list[dict[str, Any]],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    namespace = str(plan["selection"]["namespace"])
    quota = int(plan["selection"]["rationale_alignment_quota_for_p3"])
    candidates = [
        row
        for row in p3_rows
        if str(row["pair_task"]) == "rationale"
        and str(row["pair_type"]) == "rationale_alignment"
    ]
    candidates.sort(
        key=lambda row: (
            _selection_hash(row, namespace=namespace),
            _key(row),
        )
    )
    if len(candidates) < quota:
        raise ValueError("insufficient rationale smoke candidates")
    return candidates[:quota]


def _arm_report(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    cutoff_len: int,
) -> dict[str, Any]:
    task_counts = Counter(str(row["pair_task"]) for row in rows)
    type_counts = Counter(str(row["pair_type"]) for row in rows)
    return {
        "pair_count": len(rows),
        "pair_task_counts": dict(sorted(task_counts.items())),
        "pair_type_counts": dict(sorted(type_counts.items())),
        "optimizer_steps": 1,
        "accumulation_group_pair_count": len(rows),
        "final_group_pair_count": len(rows),
        "chosen_unpadded_tokens": sum(
            len(row["chosen_input_ids"]) for row in rows
        ),
        "rejected_unpadded_tokens": sum(
            len(row["rejected_input_ids"]) for row in rows
        ),
        "fixed_padded_forward_tokens": 2 * len(rows) * cutoff_len,
        "manifest_sha256": sha256_file(path),
        "record_block_vector_sha256": vector_sha256(
            _key(row) for row in rows
        ),
        "pair_id_vector_sha256": vector_sha256(
            str(row["pair_id"]) for row in rows
        ),
        "chosen_sequence_vector_sha256": vector_sha256(
            str(row["chosen_input_ids_sha256"]) for row in rows
        ),
        "chosen_field_mask_vector_sha256": vector_sha256(
            str(row["chosen_field_token_positions_sha256"]) for row in rows
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-training-lock",
        type=Path,
        default=DEFAULT_FROZEN_LOCK,
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.output_dir / "smoke_package_report.json"
    lock_path = args.output_dir / "smoke_package_lock.json"
    if report_path.exists() or lock_path.exists():
        raise FileExistsError("SORC-DPO smoke package already exists")
    for path in (
        args.frozen_training_lock,
        args.plan,
        *SOURCE_MANIFESTS.values(),
    ):
        reject_eval_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = read_json(args.frozen_training_lock)
    plan = read_json(args.plan)
    if (
        frozen.get("status")
        != "SORC_DPO_TRAINING_FROZEN_SMOKE_PACKAGE_BUILD_ALLOWED"
        or frozen.get("smoke_subset_build_allowed") is not True
        or frozen.get("gpu_smoke_allowed") is not False
    ):
        raise PermissionError("training freeze does not allow smoke build")
    if (
        plan.get("status")
        != "DETERMINISTIC_TRAIN_ONLY_SMOKE_PACKAGE_BUILD_ALLOWED_EXECUTION_FORBIDDEN"
        or plan["source_split"] != "train"
        or plan["execution_contract"]["gpu_smoke_allowed"] is not False
    ):
        raise PermissionError("smoke plan is not fail-closed")

    source_rows = {}
    for arm, path in SOURCE_MANIFESTS.items():
        expected_hash = str(frozen["private_manifest_hashes"][arm])
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{arm}: frozen source manifest differs")
        source_rows[arm] = read_jsonl(path)
    selected_keys = _selected_score_keys(
        source_rows["P1_FIELD_DPO"],
        plan,
    )
    p1_index = _index(source_rows["P1_FIELD_DPO"], label="P1")
    p2_index = _index(source_rows["P2_SORC_SCORE"], label="P2")
    syn_index = _index(source_rows["P1_SYN_SEED42"], label="P1-SYN")
    p3_score_index = _index(
        [
            row
            for row in source_rows["P3_JOINT_SORC"]
            if str(row["pair_task"]) == "score"
        ],
        label="P3 score",
    )
    for label, index in (
        ("P1", p1_index),
        ("P2", p2_index),
        ("P1-SYN", syn_index),
        ("P3 score", p3_score_index),
    ):
        if any(key not in index for key in selected_keys):
            raise ValueError(f"{label}: selected smoke key is missing")

    subsets = {
        "P1_FIELD_DPO": [p1_index[key] for key in selected_keys],
        "P2_SORC_SCORE": [p2_index[key] for key in selected_keys],
        "P1_SYN_SEED42": [syn_index[key] for key in selected_keys],
        "P3_JOINT_SORC": [
            *[p3_score_index[key] for key in selected_keys],
            *_select_rationale(source_rows["P3_JOINT_SORC"], plan),
        ],
    }
    if [
        _key(row) for row in subsets["P1_FIELD_DPO"]
    ] != [
        _key(row) for row in subsets["P2_SORC_SCORE"]
    ] or [
        _key(row) for row in subsets["P1_FIELD_DPO"]
    ] != [
        _key(row) for row in subsets["P1_SYN_SEED42"]
    ]:
        raise AssertionError("score smoke record/block vectors differ")
    if [
        row["chosen_input_ids_sha256"]
        for row in subsets["P1_FIELD_DPO"]
    ] != [
        row["chosen_input_ids_sha256"]
        for row in subsets["P1_SYN_SEED42"]
    ]:
        raise AssertionError("P1/P1-SYN smoke chosen sequences differ")
    if [
        row["chosen_field_token_positions_sha256"]
        for row in subsets["P1_FIELD_DPO"]
    ] != [
        row["chosen_field_token_positions_sha256"]
        for row in subsets["P1_SYN_SEED42"]
    ]:
        raise AssertionError("P1/P1-SYN smoke chosen masks differ")

    private_paths = {}
    for arm, rows in subsets.items():
        expected = plan["arms"][arm]
        if (
            len(rows) != int(expected["pair_count"])
            or sum(row["pair_task"] == "score" for row in rows)
            != int(expected["score_pairs"])
            or sum(row["pair_task"] == "rationale" for row in rows)
            != int(expected["rationale_pairs"])
        ):
            raise ValueError(f"{arm}: smoke subset count differs from plan")
        path = args.output_dir / "private" / f"{arm.lower()}.jsonl"
        write_jsonl(path, rows)
        private_paths[arm] = path

    cutoff_len = int(plan["execution_contract"]["fixed_right_padding_to"])
    arm_reports = {
        arm: _arm_report(rows, private_paths[arm], cutoff_len=cutoff_len)
        for arm, rows in subsets.items()
    }
    report = {
        "schema_version": "exp54-sorc-dpo-smoke-package-report-v1",
        "status": "SORC_DPO_SMOKE_PACKAGE_CANDIDATE_EXECUTION_FORBIDDEN",
        "source_split": "train",
        "source_training_seed": 42,
        "arms": arm_reports,
        "p1_p2_p1_syn_record_block_vectors_equal": True,
        "p1_p1_syn_chosen_sequence_and_mask_vectors_equal": True,
        "p3_score_smoke_vector_equals_p2": True,
        "optimizer_steps_per_arm": 1,
        "model_loaded": False,
        "forward_backward_executed": False,
        "gpu_smoke_allowed": False,
        "formal_preference_training_allowed": False,
        "evaluator_called": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    write_json(report_path, report)
    lock = {
        "schema_version": "exp54-sorc-dpo-smoke-package-lock-v1",
        "status": "SORC_DPO_SMOKE_PACKAGE_CANDIDATE_NOT_FROZEN_EXECUTION_FORBIDDEN",
        "smoke_package_report_sha256": sha256_file(report_path),
        "private_subset_hashes": {
            arm: sha256_file(path) for arm, path in private_paths.items()
        },
        "source_manifest_hashes": {
            arm: sha256_file(path) for arm, path in SOURCE_MANIFESTS.items()
        },
        "source_hashes": {
            "frozen_training_lock": sha256_file(args.frozen_training_lock),
            "smoke_plan": sha256_file(args.plan),
            "smoke_builder": sha256_file(Path(__file__)),
        },
        "source_split": "train",
        "subset_frozen": False,
        "gpu_smoke_allowed": False,
        "formal_preference_training_allowed": False,
        "evaluator_calls_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    write_json(lock_path, lock)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
