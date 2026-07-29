"""Independently audit the deterministic train-only SORC-DPO smoke package."""

from __future__ import annotations

import argparse
import ast
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
SMOKE_ROOT = RAR_ROOT / "preference_smoke"
DEFAULT_FROZEN_LOCK = TRAINING_ROOT / "preference_training_frozen_lock.json"
DEFAULT_PLAN = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/sorc_dpo_smoke_plan_v1.json"
)
DEFAULT_LOCK = SMOKE_ROOT / "smoke_package_lock.json"
DEFAULT_REPORT = SMOKE_ROOT / "smoke_package_report.json"
DEFAULT_OUTPUT = SMOKE_ROOT / "smoke_package_audit_report.json"
RUNNER = REPO_ROOT / "thesis_exp/exp54_rar_sft/train_sorc_dpo_smoke.py"
SOURCE_MANIFESTS = {
    "P1_FIELD_DPO": TRAINING_ROOT / "private/p1_field_dpo.jsonl",
    "P2_SORC_SCORE": TRAINING_ROOT / "private/p2_sorc_score.jsonl",
    "P3_JOINT_SORC": TRAINING_ROOT / "private/p3_joint_sorc.jsonl",
    "P1_SYN_SEED42": TRAINING_ROOT / "private/p1_syn_seed42.jsonl",
}
SUBSET_FILES = {
    arm: SMOKE_ROOT / "private" / f"{arm.lower()}.jsonl"
    for arm in SOURCE_MANIFESTS
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
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["record_id"]), str(row["pair_type"])


def _stable_hash(row: dict[str, Any], namespace: str) -> str:
    return hashlib.sha256(
        "|".join(
            [namespace, str(row["record_id"]), str(row["pair_type"])]
        ).encode("utf-8")
    ).hexdigest()


def _index(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    output = {_key(row): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{label}: duplicate record/block key")
    return output


def _derive_expected(
    source: dict[str, list[dict[str, Any]]],
    plan: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    namespace = str(plan["selection"]["namespace"])
    quotas = {
        str(name): int(count)
        for name, count in plan["selection"]["score_block_quotas"].items()
    }
    selected_keys = []
    for pair_type in SCORE_TYPES:
        rows = [
            row
            for row in source["P1_FIELD_DPO"]
            if str(row["pair_type"]) == pair_type
        ]
        rows.sort(key=lambda row: (_stable_hash(row, namespace), _key(row)))
        selected_keys.extend(
            _key(row) for row in rows[: quotas[pair_type]]
        )
    indexes = {
        "P1_FIELD_DPO": _index(source["P1_FIELD_DPO"], label="P1"),
        "P2_SORC_SCORE": _index(source["P2_SORC_SCORE"], label="P2"),
        "P1_SYN_SEED42": _index(
            source["P1_SYN_SEED42"],
            label="P1-SYN",
        ),
        "P3_JOINT_SORC": _index(
            [
                row
                for row in source["P3_JOINT_SORC"]
                if str(row["pair_task"]) == "score"
            ],
            label="P3 score",
        ),
    }
    rationale = [
        row
        for row in source["P3_JOINT_SORC"]
        if str(row["pair_task"]) == "rationale"
    ]
    rationale.sort(
        key=lambda row: (_stable_hash(row, namespace), _key(row))
    )
    rationale_quota = int(
        plan["selection"]["rationale_alignment_quota_for_p3"]
    )
    return {
        "P1_FIELD_DPO": [
            indexes["P1_FIELD_DPO"][key] for key in selected_keys
        ],
        "P2_SORC_SCORE": [
            indexes["P2_SORC_SCORE"][key] for key in selected_keys
        ],
        "P1_SYN_SEED42": [
            indexes["P1_SYN_SEED42"][key] for key in selected_keys
        ],
        "P3_JOINT_SORC": [
            *[
                indexes["P3_JOINT_SORC"][key]
                for key in selected_keys
            ],
            *rationale[:rationale_quota],
        ],
    }


def _assert_runner_fail_closed() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            if any(name == "torch" or name.startswith("transformers") or name.startswith("peft") for name in names):
                raise ValueError("runner imports GPU/model runtime at module scope")
    main_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    if len(main_functions) != 1:
        raise ValueError("runner main function is missing or duplicated")
    calls = [
        node.func.id
        for node in ast.walk(main_functions[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    if "verify_gpu_execution_authorization" not in calls:
        raise ValueError("runner does not verify GPU authorization")
    if "execute_one_smoke_step" not in calls:
        raise ValueError("runner execution entry is missing")
    if calls.index("verify_gpu_execution_authorization") > calls.index(
        "execute_one_smoke_step"
    ):
        raise ValueError("runner executes before authorization verification")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-lock", type=Path, default=DEFAULT_FROZEN_LOCK)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--smoke-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--smoke-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = (
        args.frozen_lock,
        args.plan,
        args.smoke_lock,
        args.smoke_report,
        RUNNER,
        *SOURCE_MANIFESTS.values(),
        *SUBSET_FILES.values(),
    )
    for path in paths:
        reject_eval_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = read_json(args.frozen_lock)
    plan = read_json(args.plan)
    lock = read_json(args.smoke_lock)
    report = read_json(args.smoke_report)
    if (
        frozen.get("status")
        != "SORC_DPO_TRAINING_FROZEN_SMOKE_PACKAGE_BUILD_ALLOWED"
        or lock.get("status")
        != "SORC_DPO_SMOKE_PACKAGE_CANDIDATE_NOT_FROZEN_EXECUTION_FORBIDDEN"
        or report.get("status")
        != "SORC_DPO_SMOKE_PACKAGE_CANDIDATE_EXECUTION_FORBIDDEN"
    ):
        raise ValueError("smoke package state differs")
    if sha256_file(args.smoke_report) != str(
        lock["smoke_package_report_sha256"]
    ):
        raise ValueError("smoke report hash differs")
    if sha256_file(args.frozen_lock) != str(
        lock["source_hashes"]["frozen_training_lock"]
    ):
        raise ValueError("frozen training lock hash differs")
    if sha256_file(args.plan) != str(lock["source_hashes"]["smoke_plan"]):
        raise ValueError("smoke plan hash differs")

    source = {}
    actual = {}
    for arm in SOURCE_MANIFESTS:
        if sha256_file(SOURCE_MANIFESTS[arm]) != str(
            lock["source_manifest_hashes"][arm]
        ):
            raise ValueError(f"{arm}: source manifest hash differs")
        if sha256_file(SUBSET_FILES[arm]) != str(
            lock["private_subset_hashes"][arm]
        ):
            raise ValueError(f"{arm}: private subset hash differs")
        source[arm] = read_jsonl(SOURCE_MANIFESTS[arm])
        actual[arm] = read_jsonl(SUBSET_FILES[arm])
    expected = _derive_expected(source, plan)
    cutoff_len = int(plan["execution_contract"]["fixed_right_padding_to"])
    aggregate = {}
    for arm in SOURCE_MANIFESTS:
        if actual[arm] != expected[arm]:
            raise ValueError(f"{arm}: deterministic subset content differs")
        tasks = Counter(str(row["pair_task"]) for row in actual[arm])
        types = Counter(str(row["pair_type"]) for row in actual[arm])
        public = report["arms"][arm]
        derived_public = {
            "pair_count": len(actual[arm]),
            "pair_task_counts": dict(sorted(tasks.items())),
            "pair_type_counts": dict(sorted(types.items())),
            "optimizer_steps": 1,
            "accumulation_group_pair_count": len(actual[arm]),
            "final_group_pair_count": len(actual[arm]),
            "chosen_unpadded_tokens": sum(
                len(row["chosen_input_ids"]) for row in actual[arm]
            ),
            "rejected_unpadded_tokens": sum(
                len(row["rejected_input_ids"]) for row in actual[arm]
            ),
            "fixed_padded_forward_tokens": (
                2 * len(actual[arm]) * cutoff_len
            ),
            "manifest_sha256": sha256_file(SUBSET_FILES[arm]),
            "record_block_vector_sha256": vector_sha256(
                _key(row) for row in actual[arm]
            ),
            "pair_id_vector_sha256": vector_sha256(
                str(row["pair_id"]) for row in actual[arm]
            ),
            "chosen_sequence_vector_sha256": vector_sha256(
                str(row["chosen_input_ids_sha256"]) for row in actual[arm]
            ),
            "chosen_field_mask_vector_sha256": vector_sha256(
                str(row["chosen_field_token_positions_sha256"])
                for row in actual[arm]
            ),
        }
        for field, value in derived_public.items():
            if public.get(field) != value:
                raise ValueError(f"{arm}: public {field} differs")
        aggregate[arm] = {
            "pairs_verified": len(actual[arm]),
            "deterministic_selection_mismatches": 0,
            "public_budget_mismatches": 0,
        }
    _assert_runner_fail_closed()
    for value in (frozen, plan["execution_contract"], lock, report):
        if (
            value.get("gpu_smoke_allowed") is not False
            or value.get("formal_preference_training_allowed") is not False
            or value.get("dev_accessed") is not False
            or value.get("test_accessed") is not False
        ):
            raise PermissionError("smoke package authorization boundary differs")
    audit = {
        "schema_version": "exp54-sorc-dpo-smoke-package-audit-v1",
        "status": "SORC_DPO_SMOKE_PACKAGE_CANDIDATE_AUDIT_PASS",
        "aggregate": aggregate,
        "selection_independently_rederived": True,
        "runner_authorization_precedes_execution": True,
        "runner_has_no_top_level_gpu_or_model_import": True,
        "private_subset_hash_vector_sha256": vector_sha256(
            lock["private_subset_hashes"][arm] for arm in SOURCE_MANIFESTS
        ),
        "smoke_package_report_sha256": sha256_file(args.smoke_report),
        "smoke_package_lock_sha256": sha256_file(args.smoke_lock),
        "runner_source_sha256": sha256_file(RUNNER),
        "auditor_source_sha256": sha256_file(Path(__file__)),
        "gpu_smoke_allowed": False,
        "formal_preference_training_allowed": False,
        "evaluator_called": False,
        "dev_accessed": False,
        "test_accessed": False,
        "training_used": False,
    }
    write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
