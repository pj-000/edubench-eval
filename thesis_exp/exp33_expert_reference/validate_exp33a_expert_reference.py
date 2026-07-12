#!/usr/bin/env python3
"""Validate Exp33A sampling, blind packets, schemas, audits, and commit scope.

Validation reconstructs every packet from permitted train/dev source fields and
re-solves the deterministic sampling plans. It never reads the paper test split.
Use ``--heavy --staged`` after explicitly staging only lightweight public files
to enforce the final commit boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp33_expert_reference.build_exp33a_private_source_reference import (  # noqa: E402
    DEFAULT_EXP28E_DECISION,
    DEFAULT_TEACHER_SUMMARY_DIR,
    read_jsonl,
    resolve_teacher_inputs,
)
from thesis_exp.exp33_expert_reference.prepare_exp33a_expert_reference import (  # noqa: E402
    CLEAN_DEV_QUOTAS,
    PACKET_FIELDS,
    REPRESENTATIVE_QUOTAS,
    RISK_NONLOW_QUOTAS,
    canonical_hash,
    packet_for,
    qkey,
    sample_id,
    select_representative,
    select_risk_enriched,
    solve_clean_dev_milp,
    teacher_annotation_map,
    validate_source_boundaries,
)


DEFAULT_SPLIT_DIR = Path("thesis_exp/data/splits/paper_like_triple_seed42")
DEFAULT_PROCESSED = Path("thesis_exp/data/processed/edubench_scoring_all.jsonl")
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp33_expert_reference/outputs/exp33a_expert_reference_seed42"
)
DEFAULT_REVIEW_SCHEMA = Path(
    "thesis_exp/exp33_expert_reference/schemas/exp33a_blind_review_schema.json"
)
DEFAULT_ADJUDICATION_SCHEMA = Path(
    "thesis_exp/exp33_expert_reference/schemas/exp33a_adjudication_schema.json"
)

REQUIRED_PUBLIC = (
    "configs/exp33a_sampling_lock.json",
    "configs/exp33a_review_protocol_lock.json",
    "tables/exp33a_resolved_teacher_input_manifest.csv",
    "tables/exp33a_train_representative_distribution.csv",
    "tables/exp33a_train_risk_distribution.csv",
    "tables/exp33a_clean_dev_distribution.csv",
    "tables/exp33a_question_key_partition.csv",
    "tables/exp33a_sampling_design_weights.csv",
    "tables/exp33a_view_overlap_audit.csv",
    "tables/exp33a_blind_leakage_audit.csv",
    "tables/exp33a_review_completion.csv",
    "tables/exp33a_reviewer_agreement.csv",
    "tables/exp33a_domain_escalation_summary.csv",
    "reports/exp33a_expert_reference_prepare_report.md",
    "decision/exp33a_expert_reference_decision.json",
)
REQUIRED_PRIVATE = (
    "private/exp33a_selected_sample_manifest.jsonl",
    "private/exp33a_source_reference.jsonl",
    "private_review/exp33a_review_assignment_manifest.jsonl",
    "private_review/blind_packets/exp33a_reviewer_a_packet.jsonl",
    "private_review/blind_packets/exp33a_reviewer_b_packet.jsonl",
)


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def guarded(path: Path) -> Path:
    absolute = repo_path(path)
    if absolute.name.casefold() == "test.jsonl":
        raise PermissionError("Exp33A forbids access to the sealed paper test split")
    return absolute


def read_json(path: Path) -> dict[str, Any]:
    with guarded(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with guarded(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.rstrip() + "\n", encoding="utf-8")


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.rows.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    def run(self, name: str, function: Callable[[], tuple[bool, str]]) -> None:
        try:
            passed, detail = function()
        except Exception as exc:  # validation must report all failures together
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        self.add(name, bool(passed), detail)

    @property
    def passed(self) -> bool:
        return all(row["status"] == "PASS" for row in self.rows)


def check_files(out: Path, relative_paths: tuple[str, ...]) -> tuple[bool, str]:
    missing = [relative for relative in relative_paths if not guarded(out / relative).is_file()]
    return not missing, f"missing={missing or 'none'}"


def check_gitignored(out: Path) -> tuple[bool, str]:
    failures = []
    for relative in REQUIRED_PRIVATE:
        path = guarded(out / relative)
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(path)],
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode != 0:
            failures.append(relative)
    return not failures, f"not_ignored={failures or 'none'}"


def check_sampling(
    selections: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    teacher_resolved: dict[str, dict[str, Any]],
) -> tuple[bool, str, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_view: dict[str, list[dict[str, Any]]] = {}
    train_map = {sample_id(row): row for row in train_rows}
    dev_map = {sample_id(row): row for row in dev_rows}
    for view in ("representative_train", "risk_enriched_train", "clean_dev"):
        records = [row for row in selections if row.get("view") == view]
        source_map = dev_map if view == "clean_dev" else train_map
        by_view[view] = [source_map[str(row["sample_id"])] for row in records if str(row.get("sample_id")) in source_map]

    ids = [str(row.get("sample_id")) for row in selections]
    failures: list[str] = []
    if len(selections) != 420 or len(ids) != len(set(ids)):
        failures.append("420_unique_sample_ids")
    expected_sizes = {"representative_train": 120, "risk_enriched_train": 120, "clean_dev": 180}
    for view, expected in expected_sizes.items():
        if len(by_view[view]) != expected:
            failures.append(f"{view}_rows")
    rep_counts = Counter(int(row["label_5"]) for row in by_view["representative_train"])
    clean_counts = Counter(int(row["label_5"]) for row in by_view["clean_dev"])
    risk_counts = Counter(int(row["label_5"]) for row in by_view["risk_enriched_train"])
    expected_risk = Counter(
        {
            1: Counter(int(row["label_5"]) for row in train_rows)[1] - REPRESENTATIVE_QUOTAS[1],
            2: Counter(int(row["label_5"]) for row in train_rows)[2] - REPRESENTATIVE_QUOTAS[2],
            **RISK_NONLOW_QUOTAS,
        }
    )
    if rep_counts != Counter(REPRESENTATIVE_QUOTAS):
        failures.append(f"representative_quotas={dict(rep_counts)}")
    if clean_counts != Counter(CLEAN_DEV_QUOTAS):
        failures.append(f"clean_dev_quotas={dict(clean_counts)}")
    if risk_counts != expected_risk:
        failures.append(f"risk_quotas={dict(risk_counts)}")

    reconstructed_clean, clean_solver = solve_clean_dev_milp(dev_rows, CLEAN_DEV_QUOTAS, 42)
    reconstructed_rep = select_representative(train_rows, REPRESENTATIVE_QUOTAS, 42)
    primary = teacher_annotation_map(teacher_resolved["primary"]["rows"])
    secondary = teacher_annotation_map(teacher_resolved["secondary"]["rows"])
    reconstructed_risk, _ = select_risk_enriched(train_rows, reconstructed_rep, primary, secondary, 42)
    for view, reconstructed in (
        ("clean_dev", reconstructed_clean),
        ("representative_train", reconstructed_rep),
        ("risk_enriched_train", reconstructed_risk),
    ):
        if {sample_id(row) for row in by_view[view]} != {sample_id(row) for row in reconstructed}:
            failures.append(f"{view}_not_deterministically_reproduced")

    train_label_pop = Counter(int(row["label_5"]) for row in train_rows)
    for selected in selections:
        if selected["view"] != "representative_train":
            if selected.get("design_weight") is not None or selected.get("inclusion_probability") is not None:
                failures.append("nonrepresentative_design_weight_present")
                break
            continue
        label = int(train_map[str(selected["sample_id"])]["label_5"])
        expected_probability = REPRESENTATIVE_QUOTAS[label] / train_label_pop[label]
        expected_weight = train_label_pop[label] / REPRESENTATIVE_QUOTAS[label]
        if not math_close(float(selected["inclusion_probability"]), expected_probability) or not math_close(float(selected["design_weight"]), expected_weight):
            failures.append(f"representative_weight_label_{label}")
            break
    return not failures, f"failures={failures or 'none'}; clean_dev_max_qkeys={clean_solver['maximum_unique_question_keys']}", by_view, clean_solver


def math_close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def check_packets(
    out: Path,
    selections: list[dict[str, Any]],
    by_view: dict[str, list[dict[str, Any]]],
) -> tuple[bool, str, list[dict[str, Any]]]:
    packet_a = read_jsonl(out / "private_review/blind_packets/exp33a_reviewer_a_packet.jsonl")
    packet_b = read_jsonl(out / "private_review/blind_packets/exp33a_reviewer_b_packet.jsonl")
    failures: list[str] = []
    if packet_a != packet_b or len(packet_a) != 420:
        failures.append("reviewer_packet_copies_or_count")
    source_by_id = {sample_id(row): row for rows in by_view.values() for row in rows}
    expected_order = [str(row["sample_id"]) for row in selections]
    if [str(row.get("sample_id")) for row in packet_a] != expected_order:
        failures.append("packet_assignment_order")
    for index, packet in enumerate(packet_a):
        sid = str(packet.get("sample_id") or "")
        if set(packet) != set(PACKET_FIELDS):
            failures.append(f"packet_field_allowlist_row_{index}")
            break
        source = source_by_id.get(sid)
        if source is None or packet != packet_for(source, 42):
            failures.append(f"packet_source_projection_row_{index}")
            break
        packet_without_hash = dict(packet)
        actual_hash = packet_without_hash.pop("packet_hash")
        if actual_hash != canonical_hash(packet_without_hash):
            failures.append(f"packet_hash_row_{index}")
            break
    return not failures, f"failures={failures or 'none'}; rows={len(packet_a)}", packet_a


def check_schemas(review_schema: dict[str, Any], adjudication_schema: dict[str, Any]) -> tuple[bool, str]:
    jsonschema.Draft202012Validator.check_schema(review_schema)
    jsonschema.Draft202012Validator.check_schema(adjudication_schema)
    model_review = {
        "sample_id": "synthetic",
        "reviewer_role": "reviewer_a",
        "reviewer_type": "model",
        "reviewer_provenance": "independent_model_reviewer",
        "reviewer_provider": "synthetic-provider",
        "reviewer_model_id": "synthetic-model-version",
        "reviewer_run_id": "synthetic-run-a",
        "target_scope_confirmed": True,
        "score_range": [2, 3],
        "most_plausible_score": 3,
        "failure_bucket": "visible_failure",
        "major_failures": ["synthetic failure"],
        "rubric_evidence": "synthetic rubric evidence",
        "evaluator_output_evidence": "synthetic output",
        "missing_evidence_reason": None,
        "score_cap": 3,
        "student_input_sufficiency": "sufficient",
        "confidence": "medium",
        "domain_uncertainty": False,
        "domain_escalation_required": False,
        "needs_adjudication": True,
        "review_reason": "synthetic concise reason"
    }
    human_review = dict(model_review)
    human_review.update(
        {
            "reviewer_type": "human",
            "reviewer_provenance": "independent_human_reviewer",
            "reviewer_provider": None,
            "reviewer_model_id": None,
        }
    )
    jsonschema.validate(model_review, review_schema)
    jsonschema.validate(human_review, review_schema)
    invalid_model = dict(model_review)
    invalid_model["reviewer_model_id"] = None
    if not list(jsonschema.Draft202012Validator(review_schema).iter_errors(invalid_model)):
        return False, "model review incorrectly accepts null reviewer_model_id"
    adjudication = {
        "sample_id": "synthetic",
        "reviewer_role": "adjudicator",
        "reviewer_type": "model",
        "reviewer_provenance": "independent_model_adjudicator",
        "reviewer_provider": "synthetic-provider",
        "reviewer_model_id": "synthetic-model-version",
        "reviewer_run_id": "synthetic-run-adj",
        "reviewer_a_result_hash": "a" * 64,
        "reviewer_b_result_hash": "b" * 64,
        "source_comparison_frozen_hash": "c" * 64,
        "source_provenance_seen": ["human_1", "human_2", "human_3", "rounded_human", "qwen", "deepseek"],
        "target_scope_confirmed": True,
        "final_score_range": [2, 3],
        "final_most_plausible_score": 3,
        "final_score_posterior": {"1": 0.0, "2": 0.25, "3": 0.75, "4": 0.0, "5": 0.0},
        "final_failure_bucket": "visible_failure",
        "student_input_sufficiency": "sufficient",
        "confidence": "medium",
        "domain_escalation_required": False,
        "final_status": "model_reviewed_silver",
        "correction_action": "soft_posterior",
        "rubric_evidence": "synthetic rubric evidence",
        "evaluator_output_evidence": "synthetic output",
        "adjudication_reason": "synthetic adjudication reason"
    }
    jsonschema.validate(adjudication, adjudication_schema)
    return True, "draft-2020-12 schemas valid for model/human provenance and staged adjudication"


def check_protocol(
    out: Path,
    sampling_lock: dict[str, Any],
    review_lock: dict[str, Any],
    decision: dict[str, Any],
    source_audit: dict[str, Any],
    clean_solver: dict[str, Any],
) -> tuple[bool, str]:
    failures = []
    paper = sampling_lock.get("paper_protocol") or {}
    expected = {
        "train_dev_triple_key_disjoint": True,
        "train_dev_question_key_disjoint": False,
        "future_train_question_key_exclusion": False,
        "future_train_rows_removed": 0,
        "future_train_rows_retained": 2654,
        "test_access_count": 0,
    }
    for key, value in expected.items():
        if paper.get(key) != value:
            failures.append(f"paper_protocol.{key}")
    if source_audit["train_dev_question_key_overlap"] != 184:
        failures.append("train_dev_question_key_overlap_not_184")
    if source_audit["train_rows_on_train_dev_shared_question_keys"] != 2562:
        failures.append("shared_qkey_train_rows_not_2562")
    clean_lock = ((sampling_lock.get("views") or {}).get("clean_dev") or {})
    if clean_lock.get("maximum_unique_question_keys") != clean_solver["maximum_unique_question_keys"]:
        failures.append("clean_dev_maximum_qkeys_lock")
    if review_lock.get("locked_reviewer_type") != "model" or review_lock.get("default_reviewer_type") != "model":
        failures.append("current_reviewer_type_not_model")
    if not review_lock.get("provider_agnostic_method"):
        failures.append("provider_agnostic_method_missing")
    if review_lock.get("planned_model_family") != "GPT-5.6":
        failures.append("planned_model_provenance_missing")
    for key in (
        "model_silver_reference_complete", "expert_reference_complete", "teacher_reliability_ready",
        "recommend_student_training", "recommend_test_access",
    ):
        if decision.get(key) is not False:
            failures.append(f"decision.{key}_must_be_false_before_reviews")
    if decision.get("future_train_rows_removed") != 0 or decision.get("future_train_rows_retained") != 2654:
        failures.append("decision_train_retention")
    if any((repo_path(out) / relative).exists() for relative in (
        "private/exp33a_future_training_exclusion_qkeys.json",
        "tables/exp33a_question_key_exclusion_pareto.csv",
    )):
        failures.append("revoked_qkey_exclusion_artifact_present")
    return not failures, f"failures={failures or 'none'}"


def check_public_privacy(out: Path, selected_ids: set[str], selected_qkeys: set[str]) -> tuple[bool, str]:
    failures: list[str] = []
    for subdir in ("configs", "tables", "reports", "decision"):
        directory = guarded(out / subdir)
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(value in text for value in selected_ids):
                failures.append(f"sample_id:{path.relative_to(repo_path(out))}")
            if any(value in text for value in selected_qkeys):
                failures.append(f"question_key:{path.relative_to(repo_path(out))}")
            lowered = text.casefold()
            if "human expert gold" in lowered or "human-expert gold" in lowered:
                failures.append(f"misclaimed_reference:{path.relative_to(repo_path(out))}")
    return not failures, f"failures={failures or 'none'}"


def check_distribution_dimensions(out: Path) -> tuple[bool, str]:
    failures = []
    for relative in (
        "tables/exp33a_train_representative_distribution.csv",
        "tables/exp33a_train_risk_distribution.csv",
        "tables/exp33a_clean_dev_distribution.csv",
    ):
        dimensions = {row["dimension"] for row in read_csv(out / relative)}
        required = {"label", "language", "metric_family", "metric", "subject"}
        if not required <= dimensions:
            failures.append(f"{relative}:{sorted(required - dimensions)}")
    return not failures, f"missing_dimensions={failures or 'none'}"


def staged_heavy_check(out: Path, selected_ids: set[str], selected_qkeys: set[str]) -> tuple[bool, str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = [Path(value.decode("utf-8")) for value in result.stdout.split(b"\0") if value]
    if not paths:
        return False, "no staged files; stage the explicit lightweight allowlist first"
    forbidden_parts = {
        "private", "private_review", "blind_packets", "reviewer_a_filled",
        "reviewer_b_filled", "adjudication_filled", "source_reference",
    }
    forbidden_suffixes = {
        ".jsonl", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".bin",
        ".safetensors", ".log",
    }
    failures: list[str] = []
    for relative in paths:
        path = REPO_ROOT / relative
        if forbidden_parts & set(relative.parts):
            failures.append(f"private_path:{relative}")
        if relative.suffix.casefold() in forbidden_suffixes:
            failures.append(f"forbidden_suffix:{relative}")
        if path.is_file() and path.stat().st_size > 1024 * 1024:
            failures.append(f"over_1MiB:{relative}")
        if path.is_file() and relative.suffix.casefold() in {".py", ".md", ".json", ".csv", ".sh", ""}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(value in text for value in selected_ids):
                failures.append(f"private_sample_id_content:{relative}")
            if any(value in text for value in selected_qkeys):
                failures.append(f"private_qkey_content:{relative}")
    return not failures, f"staged_files={len(paths)}; failures={failures or 'none'}"


def validate(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out_dir
    checks = Checks()
    checks.run("required_public_files", lambda: check_files(out, REQUIRED_PUBLIC))
    checks.run("required_private_files", lambda: check_files(out, REQUIRED_PRIVATE))
    checks.run("private_files_gitignored", lambda: check_gitignored(out))

    train_rows = read_jsonl(args.split_dir / "train.jsonl")
    dev_rows = read_jsonl(args.split_dir / "dev.jsonl")
    processed_rows = read_jsonl(args.processed_data)
    source_audit = validate_source_boundaries(train_rows, dev_rows, processed_rows)
    checks.add(
        "paper_train_dev_boundary",
        source_audit["train_dev_sample_overlap"] == 0 and source_audit["train_dev_triple_key_overlap"] == 0,
        json.dumps(source_audit, sort_keys=True),
    )
    teacher_manifest, teacher_resolved = resolve_teacher_inputs(args.teacher_summary_dir, args.exp28e_decision)
    public_manifest = read_csv(out / "tables/exp33a_resolved_teacher_input_manifest.csv")
    checks.add(
        "teacher_input_manifest",
        len(teacher_manifest) == 2 and all(
            str(expected[key]) == str(actual[key])
            for expected, actual in zip(teacher_manifest, public_manifest, strict=True)
            for key in ("teacher_role", "provider", "model", "annotation_path", "sha256", "row_count", "valid_row_count")
        ),
        "; ".join(f"{row['teacher_role']}={row['row_count']}:{row['sha256']}" for row in teacher_manifest),
    )

    selections = read_jsonl(out / "private/exp33a_selected_sample_manifest.jsonl")
    sampling_ok, sampling_detail, by_view, clean_solver = check_sampling(
        selections, train_rows, dev_rows, teacher_resolved
    )
    checks.add("deterministic_sampling_and_weights", sampling_ok, sampling_detail)
    packet_ok, packet_detail, packets = check_packets(out, selections, by_view)
    checks.add("blind_packets_and_hashes", packet_ok, packet_detail)

    review_schema = read_json(args.review_schema)
    adjudication_schema = read_json(args.adjudication_schema)
    checks.run("schemas_and_provenance", lambda: check_schemas(review_schema, adjudication_schema))
    sampling_lock = read_json(out / "configs/exp33a_sampling_lock.json")
    review_lock = read_json(out / "configs/exp33a_review_protocol_lock.json")
    decision = read_json(out / "decision/exp33a_expert_reference_decision.json")
    checks.run(
        "paper_protocol_and_pre_review_decision",
        lambda: check_protocol(out, sampling_lock, review_lock, decision, source_audit, clean_solver),
    )
    checks.run("public_distribution_dimensions", lambda: check_distribution_dimensions(out))
    selected_ids = {str(row["sample_id"]) for row in selections}
    source_rows = {sample_id(row): row for rows in by_view.values() for row in rows}
    selected_qkeys = {qkey(source_rows[sid]) for sid in selected_ids}
    checks.run("public_private_identity_leakage", lambda: check_public_privacy(out, selected_ids, selected_qkeys))

    leakage_types = (
        "human_label", "human_reason", "teacher_score", "teacher_reason",
        "student_prediction", "campaign_conflict_flag", "sampling_risk_reason",
        "b0_b4_variant", "train_dev_metric_result",
    )
    leakage_rows = [
        {
            "leakage_type": value,
            "count": 0 if packet_ok else 1,
            "status": "PASS" if packet_ok else "FAIL",
            "audit_method": "exact_packet_allowlist_plus_reconstruction_from_permitted_source_projection",
        }
        for value in leakage_types
    ]
    write_csv(
        out / "tables/exp33a_blind_leakage_audit.csv",
        leakage_rows,
        ["leakage_type", "count", "status", "audit_method"],
    )
    checks.add("blind_leakage_zero", packet_ok, "all nine forbidden leakage classes reconstructed as zero")
    checks.add("old_test_access_count_zero", True, "sealed test path was never passed to a reader; inherited locked test row count only")
    checks.add("no_api_gpu_training_inference", True, "validator is CPU data/schema audit only")
    if args.heavy:
        # Full private source-reference identity/coverage scan.
        source_reference = read_jsonl(out / "private/exp33a_source_reference.jsonl")
        checks.add(
            "heavy_private_source_reference_coverage",
            len(source_reference) == 420 and {str(row["sample_id"]) for row in source_reference} == selected_ids,
            f"rows={len(source_reference)}",
        )
    if args.staged:
        checks.run("staged_heavy_commit_boundary", lambda: staged_heavy_check(out, selected_ids, selected_qkeys))

    passed = checks.passed
    report_lines = [
        "# Exp33A Independent Model-Reviewed Silver Reference Validation",
        "",
        f"Overall status: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for row in checks.rows:
        detail = row["detail"].replace("|", "\\|").replace("\n", " ")
        report_lines.append(f"| {row['check']} | {row['status']} | {detail} |")
    report_lines.extend(
        [
            "",
            "The paper protocol remains triple-key-disjoint rather than question-key-disjoint. Question-key overlap is expected and removes zero train rows. The provider-agnostic method claim is blind-first source comparison, conflict adjudication, direction-aware correction, and uncertainty fallback. Actual reviewer provider/model IDs remain mandatory provenance.",
            "",
            "No reviewer result was fabricated. `model_silver_reference_complete=false`, `expert_reference_complete=false`, and `teacher_reliability_ready=false` remain locked before review. No API, GPU, training, student inference, or sealed-test access occurred.",
        ]
    )
    write_text(out / "reports/exp33a_expert_reference_validation_report.md", "\n".join(report_lines))
    decision.update(
        {
            "validation_complete": passed,
            "validation_status": "PASS" if passed else "FAIL",
            "validation_check_count": len(checks.rows),
            "validation_failure_count": sum(row["status"] == "FAIL" for row in checks.rows),
            "model_silver_reference_complete": False,
            "expert_reference_complete": False,
            "teacher_reliability_ready": False,
            "recommend_new_teacher_training": False,
            "recommend_student_training": False,
            "recommend_test_access": False,
            "test_access_count": 0,
        }
    )
    write_json(out / "decision/exp33a_expert_reference_decision.json", decision)
    summary = {
        "status": "PASS" if passed else "FAIL",
        "checks": len(checks.rows),
        "failures": [row["check"] for row in checks.rows if row["status"] == "FAIL"],
        "representative_train": len(by_view["representative_train"]),
        "risk_enriched_train": len(by_view["risk_enriched_train"]),
        "clean_dev": len(by_view["clean_dev"]),
        "clean_dev_unique_question_keys": clean_solver["maximum_unique_question_keys"],
        "packet_rows": len(packets),
        "test_access_count": 0,
    }
    if not passed:
        raise RuntimeError(json.dumps(summary, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--processed-data", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--teacher-summary-dir", type=Path, default=DEFAULT_TEACHER_SUMMARY_DIR)
    parser.add_argument("--exp28e-decision", type=Path, default=DEFAULT_EXP28E_DECISION)
    parser.add_argument("--review-schema", type=Path, default=DEFAULT_REVIEW_SCHEMA)
    parser.add_argument("--adjudication-schema", type=Path, default=DEFAULT_ADJUDICATION_SCHEMA)
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    if args.staged and not args.heavy:
        parser.error("--staged requires --heavy")
    return args


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), ensure_ascii=False, sort_keys=True))
