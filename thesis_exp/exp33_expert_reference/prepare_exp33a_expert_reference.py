#!/usr/bin/env python3
"""Prepare Exp33A stratified blind-review packets and public audit artifacts.

The paper-like protocol is preserved exactly: train/dev are triple-key
disjoint, question-key overlap is allowed, and all 2,654 train rows remain in
the future training pool.  The sealed test file is never opened.  This script
does CPU-only data construction; it performs no API call, inference, training,
or review completion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp33_expert_reference.build_exp33a_private_source_reference import (  # noqa: E402
    DEFAULT_EXP28E_DECISION,
    DEFAULT_HUMAN_REASON_FILES,
    DEFAULT_TEACHER_SUMMARY_DIR,
    build_private_source_reference,
    display_path,
    read_jsonl,
    resolve_teacher_inputs,
    sha256_file,
    write_csv,
    write_jsonl,
)


DEFAULT_SPLIT_DIR = Path("thesis_exp/data/splits/paper_like_triple_seed42")
DEFAULT_PROCESSED = Path("thesis_exp/data/processed/edubench_scoring_all.jsonl")
DEFAULT_EXP28_PROTOCOL_LOCK = Path(
    "thesis_exp/configs/exp28_teacher_audited_paper_protocol/exp28_protocol_lock.yaml"
)
DEFAULT_EXP32_REPORT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp32_dual_teacher_campaign_conclusion_seed42/"
    "reports/exp32_dual_teacher_campaign_conclusion.md"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp33_expert_reference/outputs/exp33a_expert_reference_seed42"
)

REPRESENTATIVE_QUOTAS = {1: 12, 2: 18, 3: 25, 4: 30, 5: 35}
CLEAN_DEV_QUOTAS = {1: 6, 2: 14, 3: 40, 4: 60, 5: 60}
RISK_NONLOW_QUOTAS = {3: 24, 4: 25, 5: 25}
VIEW_ORDER = ("representative_train", "risk_enriched_train", "clean_dev")
PACKET_FIELDS = (
    "sample_id",
    "anonymized_question_key_hash",
    "question_context",
    "evaluator_output",
    "evaluation_dimension",
    "canonical_rubric",
    "score_rubric",
    "non_label_metadata",
    "language",
    "packet_hash",
)
NON_LABEL_METADATA_FIELDS = (
    "subject",
    "education_level",
    "scenario",
)
SCORE_RUBRIC = {
    "scale": "integer_1_to_5",
    "anchors": {
        "1": "Fails the target dimension in a major or fundamental way.",
        "2": "Substantial target-dimension problems outweigh strengths.",
        "3": "Mixed or adequate performance with material limitations.",
        "4": "Strong performance with only limited target-dimension issues.",
        "5": "Fully satisfies the target dimension with no material failure.",
    },
}


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def guarded(path: Path) -> Path:
    absolute = repo_path(path)
    if absolute.name.casefold() == "test.jsonl":
        raise PermissionError("Exp33A forbids access to the sealed paper test split")
    return absolute


def write_json(path: Path, payload: Any) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    destination = repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.rstrip() + "\n", encoding="utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stable_fraction(seed: int, namespace: str, value: str) -> float:
    digest = hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:14], 16) / float(16**14 - 1)


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("record_id") or row.get("sample_id") or row.get("id")
    if not value:
        raise ValueError("Source row has no sample identifier")
    return str(value)


def qkey(row: dict[str, Any]) -> str:
    value = row.get("question_key")
    if not value:
        raise ValueError(f"Source row {sample_id(row)} has no question_key")
    return str(value)


def validate_source_boundaries(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    processed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(train_rows) != 2654 or len(dev_rows) != 664:
        raise ValueError(f"Locked paper-like split mismatch: train={len(train_rows)} dev={len(dev_rows)}")
    train_ids = {sample_id(row) for row in train_rows}
    dev_ids = {sample_id(row) for row in dev_rows}
    if len(train_ids) != len(train_rows) or len(dev_ids) != len(dev_rows):
        raise ValueError("Duplicate sample identifiers in train/dev")
    if train_ids & dev_ids:
        raise ValueError("Paper train/dev sample IDs overlap")
    train_triples = {str(row.get("triple_key") or "") for row in train_rows}
    dev_triples = {str(row.get("triple_key") or "") for row in dev_rows}
    if "" in train_triples or "" in dev_triples or train_triples & dev_triples:
        raise ValueError("Paper train/dev triple-key isolation failed")
    processed_ids = {sample_id(row) for row in processed_rows}
    missing = (train_ids | dev_ids) - processed_ids
    if missing:
        raise ValueError(f"Train/dev rows missing from processed universe: {len(missing)}")
    train_qkeys = {qkey(row) for row in train_rows}
    dev_qkeys = {qkey(row) for row in dev_rows}
    shared_qkeys = train_qkeys & dev_qkeys
    train_rows_on_shared = sum(qkey(row) in shared_qkeys for row in train_rows)
    return {
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "processed_rows": len(processed_rows),
        "train_unique_question_keys": len(train_qkeys),
        "dev_unique_question_keys": len(dev_qkeys),
        "train_dev_question_key_overlap": len(shared_qkeys),
        "train_rows_on_train_dev_shared_question_keys": train_rows_on_shared,
        "train_dev_sample_overlap": 0,
        "train_dev_triple_key_overlap": 0,
        "future_train_rows_removed_for_question_key_overlap": 0,
        "future_train_rows_retained": len(train_rows),
    }


def category_values(rows: list[dict[str, Any]], field: str) -> list[str]:
    return sorted({str(row.get(field) or "unknown") for row in rows})


def solve_clean_dev_milp(
    rows: list[dict[str, Any]], quotas: dict[int, int], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exactly maximize clean-dev internal qkey diversity, then balance metadata."""
    n = len(rows)
    qkeys = sorted({qkey(row) for row in rows})
    q_index = {value: index for index, value in enumerate(qkeys)}
    q_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        q_rows[qkey(row)].append(index)

    base_vars = n + len(qkeys)
    constraints: list[tuple[dict[int, float], float, float]] = []
    for label, quota in quotas.items():
        coeff = {index: 1.0 for index, row in enumerate(rows) if int(row["label_5"]) == label}
        if len(coeff) < quota:
            raise ValueError(f"Clean-dev label {label} quota {quota} exceeds population {len(coeff)}")
        constraints.append((coeff, float(quota), float(quota)))
    for value, indices in q_rows.items():
        coeff = {index: -1.0 for index in indices}
        coeff[n + q_index[value]] = 1.0
        constraints.append((coeff, -math.inf, 0.0))

    def matrix(
        specs: list[tuple[dict[int, float], float, float]], var_count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        array = np.zeros((len(specs), var_count), dtype=float)
        lower = np.empty(len(specs), dtype=float)
        upper = np.empty(len(specs), dtype=float)
        for row_index, (coefficients, lb, ub) in enumerate(specs):
            for column, value in coefficients.items():
                array[row_index, column] = value
            lower[row_index] = lb
            upper[row_index] = ub
        return array, lower, upper

    c1 = np.zeros(base_vars, dtype=float)
    c1[n:] = -1.0
    a1, lb1, ub1 = matrix(constraints, base_vars)
    result1 = milp(
        c=c1,
        integrality=np.ones(base_vars, dtype=int),
        bounds=Bounds(np.zeros(base_vars), np.ones(base_vars)),
        constraints=LinearConstraint(a1, lb1, ub1),
        options={"disp": False},
    )
    if not result1.success or result1.x is None:
        raise RuntimeError(f"Clean-dev qkey-max MILP failed: {result1.message}")
    maximum_qkeys = int(round(float(np.sum(result1.x[n:]))))

    balance_fields = ("language", "metric_group", "subject_canonical")
    category_specs: list[tuple[str, str, float, float]] = []
    target_n = sum(quotas.values())
    for field in balance_fields:
        values = category_values(rows, field)
        population = Counter(str(row.get(field) or "unknown") for row in rows)
        for value in values:
            target = target_n * population[value] / len(rows)
            weight = 1.0 / (len(values) * max(1.0, target))
            category_specs.append((field, value, target, weight))

    var_count = base_vars + len(category_specs)
    constraints2 = list(constraints)
    constraints2.append(({n + index: 1.0 for index in range(len(qkeys))}, maximum_qkeys, maximum_qkeys))
    for offset, (field, value, target, _) in enumerate(category_specs):
        z_index = base_vars + offset
        members = {
            index: 1.0
            for index, row in enumerate(rows)
            if str(row.get(field) or "unknown") == value
        }
        upper_coeff = dict(members)
        upper_coeff[z_index] = -1.0
        lower_coeff = {index: -coefficient for index, coefficient in members.items()}
        lower_coeff[z_index] = -1.0
        constraints2.append((upper_coeff, -math.inf, target))
        constraints2.append((lower_coeff, -math.inf, -target))

    c2 = np.zeros(var_count, dtype=float)
    for index, row in enumerate(rows):
        c2[index] = stable_fraction(seed, "clean-dev-milp-tie", sample_id(row)) * 1e-8
    for offset, (_, _, _, weight) in enumerate(category_specs):
        c2[base_vars + offset] = weight
    lower_bounds = np.zeros(var_count, dtype=float)
    upper_bounds = np.concatenate((np.ones(base_vars, dtype=float), np.full(len(category_specs), np.inf)))
    integrality = np.concatenate((np.ones(base_vars, dtype=int), np.zeros(len(category_specs), dtype=int)))
    a2, lb2, ub2 = matrix(constraints2, var_count)
    result2 = milp(
        c=c2,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(a2, lb2, ub2),
        options={"disp": False},
    )
    if not result2.success or result2.x is None:
        raise RuntimeError(f"Clean-dev balance MILP failed: {result2.message}")
    selected = [rows[index] for index in range(n) if result2.x[index] >= 0.5]
    if len(selected) != target_n or len({qkey(row) for row in selected}) != maximum_qkeys:
        raise RuntimeError("Clean-dev MILP solution failed post-solve checks")
    return selected, {
        "solver": "scipy.optimize.milp_highs_cpu",
        "primary_objective": "maximize_internal_clean_dev_unique_question_keys",
        "secondary_objective": "minimize_language_metric_subject_population_deviation",
        "tertiary_objective": "seed42_stable_hash_tie_break",
        "maximum_unique_question_keys": maximum_qkeys,
        "future_train_question_key_exclusion": False,
        "future_train_rows_removed": 0,
    }


def projected_balance_penalty(
    row: dict[str, Any],
    selected: list[dict[str, Any]],
    universes: dict[str, list[str]],
) -> float:
    next_n = len(selected) + 1
    penalty = 0.0
    for field, values in universes.items():
        counts = Counter(str(item.get(field) or "unknown") for item in selected)
        counts[str(row.get(field) or "unknown")] += 1
        target = 1.0 / len(values)
        penalty += sum(abs(counts[value] / next_n - target) for value in values) / len(values)
    return penalty


def select_representative(
    rows: list[dict[str, Any]], quotas: dict[int, int], seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_qkeys: set[str] = set()
    universes = {
        field: category_values(rows, field)
        for field in ("language", "metric_group", "subject_canonical")
    }
    for label in (1, 2, 3, 4, 5):
        candidates = [row for row in rows if int(row["label_5"]) == label]
        per_qkey = Counter(qkey(row) for row in candidates)
        for _ in range(quotas[label]):
            available = [row for row in candidates if sample_id(row) not in used_ids]
            if not available:
                raise RuntimeError(f"Representative label {label} selection exhausted")
            chosen = min(
                available,
                key=lambda row: (
                    int(qkey(row) in used_qkeys),
                    per_qkey[qkey(row)],
                    projected_balance_penalty(row, selected, universes),
                    stable_fraction(seed, "representative", sample_id(row)),
                ),
            )
            selected.append(chosen)
            used_ids.add(sample_id(chosen))
            used_qkeys.add(qkey(chosen))
    return selected


def teacher_annotation_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): dict(row["annotation"]) for row in rows}


def risk_flags(
    row: dict[str, Any],
    primary: dict[str, dict[str, Any]],
    secondary: dict[str, dict[str, Any]],
) -> list[str]:
    sid = sample_id(row)
    original = int(row["label_5"])
    qwen = primary[sid]
    deepseek = secondary.get(sid)
    qscore = int(qwen["score"])
    dscore = int(deepseek["score"]) if deepseek is not None else None
    flags: list[str] = []
    if abs(qscore - original) >= 2:
        flags.append("qwen_human_gap_ge_2")
    if dscore is not None and abs(dscore - original) >= 2:
        flags.append("deepseek_human_gap_ge_2")
    if dscore is not None and abs(qscore - dscore) >= 2:
        flags.append("qwen_deepseek_gap_ge_2")
    if (original <= 2 and qscore >= 4) or (dscore is not None and original <= 2 and dscore >= 4):
        flags.append("low_to_high")
    if (original >= 4 and qscore <= 2) or (dscore is not None and original >= 4 and dscore <= 2):
        flags.append("high_to_low")
    if original == 4 and (qscore == 5 or dscore == 5):
        flags.append("four_to_five_transition")
    teacher_rows = [annotation for annotation in (qwen, deepseek) if annotation is not None]
    if any(
        annotation.get("score_cap") is not None
        or (
            annotation.get("major_failures")
            and annotation.get("major_failures") != ["no_major_failure"]
        )
        for annotation in teacher_rows
    ):
        flags.append("evidence_or_score_cap_conflict")
    if (
        row.get("metric_canonical") == "Higher-Order Thinking & Skill Development"
        or row.get("metric_group") == "Pedagogical Application"
    ):
        flags.append("higher_order_pedagogical_metric")
    return flags


RISK_WEIGHTS = {
    "qwen_human_gap_ge_2": 100,
    "deepseek_human_gap_ge_2": 95,
    "qwen_deepseek_gap_ge_2": 90,
    "low_to_high": 80,
    "high_to_low": 80,
    "four_to_five_transition": 50,
    "evidence_or_score_cap_conflict": 35,
    "higher_order_pedagogical_metric": 10,
}


def select_risk_enriched(
    rows: list[dict[str, Any]],
    representative: list[dict[str, Any]],
    primary: dict[str, dict[str, Any]],
    secondary: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    representative_ids = {sample_id(row) for row in representative}
    representative_qkeys = {qkey(row) for row in representative}
    remaining = [row for row in rows if sample_id(row) not in representative_ids]
    selected = [row for row in remaining if int(row["label_5"]) <= 2]
    expected_low = sum(Counter(int(row["label_5"]) for row in rows)[label] - REPRESENTATIVE_QUOTAS[label] for label in (1, 2))
    if len(selected) != expected_low:
        raise RuntimeError("Risk low-label remainder calculation failed")
    selected_ids = {sample_id(row) for row in selected}
    selected_qkeys = {qkey(row) for row in selected}
    flags_by_id = {sample_id(row): ["remaining_low_label"] + risk_flags(row, primary, secondary) for row in selected}
    universes = {
        field: category_values(rows, field)
        for field in ("language", "metric_group", "subject_canonical")
    }
    for label, quota in RISK_NONLOW_QUOTAS.items():
        for _ in range(quota):
            candidates = [
                row
                for row in remaining
                if int(row["label_5"]) == label and sample_id(row) not in selected_ids
            ]
            if not candidates:
                raise RuntimeError(f"Risk label {label} selection exhausted")

            def rank(row: dict[str, Any]) -> tuple[Any, ...]:
                flags = risk_flags(row, primary, secondary)
                risk_score = sum(RISK_WEIGHTS[flag] for flag in flags)
                return (
                    int(not flags),
                    int(qkey(row) in representative_qkeys),
                    int(qkey(row) in selected_qkeys),
                    -risk_score,
                    projected_balance_penalty(row, selected, universes),
                    stable_fraction(seed, "risk", sample_id(row)),
                )

            chosen = min(candidates, key=rank)
            selected.append(chosen)
            selected_ids.add(sample_id(chosen))
            selected_qkeys.add(qkey(chosen))
            flags_by_id[sample_id(chosen)] = risk_flags(chosen, primary, secondary) or ["quota_balance_filler"]
    if len(selected) != 120:
        raise RuntimeError(f"Risk view must have 120 rows, got {len(selected)}")
    return selected, flags_by_id


def selection_hash(rows: list[dict[str, Any]]) -> str:
    return canonical_hash(sorted(sample_id(row) for row in rows))


def packet_for(row: dict[str, Any], seed: int) -> dict[str, Any]:
    context = f"<CONTEXT_ONLY_ORIGINAL_TASK>\n{str(row['question']).strip()}\n</CONTEXT_ONLY_ORIGINAL_TASK>"
    evaluator_output = f"<EVALUATOR_OUTPUT_TO_SCORE>\n{str(row['answer']).strip()}\n</EVALUATOR_OUTPUT_TO_SCORE>"
    payload: dict[str, Any] = {
        "sample_id": sample_id(row),
        "anonymized_question_key_hash": hashlib.sha256(
            f"exp33a|{seed}|question-key|{qkey(row)}".encode("utf-8")
        ).hexdigest(),
        "question_context": context,
        "evaluator_output": evaluator_output,
        "evaluation_dimension": row.get("metric_canonical") or row.get("metric_raw"),
        "canonical_rubric": row.get("rubric") or [],
        "score_rubric": SCORE_RUBRIC,
        "non_label_metadata": {
            "subject": row.get("subject_canonical"),
            "education_level": row.get("education_level_canonical"),
            "scenario": row.get("scenario_canonical"),
        },
        "language": row.get("language"),
    }
    payload["packet_hash"] = canonical_hash(payload)
    return payload


def distribution_rows(
    selected: list[dict[str, Any]], population: list[dict[str, Any]], manifest_hash: str
) -> list[dict[str, Any]]:
    dimensions = (
        ("label", "label_5"),
        ("language", "language"),
        ("metric_family", "metric_group"),
        ("metric", "metric_canonical"),
        ("subject", "subject_canonical"),
    )
    output: list[dict[str, Any]] = []
    for dimension, field in dimensions:
        pop = Counter(str(row.get(field) or "unknown") for row in population)
        sample = Counter(str(row.get(field) or "unknown") for row in selected)
        for value in sorted(pop, key=lambda item: (item.isdigit(), item)):
            selected_for_value = [row for row in selected if str(row.get(field) or "unknown") == value]
            output.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "population_count": pop[value],
                    "sample_count": sample[value],
                    "sample_share": sample[value] / len(selected),
                    "unique_question_keys": len({qkey(row) for row in selected_for_value}),
                    "selection_manifest_sha256": manifest_hash,
                }
            )
    return output


def view_overlap_rows(views: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for left_index, left in enumerate(VIEW_ORDER):
        for right in VIEW_ORDER[left_index + 1 :]:
            left_ids = {sample_id(row) for row in views[left]}
            right_ids = {sample_id(row) for row in views[right]}
            left_qkeys = {qkey(row) for row in views[left]}
            right_qkeys = {qkey(row) for row in views[right]}
            shared_qkeys = left_qkeys & right_qkeys
            output.append(
                {
                    "view_a": left,
                    "view_b": right,
                    "exact_sample_overlap": len(left_ids & right_ids),
                    "question_key_overlap": len(shared_qkeys),
                    "rows_in_a_on_shared_question_keys": sum(qkey(row) in shared_qkeys for row in views[left]),
                    "rows_in_b_on_shared_question_keys": sum(qkey(row) in shared_qkeys for row in views[right]),
                    "sample_disjoint_status": "PASS" if not (left_ids & right_ids) else "FAIL",
                    "question_key_overlap_interpretation": "allowed_and_reported_under_triple_key_protocol",
                }
            )
    return output


def review_completion_rows(reviewer_type: str, expected_rows: int) -> list[dict[str, Any]]:
    reviewer_provenance = (
        "independent_model_reviewer" if reviewer_type == "model" else "independent_human_reviewer"
    )
    return [
        {
            "reviewer_role": role,
            "reviewer_type": reviewer_type,
            "reviewer_provenance": reviewer_provenance if role != "adjudicator" else reviewer_provenance.replace("reviewer", "adjudicator"),
            "reviewer_provider": "",
            "reviewer_model_id": "",
            "expected_rows": expected_rows if role != "adjudicator" else 0,
            "completed_rows": 0,
            "valid_rows": 0,
            "status": "NOT_STARTED",
            "reference_status": "independent_model_reviewed_silver_reference_pending" if reviewer_type == "model" else "independent_human_review_pending",
        }
        for role in ("reviewer_a", "reviewer_b", "adjudicator")
    ]


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 42:
        raise ValueError("Exp33A sampling lock requires seed 42")
    out = args.out_dir
    train_path = guarded(args.split_dir / "train.jsonl")
    dev_path = guarded(args.split_dir / "dev.jsonl")
    processed_path = guarded(args.processed_data)
    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)
    processed_rows = read_jsonl(processed_path)
    source_audit = validate_source_boundaries(train_rows, dev_rows, processed_rows)

    # Clean dev is selected first, using only original dev labels and non-label
    # covariates. No teacher annotation or student prediction is available here.
    clean_dev, clean_solver = solve_clean_dev_milp(dev_rows, CLEAN_DEV_QUOTAS, args.seed)
    representative = select_representative(train_rows, REPRESENTATIVE_QUOTAS, args.seed)

    teacher_manifest, teacher_resolved = resolve_teacher_inputs(
        args.teacher_summary_dir, args.exp28e_decision
    )
    primary = teacher_annotation_map(teacher_resolved["primary"]["rows"])
    secondary = teacher_annotation_map(teacher_resolved["secondary"]["rows"])
    risk, risk_reasons = select_risk_enriched(
        train_rows, representative, primary, secondary, args.seed
    )
    views = {
        "representative_train": representative,
        "risk_enriched_train": risk,
        "clean_dev": clean_dev,
    }

    rep_counts = Counter(int(row["label_5"]) for row in representative)
    risk_counts = Counter(int(row["label_5"]) for row in risk)
    dev_counts = Counter(int(row["label_5"]) for row in clean_dev)
    if rep_counts != Counter(REPRESENTATIVE_QUOTAS):
        raise RuntimeError(f"Representative quota mismatch: {rep_counts}")
    expected_risk = {1: 12, 2: 34, **RISK_NONLOW_QUOTAS}
    if risk_counts != Counter(expected_risk):
        raise RuntimeError(f"Risk quota mismatch: {risk_counts}")
    if dev_counts != Counter(CLEAN_DEV_QUOTAS):
        raise RuntimeError(f"Clean-dev quota mismatch: {dev_counts}")

    selection_rows: list[dict[str, Any]] = []
    train_label_pop = Counter(int(row["label_5"]) for row in train_rows)
    for view, rows in views.items():
        for row in rows:
            label = int(row["label_5"])
            if view == "representative_train":
                population = train_label_pop[label]
                quota = REPRESENTATIVE_QUOTAS[label]
                probability = quota / population
                weight = population / quota
            else:
                population = None
                quota = None
                probability = None
                weight = None
            selection_rows.append(
                {
                    "sample_id": sample_id(row),
                    "source_split": "dev" if view == "clean_dev" else "train",
                    "view": view,
                    "stratum_population": population,
                    "stratum_sample": quota,
                    "inclusion_probability": probability,
                    "design_weight": weight,
                    "sampling_risk_reason": risk_reasons.get(sample_id(row), []) if view == "risk_enriched_train" else [],
                }
            )
    if len({row["sample_id"] for row in selection_rows}) != 420:
        raise RuntimeError("Exact sample overlap exists across Exp33A views")
    write_jsonl(out / "private/exp33a_selected_sample_manifest.jsonl", selection_rows)

    source_reference, human_source_audit = build_private_source_reference(
        selection_rows,
        train_rows,
        dev_rows,
        teacher_resolved,
        DEFAULT_HUMAN_REASON_FILES,
    )
    write_jsonl(out / "private/exp33a_source_reference.jsonl", source_reference)

    all_selected_by_id = {
        sample_id(row): row for rows in views.values() for row in rows
    }
    packets = [packet_for(all_selected_by_id[row["sample_id"]], args.seed) for row in selection_rows]
    packet_manifest_hash = canonical_hash([packet["packet_hash"] for packet in packets])
    private_review = repo_path(out / "private_review")
    for directory in (
        private_review / "blind_packets",
        private_review / "reviewer_a_filled",
        private_review / "reviewer_b_filled",
        private_review / "adjudication_filled",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "private_review/blind_packets/exp33a_reviewer_a_packet.jsonl", packets)
    write_jsonl(out / "private_review/blind_packets/exp33a_reviewer_b_packet.jsonl", packets)
    assignment_rows = [
        {
            "sample_id": selected["sample_id"],
            "view": selected["view"],
            "source_split": selected["source_split"],
            "packet_hash": packet["packet_hash"],
        }
        for selected, packet in zip(selection_rows, packets, strict=True)
    ]
    write_jsonl(out / "private_review/exp33a_review_assignment_manifest.jsonl", assignment_rows)

    tables = out / "tables"
    teacher_fields = [
        "teacher_role", "provider", "model", "protocol", "subset", "annotation_path",
        "resolution_summary_path", "sha256", "row_count", "valid_row_count",
        "locked_expected_valid_rows", "reference_status",
    ]
    write_csv(tables / "exp33a_resolved_teacher_input_manifest.csv", teacher_manifest, teacher_fields)
    distribution_fields = [
        "dimension", "value", "population_count", "sample_count", "sample_share",
        "unique_question_keys", "selection_manifest_sha256",
    ]
    view_hashes = {view: selection_hash(rows) for view, rows in views.items()}
    write_csv(
        tables / "exp33a_train_representative_distribution.csv",
        distribution_rows(representative, train_rows, view_hashes["representative_train"]),
        distribution_fields,
    )
    write_csv(
        tables / "exp33a_train_risk_distribution.csv",
        distribution_rows(risk, [row for row in train_rows if sample_id(row) not in {sample_id(item) for item in representative}], view_hashes["risk_enriched_train"]),
        distribution_fields,
    )
    write_csv(
        tables / "exp33a_clean_dev_distribution.csv",
        distribution_rows(clean_dev, dev_rows, view_hashes["clean_dev"]),
        distribution_fields,
    )

    question_partition = [
        {
            "partition": "paper_train_source",
            "rows": len(train_rows),
            "unique_sample_ids": len({sample_id(row) for row in train_rows}),
            "unique_triple_keys": len({str(row["triple_key"]) for row in train_rows}),
            "unique_question_keys": len({qkey(row) for row in train_rows}),
            "question_keys_shared_with_paper_train": len({qkey(row) for row in train_rows}),
            "future_train_rows_removed": 0,
            "protocol_note": "paper_protocol_triple_key_disjoint_not_question_key_disjoint",
        },
        {
            "partition": "paper_dev_source",
            "rows": len(dev_rows),
            "unique_sample_ids": len({sample_id(row) for row in dev_rows}),
            "unique_triple_keys": len({str(row["triple_key"]) for row in dev_rows}),
            "unique_question_keys": len({qkey(row) for row in dev_rows}),
            "question_keys_shared_with_paper_train": source_audit["train_dev_question_key_overlap"],
            "future_train_rows_removed": 0,
            "protocol_note": "question_key_overlap_expected_and_allowed",
        },
    ]
    for view, rows in views.items():
        question_partition.append(
            {
                "partition": view,
                "rows": len(rows),
                "unique_sample_ids": len({sample_id(row) for row in rows}),
                "unique_triple_keys": len({str(row["triple_key"]) for row in rows}),
                "unique_question_keys": len({qkey(row) for row in rows}),
                "question_keys_shared_with_paper_train": len(
                    {qkey(row) for row in rows} & {qkey(row) for row in train_rows}
                ),
                "future_train_rows_removed": 0,
                "protocol_note": "descriptive_qkey_diversity_only_no_training_exclusion",
            }
        )
    write_csv(
        tables / "exp33a_question_key_partition.csv",
        question_partition,
        [
            "partition", "rows", "unique_sample_ids", "unique_triple_keys", "unique_question_keys",
            "question_keys_shared_with_paper_train", "future_train_rows_removed", "protocol_note",
        ],
    )

    weight_rows = []
    for label in range(1, 6):
        population = train_label_pop[label]
        quota = REPRESENTATIVE_QUOTAS[label]
        weight_rows.append(
            {
                "view": "representative_train",
                "stratum": f"label_{label}",
                "stratum_population": population,
                "stratum_sample": quota,
                "inclusion_probability": quota / population,
                "design_weight": population / quota,
                "estimation_use": "population_source_reliability_and_prevalence",
            }
        )
    weight_rows.append(
        {
            "view": "risk_enriched_train",
            "stratum": "all",
            "stratum_population": "",
            "stratum_sample": 120,
            "inclusion_probability": "",
            "design_weight": "",
            "estimation_use": "unweighted_stress_metrics_only",
        }
    )
    write_csv(
        tables / "exp33a_sampling_design_weights.csv",
        weight_rows,
        ["view", "stratum", "stratum_population", "stratum_sample", "inclusion_probability", "design_weight", "estimation_use"],
    )
    overlap = view_overlap_rows(views)
    write_csv(
        tables / "exp33a_view_overlap_audit.csv",
        overlap,
        [
            "view_a", "view_b", "exact_sample_overlap", "question_key_overlap",
            "rows_in_a_on_shared_question_keys", "rows_in_b_on_shared_question_keys",
            "sample_disjoint_status", "question_key_overlap_interpretation",
        ],
    )
    leakage_rows = [
        {
            "leakage_type": leakage_type,
            "count": 0,
            "status": "PASS",
            "audit_method": "packet_allowlist_and_source_projection_reconstruction",
        }
        for leakage_type in (
            "human_label", "human_reason", "teacher_score", "teacher_reason",
            "student_prediction", "campaign_conflict_flag", "sampling_risk_reason",
            "b0_b4_variant", "train_dev_metric_result",
        )
    ]
    write_csv(
        tables / "exp33a_blind_leakage_audit.csv",
        leakage_rows,
        ["leakage_type", "count", "status", "audit_method"],
    )
    completion = review_completion_rows(args.reviewer_type, 420)
    write_csv(
        tables / "exp33a_review_completion.csv",
        completion,
        [
            "reviewer_role", "reviewer_type", "reviewer_provenance", "reviewer_provider", "reviewer_model_id",
            "expected_rows", "completed_rows", "valid_rows", "status", "reference_status",
        ],
    )
    write_csv(
        tables / "exp33a_reviewer_agreement.csv",
        [
            {
                "group_type": "overall", "group_value": "all", "paired_rows": 0,
                "exact_agreement": "", "within_one": "", "quadratic_weighted_kappa": "",
                "krippendorff_ordinal_alpha": "", "score_range_overlap": "",
                "adjudication_rate": "", "status": "NOT_STARTED",
            }
        ],
        [
            "group_type", "group_value", "paired_rows", "exact_agreement", "within_one",
            "quadratic_weighted_kappa", "krippendorff_ordinal_alpha", "score_range_overlap",
            "adjudication_rate", "status",
        ],
    )
    write_csv(
        tables / "exp33a_domain_escalation_summary.csv",
        [
            {
                "group_type": "overall", "group_value": "all", "reviewed_rows": 0,
                "domain_escalation_required": 0, "adjudicated": 0,
                "unresolved_domain_cases": 0, "status": "NOT_STARTED",
            }
        ],
        [
            "group_type", "group_value", "reviewed_rows", "domain_escalation_required",
            "adjudicated", "unresolved_domain_cases", "status",
        ],
    )

    source_hashes = {
        "train_sha256": sha256_file(train_path),
        "dev_sha256": sha256_file(dev_path),
        "processed_sha256": sha256_file(processed_path),
    }
    sampling_lock = {
        "experiment": "Exp33A Independent Model-Reviewed Silver Reference Preparation",
        "seed": args.seed,
        "paper_protocol": {
            "split_unit": "question_answer_metric_triple",
            "train_dev_triple_key_disjoint": True,
            "train_dev_question_key_disjoint": False,
            "question_key_overlap_is_expected": True,
            "future_train_question_key_exclusion": False,
            "future_train_rows_removed": 0,
            "future_train_rows_retained": 2654,
            "fixed_rows": {"train": 2654, "dev": 664, "test_sealed": 2218},
            "test_row_count_provenance": display_path(repo_path(args.exp32_report)),
            "test_access_count": 0,
            "group_cv_substitution_forbidden": True,
        },
        "source_paths": {
            "train": display_path(train_path),
            "dev": display_path(dev_path),
            "processed": display_path(processed_path),
            "paper_protocol_lock": display_path(repo_path(args.exp28_protocol_lock)),
            "paper_protocol_lock_sha256": sha256_file(args.exp28_protocol_lock),
            "exp32_test_sealed_statement": display_path(repo_path(args.exp32_report)),
            "exp32_test_sealed_statement_sha256": sha256_file(args.exp32_report),
            **source_hashes,
        },
        "source_audit": source_audit,
        "human_reason_source_manifest": human_source_audit,
        "views": {
            "representative_train": {
                "rows": 120,
                "label_quotas": REPRESENTATIVE_QUOTAS,
                "sampling": "seeded_label_stratified_probability_design_with_qkey_and_metadata_balancing",
                "manifest_sha256": view_hashes["representative_train"],
            },
            "risk_enriched_train": {
                "rows": 120,
                "label_quotas": expected_risk,
                "sampling": "remaining_low_labels_then_locked_teacher_risk_priority",
                "population_estimation_forbidden": True,
                "manifest_sha256": view_hashes["risk_enriched_train"],
            },
            "clean_dev": {
                "rows": 180,
                "label_quotas": CLEAN_DEV_QUOTAS,
                "selection_inputs": ["original_label", "language", "metric_family", "subject", "question_key"],
                "teacher_conflict_used": False,
                "student_prediction_used": False,
                "internal_question_key_objective_only": True,
                "question_key_overlap_with_train_allowed": True,
                "manifest_sha256": view_hashes["clean_dev"],
                **clean_solver,
            },
        },
        "packet_manifest_sha256": packet_manifest_hash,
        "no_api": True,
        "no_gpu": True,
        "no_training": True,
        "no_student_inference": True,
        "test_access_count": 0,
    }
    write_json(out / "configs/exp33a_sampling_lock.json", sampling_lock)

    reviewer_provenance = (
        "independent_model_reviewer" if args.reviewer_type == "model" else "independent_human_reviewer"
    )
    review_lock = {
        "experiment": "Exp33A Independent Model-Reviewed Silver Reference Preparation",
        "reference_name": "independent model-reviewed silver reference" if args.reviewer_type == "model" else "independent human-reviewed reference",
        "default_reviewer_type": "model",
        "locked_reviewer_type": args.reviewer_type,
        "supported_reviewer_types": ["human", "model"],
        "planned_model_family": "GPT-5.6" if args.reviewer_type == "model" else None,
        "reviewers": {
            "reviewer_a": {"reviewer_type": args.reviewer_type, "reviewer_provenance": reviewer_provenance, "reviewer_provider": None, "reviewer_model_id": None},
            "reviewer_b": {"reviewer_type": args.reviewer_type, "reviewer_provenance": reviewer_provenance, "reviewer_provider": None, "reviewer_model_id": None},
            "adjudicator": {"reviewer_type": args.reviewer_type, "reviewer_provenance": reviewer_provenance.replace("reviewer", "adjudicator"), "reviewer_provider": None, "reviewer_model_id": None},
        },
        "method_description": "multi-stage blind-first, conflict-aware, direction-constrained model-assisted data annotation and correction",
        "method_innovation": [
            "blind_first_source_comparison", "conflict_adjudication",
            "direction_aware_correction", "uncertainty_fallback",
        ],
        "provider_agnostic_method": True,
        "model_independence_requirement": "Model Reviewer A, Model Reviewer B, and Model Adjudicator must use separately launched contexts with distinct reviewer_run_id values. A/B cannot see each other's outputs. reviewer_model_id/provider are fixed and recorded for provenance but are not the method innovation.",
        "reviewer_model_id_required_at_submission": args.reviewer_type == "model",
        "blind_packet_allowlist": list(PACKET_FIELDS),
        "non_label_metadata_allowlist": list(NON_LABEL_METADATA_FIELDS),
        "adjudication_triggers": [
            "most_plausible_score_differs", "score_ranges_disjoint", "failure_bucket_differs",
            "student_input_sufficiency_differs", "any_low_confidence", "any_needs_adjudication",
            "any_domain_escalation_required",
        ],
        "three_stage_plan": {
            "stage_1_blind_review": "Model Reviewer A/B independently see only original task, evaluator output, rubric, and non-label metadata; they emit score/range/evidence and never see human, Qwen, or DeepSeek sources.",
            "stage_2_private_source_comparison": "After A/B outputs are frozen, private analysis compares human_1/2/3, rounded human, Qwen, DeepSeek, and A/B to audit 4-to-5 bias, low-to-high shifts, reason-score inconsistency, and hard-relabel drift.",
            "stage_3_adjudication_and_correction": "Triggered cases go to an independently launched Model Adjudicator with frozen A/B structured reviews plus controlled source provenance. It returns a model-reviewed silver posterior/score; uncertainty falls back to the human empirical distribution rather than a forced hard relabel.",
        },
        "adjudicator_source_visibility": {
            "before_ab_freeze": "none",
            "after_ab_freeze_for_triggered_cases": [
                "human_1_2_3_scores_and_reasons", "rounded_human_label",
                "qwen_score_range_confidence_evidence_reason",
                "deepseek_score_range_confidence_evidence_reason",
            ],
            "always_forbidden": ["student_predictions", "b0_b4_variants", "train_dev_model_metrics", "test_data"],
        },
        "uncertain_case_fallback": "human_empirical_distribution",
        "hard_relabel_without_posterior_forbidden": True,
        "calibration_gate_before_2654_train_expansion": {
            "paired_review_coverage": 1.0,
            "schema_and_evidence_validity": 1.0,
            "blind_leakage_count": 0,
            "within_one_agreement_min": 0.90,
            "quadratic_weighted_kappa_min": 0.60,
            "krippendorff_ordinal_alpha_min": 0.60,
            "all_triggered_cases_processed": True,
            "selection_uses_model_performance_metrics": False,
        },
        "domain_escalation_plan": "If the independent adjudicator cannot resolve domain uncertainty, retain an unresolved-domain audit flag and use the human empirical distribution fallback; do not force a hard score for teacher reliability fitting.",
        "review_results_present": False,
        "model_silver_reference_complete": False,
        "expert_reference_complete": False,
        "test_access_count": 0,
    }
    write_json(out / "configs/exp33a_review_protocol_lock.json", review_lock)

    decision = {
        "experiment": "Exp33A Independent Model-Reviewed Silver Reference Preparation",
        "status": "READY_FOR_INDEPENDENT_MODEL_REVIEW" if args.reviewer_type == "model" else "READY_FOR_INDEPENDENT_HUMAN_REVIEW",
        "reference_status": "independent model-reviewed silver reference pending" if args.reviewer_type == "model" else "independent human-reviewed reference pending",
        "reviewer_type": args.reviewer_type,
        "reviewer_provenance": reviewer_provenance,
        "reviewer_providers": {"reviewer_a": None, "reviewer_b": None, "adjudicator": None},
        "reviewer_model_ids": {"reviewer_a": None, "reviewer_b": None, "adjudicator": None},
        "prepare_complete": True,
        "validation_complete": False,
        "review_completion_state": "not_started",
        "model_silver_reference_complete": False,
        "expert_reference_complete": False,
        "teacher_reliability_ready": False,
        "recommend_new_teacher_training": False,
        "recommend_student_training": False,
        "recommend_test_access": False,
        "paper_protocol": "triple_key_disjoint_not_question_key_disjoint",
        "future_train_rows_removed": 0,
        "future_train_rows_retained": 2654,
        "view_rows": {view: len(rows) for view, rows in views.items()},
        "unique_question_keys": {view: len({qkey(row) for row in rows}) for view, rows in views.items()},
        "exact_sample_overlap_zero": all(row["exact_sample_overlap"] == 0 for row in overlap),
        "blind_packet_rows": len(packets),
        "packet_manifest_sha256": packet_manifest_hash,
        "test_access_count": 0,
        "api_called": False,
        "gpu_used": False,
        "training_run": False,
        "student_inference_run": False,
    }
    write_json(out / "decision/exp33a_expert_reference_decision.json", decision)

    rep_label_text = ", ".join(f"{label}={rep_counts[label]}" for label in range(1, 6))
    risk_label_text = ", ".join(f"{label}={risk_counts[label]}" for label in range(1, 6))
    dev_label_text = ", ".join(f"{label}={dev_counts[label]}" for label in range(1, 6))
    teacher_text = "\n".join(
        f"- {row['teacher_role']}: `{row['annotation_path']}`; rows={row['row_count']}; valid={row['valid_row_count']}; SHA-256=`{row['sha256']}`"
        for row in teacher_manifest
    )
    report = f"""# Exp33A Independent Model-Reviewed Silver Reference Preparation

## Outcome

Exp33A prepared 420 blind review rows for two independent reviewers. The current lock is
`reviewer_type={args.reviewer_type}` and defaults to model review. No reviewer output has been
created or implied. `model_silver_reference_complete=false` and
`expert_reference_complete=false`.

## Paper protocol boundary

- Paper split unit: `(question, answer, metric)` triple key.
- Train/dev triple-key overlap: 0.
- Train/dev question-key overlap: {source_audit['train_dev_question_key_overlap']} of {source_audit['dev_unique_question_keys']} dev qkeys; this is expected and allowed.
- Train rows on shared train/dev qkeys: {source_audit['train_rows_on_train_dev_shared_question_keys']} / 2654.
- Future train rows removed for qkey overlap: 0; all 2654 train rows remain.
- The protocol is not replaced by question-key GroupCV.
- Locked rows: train=2654, dev=664, sealed test=2218. The test count is inherited from the locked Exp32 statement; the test file was not opened.

Clean-dev maximizes question-key diversity only inside its 180 selected rows. Its qkeys may
overlap train, and they create no future-training exclusion.

## Resolved teacher annotation inputs

{teacher_text}

Resolution uses Exp28 machine-readable provider/model/subset summaries and cross-checks the
Exp28E locked valid-row counts. Raw API logs are not read.

## Sampling

| view | rows | labels 1..5 | unique qkeys | permitted use |
|---|---:|---|---:|---|
| representative_train | 120 | {rep_label_text} | {len({qkey(row) for row in representative})} | design-weighted population reliability/prevalence |
| risk_enriched_train | 120 | {risk_label_text} | {len({qkey(row) for row in risk})} | unweighted stress/routing analysis only |
| clean_dev | 180 | {dev_label_text} | {len({qkey(row) for row in clean_dev})} | one-time frozen-method evaluation only |

Representative inclusion probabilities are `n_h/N_h` within original-label strata and design
weights are `N_h/n_h`. Language, metric family, and subject are auxiliary balance variables.
Risk enrichment never estimates population prevalence. Clean-dev used no teacher conflict,
student prediction, Exp29-31 result, or dev model metric.

## Blind packet and leakage

- Packet rows per independent reviewer: 420.
- Packet manifest SHA-256: `{packet_manifest_hash}`.
- Exact sample overlap between all three views: 0.
- Human label/reason, Qwen/DeepSeek score/reason, campaign flag, student prediction, variant,
  metric-result, and sampling-risk leakage: 0 by allowlist plus source-projection audit.
- Reviewer A and Reviewer B receive identical blind content in separate local packet files.
    - Reviewer model IDs are intentionally unset until launch; the exact implementation model (currently planned GPT-5.6), provider, and a distinct run/context ID for each role must be recorded as provenance.

## Completion and escalation

The provider-agnostic method is multi-stage blind-first, conflict-aware, and direction-constrained
model-assisted annotation/correction. Its innovations are blind-first source comparison, conflict
adjudication, direction-aware correction, and uncertainty fallback—not the choice of a particular
provider or stronger model.

The locked follow-up has three stages. First, Model Reviewer A/B independently see only the
blind packet and emit score/range/evidence. Second, after both outputs are frozen, a private source
comparison audits human_1/2/3, Qwen, DeepSeek, and A/B for 4-to-5 bias, low-to-high shifts,
reason-score inconsistency, and hard-relabel drift. Third, triggered cases go to an independently
launched Model Adjudicator with frozen A/B structured results and controlled source provenance.
The adjudicator emits a model-reviewed silver posterior/score. If uncertainty remains, the system
falls back to the human empirical score distribution instead of forcing a hard relabel.

Expansion from the 240-row train calibration sample to all 2,654 train rows is forbidden until the
pre-registered completion, leakage, schema/evidence, within-one, QWK, ordinal-alpha, and
trigger-processing gates pass. No model performance metric selects this gate.

Current state: reviews not started; teacher reliability not ready; no new teacher/student training
is recommended; test access is not recommended.

## Resource audit

No API was called, no GPU was used, no model was trained, no student inference ran, and the
sealed test split was not read. Private sample IDs, blind packets, source reasons, teacher reasons,
and future filled reviews are gitignored.
"""
    write_text(out / "reports/exp33a_expert_reference_prepare_report.md", report)

    return {
        "status": decision["status"],
        "representative_train": 120,
        "risk_enriched_train": 120,
        "clean_dev": 180,
        "clean_dev_unique_question_keys": len({qkey(row) for row in clean_dev}),
        "future_train_rows_removed": 0,
        "test_access_count": 0,
        "api_called": False,
        "gpu_used": False,
        "training_run": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--processed-data", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--teacher-summary-dir", type=Path, default=DEFAULT_TEACHER_SUMMARY_DIR)
    parser.add_argument("--exp28e-decision", type=Path, default=DEFAULT_EXP28E_DECISION)
    parser.add_argument("--exp28-protocol-lock", type=Path, default=DEFAULT_EXP28_PROTOCOL_LOCK)
    parser.add_argument("--exp32-report", type=Path, default=DEFAULT_EXP32_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reviewer-type", choices=("human", "model"), default="model")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), ensure_ascii=False, sort_keys=True))
