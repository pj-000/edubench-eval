"""Prepare Exp26 train-only hidden-failure evidence expansion data.

Exp26A is a data construction step only. It prepares an evidence-aware SFT
dataset, a hidden-failure-oriented DPO dataset, and annotation/audit manifests
for the next SFT -> ORC/SRC-DPO round.

Guardrails:
- build train data only from question_seed42/train.jsonl;
- read dev/test only for sample_id/question_key leakage checks;
- never place human rationale in the user prompt;
- mark weak/counterfactual/teacher-needed fields explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_r7_reason_recovered_real_dpo import (  # noqa: E402
    build_reason_bundles,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    MODEL_PATH,
    OPENAI_TAGS,
    clean,
    clamp_score,
    language,
    messages_for,
    metric_name,
    read_csv_rows,
    read_jsonl,
    sample_id,
    subject,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp25_structured_src_dpo import (  # noqa: E402
    parse_assistant_payload,
)
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_text  # noqa: E402


DEFAULT_TRAIN_JSONL = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV_JSONL = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_TEST_JSONL = Path("thesis_exp/data/splits/question_seed42/test.jsonl")
DEFAULT_REASON_ROOT = Path("5-grades")
DEFAULT_A0_CANDIDATES = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_hidden_failure_candidates.csv"
)
DEFAULT_A0_HIGH_CONTROLS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_clean_high_controls.csv"
)
DEFAULT_EXP21_CANDIDATES = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp21_d1_like_risk_annotation_seed42/"
    "train_candidates/exp21_train_risk_annotation_candidates.csv"
)
DEFAULT_R7H_MIXED = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp25_structured_src_dpo_seed42/"
    "data/edubench_r7h_structured_src_dpo_train.json"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp26_hidden_failure_expansion_seed42"
)

SFT_NAME = "edubench_exp26a_evidence_aware_sft_train"
DPO_NAME = "edubench_exp26b_hidden_failure_dpo_train"
SFT_FILE = "data/edubench_exp26a_evidence_aware_sft_train.json"
DPO_FILE = "data/edubench_exp26b_hidden_failure_dpo_train.json"

FAILURE_VOCAB = {
    "missing_key_point",
    "factual_or_rubric_mismatch",
    "answer_key_or_reference_mismatch",
    "surface_fluent_but_hidden_defect",
    "insufficient_evidence",
    "task_constraint_violation",
    "format_violation",
    "possible_label_conflict",
    "partial_or_incomplete",
    "no_major_failure",
    "unclear",
}


def sha1(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def assistant_message(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    }


def sft_example(row: dict[str, Any], target: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": messages_for(row) + [assistant_message(target)],
        **meta,
    }


def dpo_example(
    row: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "messages": messages_for(row),
        "chosen": assistant_message(chosen),
        "rejected": assistant_message(rejected),
        **meta,
    }


def split_identity(path: Path) -> tuple[set[str], set[str], int]:
    sample_ids: set[str] = set()
    question_keys: set[str] = set()
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            sid = clean(row.get("sample_id") or row.get("record_id") or row.get("id"))
            qkey = clean(row.get("question_key") or row.get("question_id"))
            if sid:
                sample_ids.add(sid)
            if qkey:
                question_keys.add(qkey)
    return sample_ids, question_keys, n


def question_key(row: dict[str, Any]) -> str:
    return clean(row.get("question_key") or row.get("question_id"))


def rubric_clause(row: dict[str, Any], score: int) -> str:
    rubric = clean(row.get("rubric"))
    candidates: list[str] = []
    try:
        parsed = json.loads(rubric)
        if isinstance(parsed, list):
            candidates = [clean(item) for item in parsed]
    except Exception:
        candidates = [line.strip(" -") for line in rubric.splitlines() if line.strip()]
    pattern = re.compile(rf"^\s*{score}\s*[:：]")
    for item in candidates:
        if pattern.search(item):
            return item
    for item in candidates:
        if str(score) in item[:8]:
            return item
    return ""


def score_cap(label: int) -> int | None:
    if label <= 2:
        return label
    if label == 3:
        return 3
    return None


def normalized_failure(value: str) -> str:
    value = clean(value)
    return value if value in FAILURE_VOCAB else "unclear"


def preview_text(value: Any, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", clean(value)).strip()[:limit]


def build_maps(rows: list[dict[str, str]], key: str = "sample_id") -> dict[str, dict[str, str]]:
    return {clean(row.get(key)): row for row in rows if clean(row.get(key))}


def reason_from_sources(
    sid: str,
    reason_bundle: dict[str, Any],
    a0_row: dict[str, str] | None,
    high_row: dict[str, str] | None,
) -> tuple[str, str, bool]:
    a0_reason = clean((a0_row or {}).get("recovered_reason_summary"))
    high_reason = clean((high_row or {}).get("recovered_reason_summary"))
    bundle_reason = clean(reason_bundle.get("reason_summary"))
    if a0_reason:
        return a0_reason, "a0_recovered_human_reason", True
    if high_reason:
        return high_reason, "a0_clean_high_recovered_human_reason", True
    if bundle_reason:
        return bundle_reason, "raw_5grades_recovered_human_reason", bool(
            int(reason_bundle.get("label_consistent_reason_count") or 0) > 0
        )
    return "", "missing_human_reason", False


def target_for_row(
    row: dict[str, Any],
    reason: str,
    reason_source: str,
    label_consistent_reason: bool,
    a0_row: dict[str, str] | None,
    high_row: dict[str, str] | None,
) -> dict[str, Any]:
    label = clamp_score(row.get("label_5"))
    if label >= 4:
        failures = ["no_major_failure"]
        risk_flag = "clean_high"
        annotation_status = "weak_high_control"
    elif label == 3:
        failures = ["partial_or_incomplete"]
        risk_flag = "mid_borderline"
        annotation_status = "needs_optional_review"
    else:
        mode = normalized_failure((a0_row or {}).get("failure_mode_auto", ""))
        if mode in {"possible_label_conflict", "unclear"}:
            failures = [mode]
            annotation_status = "needs_teacher_or_human_review"
        else:
            failures = [mode]
            annotation_status = "weak_evidence_ready"
        risk_flag = "hidden_low_failure"

    if reason:
        final_reason = reason
    elif label >= 4:
        final_reason = "No major rubric-linked failure is evident; the answer should remain high-scoring unless a specific hidden defect is found."
    elif label == 3:
        final_reason = "The answer partially satisfies the rubric, but some requirements are incomplete or insufficiently supported."
    else:
        final_reason = "A low score is assigned by human labels, but the specific hidden-failure evidence still requires annotation."

    return {
        "reason": final_reason,
        "major_failures": failures,
        "evidence_span": None,
        "rubric_clause": rubric_clause(row, label),
        "score_cap": score_cap(label),
        "risk_flag": risk_flag,
        "annotation_source": reason_source,
        "annotation_status": annotation_status,
        "label_consistent_human_reason": label_consistent_reason,
        "score": label,
    }


def target_meta(row: dict[str, Any], target: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "split_source": "train",
        "source_sample_id": sample_id(row),
        "source_question_key": question_key(row),
        "gold_label": clamp_score(row.get("label_5")),
        "metric": metric_name(row),
        "language": language(row),
        "subject": subject(row),
        "risk_flag": target["risk_flag"],
        "annotation_source": target["annotation_source"],
        "annotation_status": target["annotation_status"],
        "reason_hash": sha1(reason),
        "human_reason_in_prompt": False,
    }


def high_protection_rejected(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "reason": "A serious hidden failure is incorrectly asserted, so the answer is capped as low-scoring.",
        "major_failures": ["surface_fluent_but_hidden_defect"],
        "evidence_span": None,
        "rubric_clause": rubric_clause(row, 2),
        "score_cap": 2,
        "risk_flag": "false_positive_low_risk",
        "annotation_source": "counterfactual_high_protection_train_only_not_human",
        "annotation_status": "counterfactual_negative",
        "label_consistent_human_reason": False,
        "score": 2,
    }


def low_erasure_rejected(row: dict[str, Any], rejected_score: int = 5) -> dict[str, Any]:
    return {
        "reason": "No major failure is identified; the answer appears to satisfy the rubric.",
        "major_failures": ["no_major_failure"],
        "evidence_span": None,
        "rubric_clause": rubric_clause(row, rejected_score),
        "score_cap": None,
        "risk_flag": "hidden_failure_erased",
        "annotation_source": "counterfactual_failure_erasure_train_only_not_human",
        "annotation_status": "counterfactual_negative",
        "label_consistent_human_reason": False,
        "score": rejected_score,
    }


def score_mismatch_rejected(chosen: dict[str, Any], rejected_score: int) -> dict[str, Any]:
    out = dict(chosen)
    out["score"] = rejected_score
    out["annotation_source"] = "real_model_score_output_same_evidence"
    out["annotation_status"] = "real_score_negative"
    if rejected_score >= 4:
        out["score_cap"] = None
    return out


def parse_int(value: Any) -> int | None:
    try:
        score = int(round(float(value)))
    except Exception:
        return None
    return score if 1 <= score <= 5 else None


def load_r7h_hard_scores(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = clean(row.get("source_sample_id"))
        if not sid:
            continue
        by_sid[sid].append(row)
    return by_sid


def pair_id(meta: dict[str, Any], chosen: dict[str, Any], rejected: dict[str, Any]) -> str:
    return sha1(
        {
            "source_sample_id": meta.get("source_sample_id"),
            "pair_type": meta.get("pair_type"),
            "chosen": chosen,
            "rejected": rejected,
        }
    )


def build_sft_rows(
    train_rows: list[dict[str, Any]],
    targets_by_sid: dict[str, dict[str, Any]],
    reason_by_sid: dict[str, str],
    high_controls: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    low = [row for row in train_rows if clamp_score(row.get("label_5")) <= 2 and clean(reason_by_sid.get(sample_id(row)))]
    mid = [row for row in train_rows if clamp_score(row.get("label_5")) == 3 and clean(reason_by_sid.get(sample_id(row)))]
    high = [
        row
        for row in train_rows
        if clamp_score(row.get("label_5")) >= 4
        and (sample_id(row) in high_controls or clean(reason_by_sid.get(sample_id(row))))
    ]
    rng.shuffle(mid)
    rng.shuffle(high)
    selected = low + mid[: args.max_mid_sft] + high[: args.max_high_sft]
    rng.shuffle(selected)
    out: list[dict[str, Any]] = []
    for row in selected:
        sid = sample_id(row)
        target = targets_by_sid[sid]
        out.append(sft_example(row, target, target_meta(row, target, reason_by_sid.get(sid, ""))))
    return out


def build_dpo_rows(
    train_rows: list[dict[str, Any]],
    targets_by_sid: dict[str, dict[str, Any]],
    reason_by_sid: dict[str, str],
    high_controls: dict[str, dict[str, str]],
    r7h_by_sid: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    row_by_sid = {sample_id(row): row for row in train_rows}
    out: list[dict[str, Any]] = []

    low_rows = [row for row in train_rows if clamp_score(row.get("label_5")) <= 2 and clean(reason_by_sid.get(sample_id(row)))]
    for row in low_rows:
        sid = sample_id(row)
        chosen = targets_by_sid[sid]
        hard_pairs = [
            item
            for item in r7h_by_sid.get(sid, [])
            if clean(item.get("risk_type")) == "low_to_high_real_model_error"
            and parse_int(item.get("rejected_score")) is not None
            and int(item.get("rejected_score")) >= 4
        ]
        if hard_pairs:
            for item in hard_pairs[: args.max_hard_rejected_per_low]:
                rejected_score = int(item["rejected_score"])
                meta = {
                    "split_source": "train",
                    "source_sample_id": sid,
                    "source_question_key": question_key(row),
                    "gold_label": clamp_score(row.get("label_5")),
                    "rejected_score": rejected_score,
                    "pair_type": "real_model_error_score_pair",
                    "negative_source": "real_model_score_output",
                    "risk_type": "low_to_high_real_model_error",
                    "pair_weight": 1.5,
                    "human_reason_in_prompt": False,
                    "rejected_source": clean(item.get("rejected_source")),
                }
                rejected = score_mismatch_rejected(chosen, rejected_score)
                pair = dpo_example(row, chosen, rejected, meta)
                pair["pair_id"] = pair_id(meta, chosen, rejected)
                out.append(pair)

        meta = {
            "split_source": "train",
            "source_sample_id": sid,
            "source_question_key": question_key(row),
            "gold_label": clamp_score(row.get("label_5")),
            "rejected_score": 5,
            "pair_type": "failure_erasure_negative",
            "negative_source": "counterfactual_failure_erasure_train_only_not_human",
            "risk_type": "hidden_failure_erasure_to_high",
            "pair_weight": 2.0,
            "human_reason_in_prompt": False,
            "rejected_source": "counterfactual",
        }
        rejected = low_erasure_rejected(row, 5)
        pair = dpo_example(row, chosen, rejected, meta)
        pair["pair_id"] = pair_id(meta, chosen, rejected)
        out.append(pair)

    high_rows = [
        row
        for row in train_rows
        if clamp_score(row.get("label_5")) >= 4
        and (sample_id(row) in high_controls or clean(reason_by_sid.get(sample_id(row))))
    ]
    rng.shuffle(high_rows)
    for row in high_rows[: args.max_high_dpo]:
        sid = sample_id(row)
        chosen = targets_by_sid[sid]
        meta = {
            "split_source": "train",
            "source_sample_id": sid,
            "source_question_key": question_key(row),
            "gold_label": clamp_score(row.get("label_5")),
            "rejected_score": 2,
            "pair_type": "matched_high_protection_pair",
            "negative_source": "counterfactual_spurious_failure_train_only_not_human",
            "risk_type": "high_protection_against_overconservative_dpo",
            "pair_weight": 0.75,
            "human_reason_in_prompt": False,
            "rejected_source": "counterfactual",
        }
        rejected = high_protection_rejected(row)
        pair = dpo_example(row, chosen, rejected, meta)
        pair["pair_id"] = pair_id(meta, chosen, rejected)
        out.append(pair)

    # Add a bounded score-mismatch consistency layer from existing R7H pairs.
    score_mismatch: list[dict[str, Any]] = []
    for sid, items in r7h_by_sid.items():
        row = row_by_sid.get(sid)
        if row is None or sid not in targets_by_sid:
            continue
        for item in items:
            if clean(item.get("negative_type")) not in {"score_mismatch_same_reason", "high_protection_score_mismatch"}:
                continue
            rejected_score = parse_int(item.get("rejected_score"))
            if rejected_score is None or rejected_score == clamp_score(row.get("label_5")):
                continue
            chosen = targets_by_sid[sid]
            meta = {
                "split_source": "train",
                "source_sample_id": sid,
                "source_question_key": question_key(row),
                "gold_label": clamp_score(row.get("label_5")),
                "rejected_score": rejected_score,
                "pair_type": "score_mismatch_same_evidence",
                "negative_source": "real_model_score_output_same_evidence",
                "risk_type": clean(item.get("risk_type")),
                "pair_weight": 0.5,
                "human_reason_in_prompt": False,
                "rejected_source": clean(item.get("rejected_source")),
            }
            rejected = score_mismatch_rejected(chosen, rejected_score)
            pair = dpo_example(row, chosen, rejected, meta)
            pair["pair_id"] = pair_id(meta, chosen, rejected)
            score_mismatch.append(pair)
    rng.shuffle(score_mismatch)
    out.extend(score_mismatch[: args.max_score_mismatch_dpo])

    # De-duplicate while keeping construction priority.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in out:
        pid = clean(row.get("pair_id"))
        if pid in seen:
            continue
        seen.add(pid)
        deduped.append(row)
    rng.shuffle(deduped)
    return deduped


def annotation_manifest_rows(
    train_rows: list[dict[str, Any]],
    targets_by_sid: dict[str, dict[str, Any]],
    reason_by_sid: dict[str, str],
    reason_source_by_sid: dict[str, str],
    a0_candidates: dict[str, dict[str, str]],
    exp21_by_sid: dict[str, dict[str, str]],
    r7h_by_sid: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in train_rows:
        sid = sample_id(row)
        label = clamp_score(row.get("label_5"))
        target = targets_by_sid[sid]
        hard_count = sum(
            1
            for item in r7h_by_sid.get(sid, [])
            if clean(item.get("risk_type")) == "low_to_high_real_model_error"
        )
        needs = []
        if label <= 2:
            if not reason_by_sid.get(sid):
                needs.append("recover_or_write_reason")
            if target.get("major_failures") in (["unclear"], ["possible_label_conflict"]):
                needs.append("verify_failure_type")
            needs.extend(["evidence_span", "rubric_clause_alignment", "score_cap_check"])
            if hard_count == 0:
                needs.append("on_policy_hard_negative_generation")
        elif label >= 4 and sid in exp21_by_sid:
            needs.append("verify_spurious_risk_flag")
        if not needs and label == 3:
            needs.append("optional_borderline_review")
        if not needs:
            continue
        rows.append(
            {
                "sample_id": sid,
                "question_key": question_key(row),
                "metric": metric_name(row),
                "language": language(row),
                "subject": subject(row),
                "gold_label": label,
                "reason_available": bool(reason_by_sid.get(sid)),
                "reason_source": reason_source_by_sid.get(sid, ""),
                "a0_training_use": clean((a0_candidates.get(sid) or {}).get("recommended_training_use")),
                "a0_failure_mode": clean((a0_candidates.get(sid) or {}).get("failure_mode_auto")),
                "exp21_candidate_type": clean((exp21_by_sid.get(sid) or {}).get("candidate_type")),
                "existing_low_to_high_hard_negative_count": hard_count,
                "current_annotation_status": target.get("annotation_status"),
                "needs_annotation_fields": "|".join(needs),
                "teacher_prompt_priority": "high" if label <= 2 or hard_count else "medium",
                "question_preview": preview_text(row.get("question")),
                "answer_preview": preview_text(row.get("answer")),
                "rubric_preview": preview_text(row.get("rubric")),
            }
        )
    return rows


def dataset_info() -> dict[str, Any]:
    return {
        SFT_NAME: {
            "file_name": SFT_FILE,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": OPENAI_TAGS,
        },
        DPO_NAME: {
            "file_name": DPO_FILE,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {"messages": "messages", "chosen": "chosen", "rejected": "rejected"},
            "tags": OPENAI_TAGS,
        },
    }


def validate_no_prompt_reason(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        payloads = []
        if row.get("messages") and row["messages"][-1].get("role") == "assistant":
            payloads.append(parse_assistant_payload(row["messages"][-1]))
        for key in ("chosen", "rejected"):
            if key in row:
                payloads.append(parse_assistant_payload(row[key]))
        prompt = "\n".join(clean(item.get("content")) for item in row.get("messages", []) if item.get("role") == "user")
        for payload in payloads:
            reason = clean(payload.get("reason"))
            if reason and reason[:80] in prompt:
                count += 1
    return count


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    train_rows = read_jsonl(args.train_jsonl)
    a0_candidates = build_maps(read_csv_if_exists(args.a0_candidates))
    high_controls = build_maps(read_csv_if_exists(args.a0_high_controls))
    exp21_by_sid = build_maps(read_csv_if_exists(args.exp21_candidates))
    reason_bundles, missing_reason_files = build_reason_bundles(train_rows, args.reason_root)
    r7h_by_sid = load_r7h_hard_scores(args.r7h_mixed_json)

    reason_by_sid: dict[str, str] = {}
    reason_source_by_sid: dict[str, str] = {}
    targets_by_sid: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        sid = sample_id(row)
        reason, reason_source, label_consistent = reason_from_sources(
            sid,
            reason_bundles.get(sid, {}),
            a0_candidates.get(sid),
            high_controls.get(sid),
        )
        reason_by_sid[sid] = reason
        reason_source_by_sid[sid] = reason_source
        targets_by_sid[sid] = target_for_row(
            row,
            reason,
            reason_source,
            label_consistent,
            a0_candidates.get(sid),
            high_controls.get(sid),
        )

    sft_rows = build_sft_rows(train_rows, targets_by_sid, reason_by_sid, high_controls, args)
    dpo_rows = build_dpo_rows(train_rows, targets_by_sid, reason_by_sid, high_controls, r7h_by_sid, args)
    manifest = annotation_manifest_rows(
        train_rows,
        targets_by_sid,
        reason_by_sid,
        reason_source_by_sid,
        a0_candidates,
        exp21_by_sid,
        r7h_by_sid,
    )

    write_json_array(args.out_dir / SFT_FILE, sft_rows)
    write_json_array(args.out_dir / DPO_FILE, dpo_rows)
    write_json(args.out_dir / "dataset_info_exp26_snippet.json", dataset_info())
    write_json(args.out_dir / "dataset_info.json", dataset_info())

    dev_ids, dev_qkeys, dev_n = split_identity(args.dev_jsonl)
    test_ids, test_qkeys, test_n = split_identity(args.test_jsonl)
    sft_ids = {clean(row.get("source_sample_id")) for row in sft_rows}
    dpo_ids = {clean(row.get("source_sample_id")) for row in dpo_rows}
    sft_qkeys = {clean(row.get("source_question_key")) for row in sft_rows}
    dpo_qkeys = {clean(row.get("source_question_key")) for row in dpo_rows}

    label_counts = Counter(str(clamp_score(row.get("label_5"))) for row in train_rows)
    sft_risk_counts = Counter(clean(row.get("risk_flag")) for row in sft_rows)
    dpo_pair_counts = Counter(clean(row.get("pair_type")) for row in dpo_rows)
    reason_counts = Counter(reason_source_by_sid.values())
    annotation_counts = Counter(clean(row.get("current_annotation_status")) for row in manifest)

    write_csv(
        args.out_dir / "tables" / "exp26_dataset_counts.csv",
        [
            {"name": "train_rows", "count": len(train_rows)},
            {"name": "sft_rows", "count": len(sft_rows)},
            {"name": "dpo_pairs", "count": len(dpo_rows)},
            {"name": "annotation_manifest_rows", "count": len(manifest)},
            {"name": "train_low_label_rows", "count": sum(1 for row in train_rows if clamp_score(row.get("label_5")) <= 2)},
            {"name": "train_low_with_reason_rows", "count": sum(1 for row in train_rows if clamp_score(row.get("label_5")) <= 2 and reason_by_sid.get(sample_id(row)))},
        ],
    )
    write_csv(
        args.out_dir / "tables" / "exp26_label_distribution.csv",
        [{"label": key, "count": value} for key, value in sorted(label_counts.items())],
    )
    write_csv(
        args.out_dir / "tables" / "exp26_sft_risk_distribution.csv",
        [{"risk_flag": key, "count": value} for key, value in sft_risk_counts.most_common()],
    )
    write_csv(
        args.out_dir / "tables" / "exp26_dpo_pair_type_distribution.csv",
        [{"pair_type": key, "count": value} for key, value in dpo_pair_counts.most_common()],
    )
    write_csv(
        args.out_dir / "tables" / "exp26_reason_source_distribution.csv",
        [{"reason_source": key, "count": value} for key, value in reason_counts.most_common()],
    )
    write_csv(
        args.out_dir / "tables" / "exp26_annotation_status_distribution.csv",
        [{"annotation_status": key, "count": value} for key, value in annotation_counts.most_common()],
    )
    write_csv(args.out_dir / "annotation" / "exp26_hidden_failure_annotation_manifest.csv", manifest)
    leakage_rows = [
        {
            "dataset": SFT_NAME,
            "dev_rows_read_for_id_guard": dev_n,
            "test_rows_read_for_id_guard": test_n,
            "dev_sample_overlap": len(sft_ids & dev_ids),
            "dev_question_overlap": len(sft_qkeys & dev_qkeys),
            "test_sample_overlap": len(sft_ids & test_ids),
            "test_question_overlap": len(sft_qkeys & test_qkeys),
            "human_reason_in_prompt_count": validate_no_prompt_reason(sft_rows),
            "test_label_read": False,
        },
        {
            "dataset": DPO_NAME,
            "dev_rows_read_for_id_guard": dev_n,
            "test_rows_read_for_id_guard": test_n,
            "dev_sample_overlap": len(dpo_ids & dev_ids),
            "dev_question_overlap": len(dpo_qkeys & dev_qkeys),
            "test_sample_overlap": len(dpo_ids & test_ids),
            "test_question_overlap": len(dpo_qkeys & test_qkeys),
            "human_reason_in_prompt_count": validate_no_prompt_reason(dpo_rows),
            "test_label_read": False,
        },
    ]
    write_csv(args.out_dir / "tables" / "exp26_leakage_audit.csv", leakage_rows)

    decision = {
        "recommendation": "review_exp26_data_before_training",
        "sft_dataset": SFT_NAME,
        "dpo_dataset": DPO_NAME,
        "sft_rows": len(sft_rows),
        "dpo_pairs": len(dpo_rows),
        "annotation_manifest_rows": len(manifest),
        "low_train_rows": sum(1 for row in train_rows if clamp_score(row.get("label_5")) <= 2),
        "low_train_rows_with_reason": sum(
            1 for row in train_rows if clamp_score(row.get("label_5")) <= 2 and reason_by_sid.get(sample_id(row))
        ),
        "needs_teacher_or_human_annotation": sum(
            1 for row in manifest if "evidence_span" in clean(row.get("needs_annotation_fields"))
        ),
        "missing_reason_files": missing_reason_files,
        "guardrails": {
            "train_only_construction": True,
            "dev_test_used_for_id_guard_only": True,
            "test_label_read": False,
            "human_reason_in_prompt": False,
            "teacher_annotation_already_used_as_gold": False,
        },
    }
    write_json(args.out_dir / "decision" / "exp26_hidden_failure_expansion_decision.json", decision)

    report = [
        "# Exp26 Hidden-Failure Evidence Expansion Data",
        "",
        "Exp26A prepares train-only data for the next evidence-aware SFT and field-masked ORC/SRC-DPO round.",
        "",
        "## Outputs",
        "",
        f"- SFT dataset: `{SFT_FILE}` ({len(sft_rows)} rows)",
        f"- DPO dataset: `{DPO_FILE}` ({len(dpo_rows)} pairs)",
        f"- annotation manifest: `annotation/exp26_hidden_failure_annotation_manifest.csv` ({len(manifest)} rows)",
        "",
        "## Important Scope",
        "",
        "- This step does not train a model and does not require GPU.",
        "- Dev/test are read only for sample_id/question_key leakage guards.",
        "- Human rationales are included only in assistant targets, never in user prompts.",
        "- `evidence_span` is intentionally null in this first data asset; the manifest marks rows needing teacher/human annotation.",
        "- Counterfactual rejected outputs are explicitly marked as train-only non-human negatives.",
        "",
        "## Dataset Counts",
        "",
        f"- train rows: {len(train_rows)}",
        f"- train low-label rows: {decision['low_train_rows']}",
        f"- train low-label rows with recovered reason: {decision['low_train_rows_with_reason']}",
        f"- SFT risk distribution: {dict(sft_risk_counts)}",
        f"- DPO pair type distribution: {dict(dpo_pair_counts)}",
        "",
        "## Recommendation",
        "",
        "Review this data before training. The next likely step is teacher/human annotation for high-priority low-label rows, "
        "then evidence-aware SFT, then field-masked ORC/SRC-DPO.",
    ]
    write_text(args.out_dir / "reports" / "exp26_hidden_failure_expansion_report.md", "\n".join(report))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Exp26 hidden-failure evidence expansion data.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_JSONL)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--reason-root", type=Path, default=DEFAULT_REASON_ROOT)
    parser.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    parser.add_argument("--a0-high-controls", type=Path, default=DEFAULT_A0_HIGH_CONTROLS)
    parser.add_argument("--exp21-candidates", type=Path, default=DEFAULT_EXP21_CANDIDATES)
    parser.add_argument("--r7h-mixed-json", type=Path, default=DEFAULT_R7H_MIXED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-mid-sft", type=int, default=120)
    parser.add_argument("--max-high-sft", type=int, default=360)
    parser.add_argument("--max-high-dpo", type=int, default=240)
    parser.add_argument("--max-score-mismatch-dpo", type=int, default=600)
    parser.add_argument("--max-hard-rejected-per-low", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
