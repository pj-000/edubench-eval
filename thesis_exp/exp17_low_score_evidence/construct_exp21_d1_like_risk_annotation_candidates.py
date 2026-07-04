"""Construct Exp21 D1-like hidden-failure risk annotation candidates.

Exp21 is an annotation-candidate construction step. It does not train, does not
read test, and does not use human rationale as model input. Dev cases are for
diagnostic audit only; train candidates are intended for future manual
annotation and train-side risk-gate / DPO data construction.
"""

from __future__ import annotations

import argparse
import csv
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

from thesis_exp.exp17_low_score_evidence import collect_exp19_sft_first_round_dev_results as sft_collect  # noqa: E402
from thesis_exp.exp17_low_score_evidence.evaluate_exp20_dual_model_risk_gate import (  # noqa: E402
    DEFAULT_D1_DIR,
    DEFAULT_RISK_RUNS,
    DEFAULT_SCORE_RUNS,
    load_run,
    read_csv_rows as read_reference_csv,
    safe_int,
)
from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    MODEL_PATH,
    OPENAI_TAGS,
    clamp_score,
    clean,
    language,
    messages_for,
    metric_name,
    question_group,
    read_jsonl,
    sample_id,
    sft_dataset_entry,
    subject,
)
from thesis_exp.exp17_low_score_evidence.run_exp19_r0a_qwen4b_direct_baseline import safe_rate  # noqa: E402
from thesis_exp.src.edujudge.utils.io import write_csv, write_json, write_jsonl, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp21_d1_like_risk_annotation_seed42")
DEFAULT_DEV_REFERENCE = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42/tables/exp19_dev_reference.csv"
)
DEFAULT_TRAIN_JSONL = Path("thesis_exp/data/splits/question_seed42/train.jsonl")
DEFAULT_DEV_JSONL = Path("thesis_exp/data/splits/question_seed42/dev.jsonl")
DEFAULT_EXP20C_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp20c_best_rule_case_audit_seed42")

SCORE_RUN_NAME = "r5g_a3_real_only_s50_b0p05_lr5em6"
RISK_RUN_NAME = "r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6"
DEFAULT_TRAIN_SCORE_PRED_DIR = DEFAULT_OUT_DIR / "train_predictions" / SCORE_RUN_NAME
DEFAULT_TRAIN_RISK_PRED_DIR = DEFAULT_OUT_DIR / "train_predictions" / RISK_RUN_NAME
DEFAULT_SCORE_CONFIG = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5g_risk_calibrated_dpo_seed42/configs/"
    "llamafactory_qwen3_4b_r5g_a3_real_only_s50_b0p05_lr5em6.yaml"
)
DEFAULT_RISK_CONFIG = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp19_r5h_two_stage_dpo_seed42/configs/"
    "llamafactory_qwen3_4b_r5h_h1_from_r5f2_real_highprotect_s10_b0p02_lr1e6.yaml"
)
TRAIN_PRED_DATASET = "edubench_exp21_train_score_eval"
TRAIN_PRED_FILE = "edubench_exp21_train_score_eval.json"
PREVIEW_CHARS = 260

MANUAL_FIELDS = [
    "manual_should_flag",
    "manual_final_action",
    "manual_failure_type",
    "manual_score_cap",
    "manual_label_conflict",
    "manual_notes",
]
DEV_AUDIT_FIELDS = [
    "sample_id",
    "split",
    "case_source",
    "question_key",
    "metric",
    "language",
    "subject",
    "gold_label",
    "score_model_pred",
    "risk_model_pred",
    "final_pred_after_rule",
    "gate_flag",
    "audit_bucket",
    "is_d1_hidden",
    "baseline_low_to_high",
    "final_low_to_high",
    "rescued_low_to_high",
    "gold_high_downgraded_to_3",
    "question_preview",
    "answer_preview",
    "rubric_preview",
    *MANUAL_FIELDS,
]
TRAIN_CANDIDATE_FIELDS = [
    "sample_id",
    "split",
    "candidate_type",
    "question_key",
    "metric",
    "language",
    "subject",
    "gold_label",
    "score_model_pred",
    "risk_model_pred",
    "score_risk_gap",
    "question_preview",
    "answer_preview",
    "rubric_preview",
    *MANUAL_FIELDS,
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stable_sample_id(row: dict[str, Any], idx: int) -> str:
    sid = sample_id(row)
    return sid if sid else f"row_{idx:05d}"


def row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {stable_sample_id(row, idx): row for idx, row in enumerate(rows)}


def label(row: dict[str, Any]) -> int:
    return clamp_score(row.get("label_5") or row.get("label") or row.get("gold_label"))


def preview(value: Any, max_chars: int = PREVIEW_CHARS) -> str:
    text = clean(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def prompt_prediction_record(row: dict[str, Any]) -> dict[str, Any]:
    return {"messages": messages_for(row) + [{"role": "assistant", "content": "{}"}]}


def reference_for_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        refs.append(
            {
                "eval_index": str(idx),
                "split": split,
                "sample_id": stable_sample_id(row, idx),
                "record_id": clean(row.get("record_id")),
                "question_key": clean(row.get("question_key")),
                "question_group_id": question_group(row),
                "metric": metric_name(row),
                "metric_id": clean(row.get("metric_id")),
                "language": language(row),
                "subject": subject(row),
                "gold_label": str(label(row)),
            }
        )
    return refs


def parse_output_dir_from_yaml(path: Path) -> str:
    if not path.exists():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("output_dir:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def prediction_file_exists(run_dir: Path) -> bool:
    try:
        sft_collect.find_prediction_file(run_dir)
        return True
    except FileNotFoundError:
        return False


def load_prediction_rows(
    name: str,
    run_dir: Path,
    reference: list[dict[str, str]],
    allow_missing: bool,
) -> list[dict[str, Any]] | None:
    rows = load_run(name, run_dir, reference, allow_missing)
    return rows


def to_dev_audit_row(case: dict[str, str], full_row: dict[str, Any] | None, case_source: str) -> dict[str, Any]:
    full_row = full_row or {}
    return {
        "sample_id": clean(case.get("sample_id")),
        "split": "dev",
        "case_source": case_source,
        "question_key": clean(case.get("question_key") or full_row.get("question_key")),
        "metric": clean(case.get("metric") or metric_name(full_row)),
        "language": clean(case.get("language") or language(full_row)),
        "subject": clean(case.get("subject") or subject(full_row)),
        "gold_label": clean(case.get("gold_label") or label(full_row)),
        "score_model_pred": clean(case.get("score_pred")),
        "risk_model_pred": clean(case.get("risk_pred")),
        "final_pred_after_rule": clean(case.get("final_pred")),
        "gate_flag": clean(case.get("flagged")),
        "audit_bucket": clean(case.get("audit_bucket") or case_source),
        "is_d1_hidden": clean(case.get("is_d1_hidden")),
        "baseline_low_to_high": clean(case.get("baseline_low_to_high")),
        "final_low_to_high": clean(case.get("final_low_to_high")),
        "rescued_low_to_high": clean(case.get("rescued_low_to_high")),
        "gold_high_downgraded_to_3": clean(case.get("gold_high_downgraded_to_3")),
        "question_preview": preview(full_row.get("question")),
        "answer_preview": preview(full_row.get("answer")),
        "rubric_preview": preview(full_row.get("rubric")),
        **{field: "" for field in MANUAL_FIELDS},
    }


def stratified_take(
    rows: list[dict[str, Any]],
    limit: int,
    seed: int,
    key_fields: tuple[str, ...] = ("metric", "language", "subject"),
) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "|".join(clean(row.get(field)) for field in key_fields)
        groups[key].append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while keys and len(selected) < limit:
        next_keys: list[str] = []
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def construct_dev_audit(
    exp20c_dir: Path,
    dev_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dev_by_id = row_map(dev_rows)
    changed_rows = read_csv(exp20c_dir / "tables" / "exp20c_residual_and_changed_cases.csv")
    flagged_rows = read_csv(exp20c_dir / "tables" / "exp20c_downgrade_case_audit.csv")
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in changed_rows + flagged_rows:
        bucket = clean(row.get("audit_bucket"))
        if not bucket:
            continue
        buckets[bucket].append(row)

    sampled: list[dict[str, Any]] = []
    seen: set[str] = set()
    bucket_limits = {
        "rescued_d1_low_to_high": 25,
        "rescued_low_to_high": 25,
        "residual_d1_pred_ge4": 50,
        "residual_low_to_high": 40,
        "downgraded_gold_high_to_3": 60,
        "downgraded_d1_gold_high_to_3": 20,
    }
    for bucket, limit in bucket_limits.items():
        rows = stratified_take(buckets.get(bucket, []), limit, seed + len(sampled))
        for case in rows:
            sid = clean(case.get("sample_id"))
            if sid in seen:
                continue
            sampled.append(to_dev_audit_row(case, dev_by_id.get(sid), bucket))
            seen.add(sid)

    safe_controls: list[dict[str, Any]] = []
    for idx, row in enumerate(dev_rows):
        sid = stable_sample_id(row, idx)
        if sid in seen:
            continue
        gold = label(row)
        if gold >= 4:
            safe_controls.append(
                {
                    "sample_id": sid,
                    "question_key": clean(row.get("question_key")),
                    "metric": metric_name(row),
                    "language": language(row),
                    "subject": subject(row),
                    "gold_label": gold,
                    "score_pred": "",
                    "risk_pred": "",
                    "final_pred": "",
                    "flagged": "False",
                    "audit_bucket": "unchanged_safe_high_control",
                    "is_d1_hidden": "False",
                    "baseline_low_to_high": "False",
                    "final_low_to_high": "False",
                    "rescued_low_to_high": "False",
                    "gold_high_downgraded_to_3": "False",
                }
            )
    for case in stratified_take(safe_controls, 40, seed + 91):
        sid = clean(case.get("sample_id"))
        sampled.append(to_dev_audit_row(case, dev_by_id.get(sid), "unchanged_safe_high_control"))

    counts = Counter(row["case_source"] for row in sampled)
    return sampled, dict(sorted(counts.items()))


def train_candidate_row(
    row: dict[str, Any],
    idx: int,
    candidate_type: str,
    score_pred: int | None,
    risk_pred: int | None,
) -> dict[str, Any]:
    gap = "" if score_pred is None or risk_pred is None else score_pred - risk_pred
    return {
        "sample_id": stable_sample_id(row, idx),
        "split": "train",
        "candidate_type": candidate_type,
        "question_key": clean(row.get("question_key")),
        "metric": metric_name(row),
        "language": language(row),
        "subject": subject(row),
        "gold_label": label(row),
        "score_model_pred": score_pred if score_pred is not None else "",
        "risk_model_pred": risk_pred if risk_pred is not None else "",
        "score_risk_gap": gap,
        "question_preview": preview(row.get("question")),
        "answer_preview": preview(row.get("answer")),
        "rubric_preview": preview(row.get("rubric")),
        **{field: "" for field in MANUAL_FIELDS},
    }


def construct_train_candidates(
    train_rows: list[dict[str, Any]],
    score_preds: list[dict[str, Any]],
    risk_preds: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    score_by_id = {clean(row.get("sample_id")): row for row in score_preds}
    risk_by_id = {clean(row.get("sample_id")): row for row in risk_preds}
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(train_rows):
        sid = stable_sample_id(row, idx)
        score_row = score_by_id.get(sid)
        risk_row = risk_by_id.get(sid)
        if score_row is None or risk_row is None:
            continue
        score_pred = safe_int(score_row.get("pred_label"))
        risk_pred = safe_int(risk_row.get("pred_label"))
        if score_pred is None or risk_pred is None:
            continue
        gold = label(row)
        gap = score_pred - risk_pred
        if gold <= 2 and (score_pred >= 4 or gap >= 2):
            pools["train_high_risk_low_candidates"].append(
                train_candidate_row(row, idx, "train_high_risk_low_candidates", score_pred, risk_pred)
            )
        if gold >= 4 and score_pred >= 4 and risk_pred <= 3:
            pools["train_high_false_positive_candidates"].append(
                train_candidate_row(row, idx, "train_high_false_positive_candidates", score_pred, risk_pred)
            )
        if gold == 3 and abs(gap) >= 1:
            pools["train_mid_borderline_candidates"].append(
                train_candidate_row(row, idx, "train_mid_borderline_candidates", score_pred, risk_pred)
            )
        if gold >= 4 and score_pred >= 4 and risk_pred >= 4:
            pools["train_clean_high_controls"].append(
                train_candidate_row(row, idx, "train_clean_high_controls", score_pred, risk_pred)
            )

    targets = {
        "train_high_risk_low_candidates": 80,
        "train_high_false_positive_candidates": 80,
        "train_mid_borderline_candidates": 50,
        "train_clean_high_controls": 50,
    }
    selected: list[dict[str, Any]] = []
    for offset, (candidate_type, limit) in enumerate(targets.items()):
        selected.extend(stratified_take(pools.get(candidate_type, []), limit, seed + offset))
    counts = {key: len(pools.get(key, [])) for key in targets}
    counts.update({f"sampled_{key}": sum(1 for row in selected if row["candidate_type"] == key) for key in targets})
    return selected, counts


def metric_language_summary(rows: list[dict[str, Any]], top_k: int = 8) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    metric_counts = Counter(clean(row.get("metric")) for row in rows)
    language_counts = Counter(clean(row.get("language")) for row in rows)
    return metric_counts.most_common(top_k), language_counts.most_common(top_k)


def write_prediction_assets(
    out_dir: Path,
    train_rows: list[dict[str, Any]],
    score_config_path: Path,
    risk_config_path: Path,
    model_path: str,
    per_device_eval_batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    data_dir = out_dir / "train_prediction_data"
    config_dir = out_dir / "train_predict_configs"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    prediction_data = [prompt_prediction_record(row) for row in train_rows]
    write_json(data_dir / TRAIN_PRED_FILE, prediction_data)
    dataset_info = {TRAIN_PRED_DATASET: sft_dataset_entry(TRAIN_PRED_FILE)}
    if dataset_info[TRAIN_PRED_DATASET].get("tags") != OPENAI_TAGS:
        raise ValueError("Unexpected LLaMA-Factory dataset tag schema.")
    write_json(data_dir / "dataset_info.json", dataset_info)
    write_json(out_dir / "decision" / "dataset_info_exp21_train_prediction_snippet.json", dataset_info)

    score_adapter = parse_output_dir_from_yaml(score_config_path)
    risk_adapter = parse_output_dir_from_yaml(risk_config_path)
    configs: dict[str, dict[str, str]] = {
        SCORE_RUN_NAME: {
            "adapter": score_adapter,
            "output_dir": str(out_dir / "train_predictions" / SCORE_RUN_NAME),
        },
        RISK_RUN_NAME: {
            "adapter": risk_adapter,
            "output_dir": str(out_dir / "train_predictions" / RISK_RUN_NAME),
        },
    }
    for run_name, config in configs.items():
        yaml_text = "\n".join(
            [
                f"model_name_or_path: {model_path}",
                f"adapter_name_or_path: {config['adapter']}",
                "trust_remote_code: true",
                "stage: sft",
                "do_predict: true",
                "finetuning_type: lora",
                "infer_backend: huggingface",
                f"dataset_dir: {data_dir}",
                f"eval_dataset: {TRAIN_PRED_DATASET}",
                "template: qwen3_nothink",
                "cutoff_len: 4096",
                "overwrite_cache: true",
                "preprocessing_num_workers: 16",
                f"output_dir: {config['output_dir']}",
                "overwrite_output_dir: true",
                "predict_with_generate: true",
                f"per_device_eval_batch_size: {per_device_eval_batch_size}",
                f"max_new_tokens: {max_new_tokens}",
                "do_sample: false",
                "temperature: 0.0",
                "top_p: 1.0",
                "num_beams: 1",
                "repetition_penalty: 1.05",
                "bf16: true",
                "bf16_full_eval: true",
                "report_to: none",
                "",
            ]
        )
        (config_dir / f"{run_name}.yaml").write_text(yaml_text, encoding="utf-8")
    return {
        "prediction_dataset": str(data_dir / TRAIN_PRED_FILE),
        "dataset_info": str(data_dir / "dataset_info.json"),
        "score_adapter": score_adapter,
        "risk_adapter": risk_adapter,
        "score_config": str(config_dir / f"{SCORE_RUN_NAME}.yaml"),
        "risk_config": str(config_dir / f"{RISK_RUN_NAME}.yaml"),
    }


def write_report(
    out_dir: Path,
    dev_counts: dict[str, int],
    train_predictions_available: bool,
    train_counts: dict[str, int],
    train_candidates: list[dict[str, Any]],
    prediction_assets: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    train_metrics, train_languages = metric_language_summary(train_candidates)
    lines = [
        "# Exp21 D1-like Risk Annotation Candidate Sampling",
        "",
        "Exp21 constructs annotation candidates only. It does not train, does not read test,",
        "and does not use human rationale as model input.",
        "",
        "## Exp20C Failure Modes Observed",
        "",
        "- Automatic downgrade reduced aggregate low-to-high but demoted many true high-score cases.",
        "- D1 hidden residual high-score predictions remain high, so the gate lacks enough precision/recall.",
        "- The candidate package therefore includes both likely high-risk low cases and high-score false positives.",
        "",
        "## Dev Audit Package",
        "",
    ]
    for key, value in sorted(dev_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Train Prediction Availability",
            "",
            f"- train_predictions_available: {train_predictions_available}",
        ]
    )
    if not train_predictions_available:
        lines.extend(
            [
                "- Train score/risk predictions are missing, so Exp21 generated LLaMA-Factory prediction assets.",
                f"- prediction_dataset: `{prediction_assets.get('prediction_dataset', '')}`",
                f"- score_config: `{prediction_assets.get('score_config', '')}`",
                f"- risk_config: `{prediction_assets.get('risk_config', '')}`",
                "- Run `./thesis_exp/scripts/run_exp21_train_predictions.sh`, then rerun this Exp21 constructor.",
            ]
        )
    lines.extend(["", "## Train Annotation Candidates", ""])
    if train_counts:
        for key, value in sorted(train_counts.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No train candidates sampled yet because train predictions are missing.")
    lines.extend(["", "## Dominant Metrics / Languages", ""])
    if train_candidates:
        lines.append("- metrics: " + ", ".join(f"{name}={count}" for name, count in train_metrics))
        lines.append("- languages: " + ", ".join(f"{name}={count}" for name, count in train_languages))
    else:
        lines.append("- Not available until train prediction mining is complete.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- dev_audit_ready: {decision['dev_audit_ready']}",
            f"- train_annotation_ready: {decision['train_annotation_ready']}",
            f"- need_train_prediction_generation: {decision['need_train_prediction_generation']}",
            f"- recommended_manual_annotation_count: {decision['recommended_manual_annotation_count']}",
            f"- next_step: `{decision['next_step']}`",
            "",
            "## Guardrails",
            "",
            "- Test split is not read.",
            "- No model training is performed.",
            "- Dev cases are diagnostic only and must not be used as training labels.",
            "- Full train prediction prompts/configs are generated under ignored output folders.",
        ]
    )
    write_text(out_dir / "reports" / "exp21_candidate_sampling_report.md", "\n".join(lines))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    train_rows = read_jsonl(args.train_jsonl)
    dev_rows = read_jsonl(args.dev_jsonl)
    out_dir = args.out_dir
    dev_audit, dev_counts = construct_dev_audit(args.exp20c_dir, dev_rows, args.seed)
    write_csv(out_dir / "dev_audit" / "exp21_dev_gate_failure_audit.csv", dev_audit, DEV_AUDIT_FIELDS)

    train_reference = reference_for_rows(train_rows, "train")
    train_score_available = prediction_file_exists(args.train_score_prediction_dir)
    train_risk_available = prediction_file_exists(args.train_risk_prediction_dir)
    train_predictions_available = train_score_available and train_risk_available
    prediction_assets: dict[str, Any] = {}
    train_candidates: list[dict[str, Any]] = []
    train_counts: dict[str, int] = {}
    if train_predictions_available:
        score_rows = load_prediction_rows(
            SCORE_RUN_NAME,
            args.train_score_prediction_dir,
            train_reference,
            allow_missing=False,
        )
        risk_rows = load_prediction_rows(
            RISK_RUN_NAME,
            args.train_risk_prediction_dir,
            train_reference,
            allow_missing=False,
        )
        assert score_rows is not None
        assert risk_rows is not None
        train_candidates, train_counts = construct_train_candidates(train_rows, score_rows, risk_rows, args.seed)
    else:
        prediction_assets = write_prediction_assets(
            out_dir,
            train_rows,
            args.score_config,
            args.risk_config,
            args.model_name_or_path,
            args.per_device_eval_batch_size,
            args.max_new_tokens,
        )
    write_csv(
        out_dir / "train_candidates" / "exp21_train_risk_annotation_candidates.csv",
        train_candidates,
        TRAIN_CANDIDATE_FIELDS,
    )

    decision = {
        "train_predictions_available": train_predictions_available,
        "dev_audit_ready": bool(dev_audit),
        "train_annotation_ready": bool(train_candidates),
        "need_train_prediction_generation": not train_predictions_available,
        "recommended_manual_annotation_count": min(250, max(0, len(train_candidates))),
        "next_step": "run_train_predictions" if not train_predictions_available else "manual_annotation",
        "dev_audit_counts": dev_counts,
        "train_candidate_counts": train_counts,
        "prediction_assets": prediction_assets,
        "guardrails": {
            "no_test_read": True,
            "no_training": True,
            "dev_cases_for_diagnostic_only": True,
            "human_rationale_not_used_as_model_input": True,
        },
    }
    write_json(out_dir / "decision" / "exp21_candidate_sampling_decision.json", decision)
    write_report(out_dir, dev_counts, train_predictions_available, train_counts, train_candidates, prediction_assets, decision)
    return {
        "dev_audit_count": len(dev_audit),
        "train_predictions_available": train_predictions_available,
        "train_candidate_count": len(train_candidates),
        "need_train_prediction_generation": not train_predictions_available,
        "next_step": decision["next_step"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct Exp21 D1-like risk annotation candidates.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dev-reference", type=Path, default=DEFAULT_DEV_REFERENCE)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--dev-jsonl", type=Path, default=DEFAULT_DEV_JSONL)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--exp20c-dir", type=Path, default=DEFAULT_EXP20C_DIR)
    parser.add_argument("--train-score-prediction-dir", type=Path, default=DEFAULT_TRAIN_SCORE_PRED_DIR)
    parser.add_argument("--train-risk-prediction-dir", type=Path, default=DEFAULT_TRAIN_RISK_PRED_DIR)
    parser.add_argument("--score-config", type=Path, default=DEFAULT_SCORE_CONFIG)
    parser.add_argument("--risk-config", type=Path, default=DEFAULT_RISK_CONFIG)
    parser.add_argument("--model-name-or-path", default=MODEL_PATH)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Keep dev reference and D1 dir as explicit CLI inputs for auditability even
    # though the current candidate construction relies on Exp20C outputs and
    # dev/train split rows for previews.
    _ = args.dev_reference
    _ = args.d1_dir
    print(json.dumps(evaluate(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
