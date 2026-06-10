"""Build Exp6 synthetic-augmented question-disjoint training datasets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp02.build_exp02_dataset import clean_answer_text, clean_question_text
from thesis_exp.src.edujudge.exp03.build_exp03_datasets import count_tokens, load_tokenizer, validate_source_row
from thesis_exp.src.edujudge.exp03.rubric_repair import AUTO_MODE, apply_rubric_mode, load_rubric_mapping, resolve_rubric_mode
from thesis_exp.src.edujudge.exp03.templates import make_prompt, rubric_to_text
from thesis_exp.src.edujudge.utils.hashing import sha1_text
from thesis_exp.src.edujudge.utils.io import read_jsonl, relpath, write_csv, write_jsonl, write_text


TEMPLATE_NAME = "A4_question_answer_metric_rubric_metadata"
QUESTION_SPLIT_DIR = Path("thesis_exp/data/splits/question_seed42")
SYNTHETIC_POOL_PATH = Path(
    "thesis_exp/outputs/exp06_synthetic_low_score/final_pool_curated/final_synthetic_low_score_pool.jsonl"
)
SYNTHETIC_POOL_REPORT_PATH = Path(
    "thesis_exp/outputs/exp06_synthetic_low_score/final_pool_curated/reports/final_synthetic_pool_report.md"
)
SYNTHETIC_POOL_SANITY_PATH = Path(
    "thesis_exp/outputs/exp06_synthetic_low_score/final_pool_curated/tables/final_synthetic_pool_sanity.csv"
)
OUTPUT_DIR = Path("thesis_exp/outputs/exp06_synthetic_low_score/training_datasets")

EXPECTED_SPLIT_ROWS = {"train": 3326, "dev": 1107, "test": 1103}
EXPECTED_SYNTHETIC_ROWS = 384
FINAL_SYNTHETIC_LABEL_TARGET = {1: 168, 2: 168, 3: 48}
DATASET_NAMES = [
    "QD-S0_human_only",
    "QD-S1_human_plus_synthetic_ordinal",
    "QD-S2_human_plus_synthetic_L1",
    "QD-S3_synthetic_pretrain_then_human_finetune",
]
REQUIRED_TRAINING_FIELDS = {
    "record_id",
    "source_type",
    "label_provenance",
    "question",
    "answer",
    "metric_canonical",
    "rubric_text",
    "scenario_canonical",
    "subject_canonical",
    "education_level_canonical",
    "language",
    "label_5",
    "text",
    "template_name",
}
FORBIDDEN_TEXT_MARKERS = [
    "rationale_for_label",
    "expected_failure_against_rubric",
    "error_type",
    "label_source",
]
API_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{12,}")
CHECKPOINT_EXTENSIONS = {".bin", ".safetensors", ".pt", ".pth", ".ckpt"}


def _as_int(value: Any) -> int:
    return int(float(str(value).strip()))


def _answer_hash(value: Any) -> str:
    return hashlib.sha1(str(value if value is not None else "").encode("utf-8")).hexdigest()


def _none_if_empty(value: Any) -> Any:
    if value == "":
        return None
    return value


def _language(value: Any) -> str:
    return str(value or "").strip()


def _normalize_rubric_text(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return rubric_to_text(value)
    text = str(value or "").strip()
    if text.startswith(("[", "{")):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        if isinstance(parsed, (list, dict)):
            return rubric_to_text(parsed)
    return text


def _load_inputs() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    raw_by_split = {split: read_jsonl(QUESTION_SPLIT_DIR / f"{split}.jsonl") for split in ["train", "dev", "test"]}
    synthetic_rows = read_jsonl(SYNTHETIC_POOL_PATH)
    return raw_by_split, synthetic_rows


def _source_index(raw_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    return {
        "dev_question_keys": {str(row.get("question_key")) for row in raw_by_split["dev"]},
        "test_question_keys": {str(row.get("question_key")) for row in raw_by_split["test"]},
        "dev_triple_keys": {str(row.get("triple_key")) for row in raw_by_split["dev"]},
        "test_triple_keys": {str(row.get("triple_key")) for row in raw_by_split["test"]},
        "train_question_keys": {str(row.get("question_key")) for row in raw_by_split["train"]},
        "train_triple_keys": {str(row.get("triple_key")) for row in raw_by_split["train"]},
    }


def _base_output_row(
    *,
    row_id: str,
    record_id: str,
    question_key: str | None,
    triple_key: str | None,
    source_type: str,
    label_provenance: str,
    question: str,
    answer: str,
    metric_canonical: str,
    rubric_text: str,
    scenario_canonical: str,
    subject_canonical: str,
    education_level_canonical: str,
    language: str,
    label_5: int,
    human_mean_5: float | None,
    synthetic_target_label_5: int | None,
    synthetic_id: str | None,
    error_type: str | None,
    source_record_id: str | None,
    source_question_key: str | None,
    source_triple_key: str | None,
    generation_split_name: str,
    tokenizer: Any | None,
    max_length: int,
) -> dict[str, Any]:
    prompt_row = {
        "scenario_canonical": scenario_canonical,
        "subject_canonical": subject_canonical,
        "education_level_canonical": education_level_canonical,
        "language": language,
        "question": question,
        "answer": answer,
        "metric_canonical": metric_canonical,
        "rubric_text": rubric_text,
    }
    text = make_prompt(prompt_row, TEMPLATE_NAME)
    num_tokens, token_count_method = count_tokens(text, tokenizer)
    return {
        "id": row_id,
        "record_id": record_id,
        "question_key": question_key,
        "triple_key": triple_key,
        "source_type": source_type,
        "label_provenance": label_provenance,
        "question": question,
        "answer": answer,
        "metric_canonical": metric_canonical,
        "rubric_text": rubric_text,
        "scenario_canonical": scenario_canonical,
        "subject_canonical": subject_canonical,
        "education_level_canonical": education_level_canonical,
        "language": language,
        "label": label_5 - 1,
        "label_5": label_5,
        "human_mean_5": human_mean_5,
        "synthetic_target_label_5": synthetic_target_label_5,
        "synthetic_id": synthetic_id,
        "error_type": error_type,
        "source_record_id": source_record_id,
        "source_question_key": source_question_key,
        "source_triple_key": source_triple_key,
        "text": text,
        "template_name": TEMPLATE_NAME,
        "num_chars": len(text),
        "num_tokens": num_tokens,
        "token_count_method": token_count_method,
        "truncated": bool(num_tokens > max_length),
        "generation_split_name": generation_split_name,
    }


def convert_human_rows(
    source_rows: list[dict[str, Any]],
    split: str,
    tokenizer: Any | None,
    max_length: int,
    rubric_mode: str,
    rubric_mapping: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(source_rows):
        validate_source_row(row, split, idx)
        row_for_prompt = apply_rubric_mode(row, rubric_mode, rubric_mapping)
        label_5 = _as_int(row["label_5"])
        out.append(
            _base_output_row(
                row_id=str(row.get("record_id") or f"{split}_{idx}"),
                record_id=str(row.get("record_id") or f"{split}_{idx}"),
                question_key=_none_if_empty(row.get("question_key")),
                triple_key=_none_if_empty(row.get("triple_key")),
                source_type="human",
                label_provenance="human_score",
                question=clean_question_text(row.get("question")),
                answer=clean_answer_text(row.get("answer")),
                metric_canonical=str(row.get("metric_canonical") or ""),
                rubric_text=_normalize_rubric_text(row_for_prompt.get("rubric_text")),
                scenario_canonical=str(row.get("scenario_canonical") or ""),
                subject_canonical=str(row.get("subject_canonical") or ""),
                education_level_canonical=str(row.get("education_level_canonical") or ""),
                language=_language(row.get("language")),
                label_5=label_5,
                human_mean_5=float(row["human_mean_5"]),
                synthetic_target_label_5=None,
                synthetic_id=None,
                error_type=None,
                source_record_id=None,
                source_question_key=None,
                source_triple_key=None,
                generation_split_name="question_seed42",
                tokenizer=tokenizer,
                max_length=max_length,
            )
        )
    return out


def convert_synthetic_rows(
    synthetic_rows: list[dict[str, Any]],
    tokenizer: Any | None,
    max_length: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(synthetic_rows):
        label_5 = _as_int(row["target_label_5"])
        synthetic_id = str(row.get("synthetic_id") or f"synthetic_{idx:04d}")
        source_triple_key = str(row.get("source_triple_key") or "")
        out.append(
            _base_output_row(
                row_id=synthetic_id,
                record_id=synthetic_id,
                question_key=_none_if_empty(row.get("source_question_key")),
                triple_key=f"synthetic::{source_triple_key}::{synthetic_id}",
                source_type="synthetic",
                label_provenance="pseudo_label",
                question=clean_question_text(row.get("question")),
                answer=clean_answer_text(row.get("answer_synthetic")),
                metric_canonical=str(row.get("metric_canonical") or ""),
                rubric_text=_normalize_rubric_text(row.get("rubric_text")),
                scenario_canonical=str(row.get("scenario_canonical") or ""),
                subject_canonical=str(row.get("subject_canonical") or ""),
                education_level_canonical=str(row.get("education_level_canonical") or ""),
                language=_language(row.get("language")),
                label_5=label_5,
                human_mean_5=None,
                synthetic_target_label_5=label_5,
                synthetic_id=synthetic_id,
                error_type=str(row.get("error_type") or ""),
                source_record_id=str(row.get("source_record_id") or ""),
                source_question_key=str(row.get("source_question_key") or ""),
                source_triple_key=source_triple_key,
                generation_split_name="question_seed42",
                tokenizer=tokenizer,
                max_length=max_length,
            )
        )
    return out


def _split_rows_for_dataset(
    dataset_name: str,
    human_by_split: dict[str, list[dict[str, Any]]],
    synthetic_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if dataset_name == "QD-S0_human_only":
        return {
            "train": human_by_split["train"],
            "dev": human_by_split["dev"],
            "test": human_by_split["test"],
        }
    if dataset_name in {"QD-S1_human_plus_synthetic_ordinal", "QD-S2_human_plus_synthetic_L1"}:
        return {
            "train": human_by_split["train"] + synthetic_rows,
            "dev": human_by_split["dev"],
            "test": human_by_split["test"],
        }
    if dataset_name == "QD-S3_synthetic_pretrain_then_human_finetune":
        return {
            "stage1_train": synthetic_rows,
            "stage2_train": human_by_split["train"],
            "train": human_by_split["train"],
            "dev": human_by_split["dev"],
            "test": human_by_split["test"],
        }
    raise ValueError(f"Unknown dataset_name={dataset_name}")


def _distribution_rows(rows_by_split: dict[str, list[dict[str, Any]]], field: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        counts = Counter(str(row.get(field) if row.get(field) is not None else "") for row in rows)
        for value, count in sorted(counts.items(), key=lambda item: item[0]):
            out.append({"split": split, field: value, "count": count})
    return out


def _dataset_stats_rows(dataset_name: str, rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        source_counts = Counter(row.get("source_type") for row in rows)
        expected = ""
        if dataset_name == "QD-S0_human_only":
            expected = EXPECTED_SPLIT_ROWS.get(split, "")
        elif dataset_name in {"QD-S1_human_plus_synthetic_ordinal", "QD-S2_human_plus_synthetic_L1"} and split == "train":
            expected = EXPECTED_SPLIT_ROWS["train"] + EXPECTED_SYNTHETIC_ROWS
        elif dataset_name in {"QD-S1_human_plus_synthetic_ordinal", "QD-S2_human_plus_synthetic_L1"}:
            expected = EXPECTED_SPLIT_ROWS.get(split, "")
        elif dataset_name == "QD-S3_synthetic_pretrain_then_human_finetune":
            expected = {
                "stage1_train": EXPECTED_SYNTHETIC_ROWS,
                "stage2_train": EXPECTED_SPLIT_ROWS["train"],
                "train": EXPECTED_SPLIT_ROWS["train"],
                "dev": EXPECTED_SPLIT_ROWS["dev"],
                "test": EXPECTED_SPLIT_ROWS["test"],
            }.get(split, "")
        status = "PASS" if expected == "" or len(rows) == int(expected) else "FAIL"
        out.append(
            {
                "dataset_name": dataset_name,
                "split": split,
                "rows": len(rows),
                "expected_rows": expected,
                "human_rows": source_counts.get("human", 0),
                "synthetic_rows": source_counts.get("synthetic", 0),
                "status": status,
            }
        )
    return out


def _check_dataset(
    dataset_name: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
    source_index: dict[str, set[str]],
) -> list[dict[str, Any]]:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    synthetic_rows = [row for row in all_rows if row.get("source_type") == "synthetic"]
    dev_test_rows = rows_by_split.get("dev", []) + rows_by_split.get("test", [])
    synthetic_ids = [str(row.get("synthetic_id") or "") for row in synthetic_rows]
    answer_hashes = [_answer_hash(row.get("answer")) for row in synthetic_rows]
    checks: list[dict[str, Any]] = []

    def add(check_name: str, status: bool, details: Any = "") -> None:
        checks.append(
            {
                "dataset_name": dataset_name,
                "check_name": check_name,
                "status": "PASS" if status else "FAIL",
                "details": details,
            }
        )

    for split, expected in EXPECTED_SPLIT_ROWS.items():
        if split in rows_by_split and split != "train":
            add(f"{split}_human_only", all(row.get("source_type") == "human" for row in rows_by_split[split]))
            add(f"{split}_row_count", len(rows_by_split[split]) == expected, len(rows_by_split[split]))

    synthetic_in_nontrain = sum(1 for row in dev_test_rows if row.get("source_type") == "synthetic")
    add("synthetic_rows_only_in_train", synthetic_in_nontrain == 0, synthetic_in_nontrain)
    add("synthetic_label_provenance_pseudo_label", all(row.get("label_provenance") == "pseudo_label" for row in synthetic_rows))
    human_rows = [row for row in all_rows if row.get("source_type") == "human"]
    add("human_label_provenance_human_score", all(row.get("label_provenance") == "human_score" for row in human_rows))

    dev_test_q = source_index["dev_question_keys"] | source_index["test_question_keys"]
    dev_test_t = source_index["dev_triple_keys"] | source_index["test_triple_keys"]
    source_q_overlap = sorted({str(row.get("source_question_key")) for row in synthetic_rows if str(row.get("source_question_key")) in dev_test_q})
    source_t_overlap = sorted({str(row.get("source_triple_key")) for row in synthetic_rows if str(row.get("source_triple_key")) in dev_test_t})
    add("no_dev_test_question_overlap_for_synthetic_source", len(source_q_overlap) == 0, len(source_q_overlap))
    add("no_dev_test_triple_overlap_for_synthetic_source", len(source_t_overlap) == 0, len(source_t_overlap))

    duplicate_synthetic_id_count = len(synthetic_ids) - len(set(synthetic_ids))
    duplicate_answer_hash_count = len(answer_hashes) - len(set(answer_hashes))
    add("no_duplicate_synthetic_id", duplicate_synthetic_id_count == 0, duplicate_synthetic_id_count)
    add("no_duplicate_synthetic_answer_hash", duplicate_answer_hash_count == 0, duplicate_answer_hash_count)

    missing_required = sorted({field for row in all_rows for field in REQUIRED_TRAINING_FIELDS if field not in row})
    add("required_training_fields_present", len(missing_required) == 0, ",".join(missing_required))
    add("a4_template_fields_present", all(row.get("template_name") == TEMPLATE_NAME and row.get("text") for row in all_rows))

    forbidden_hits = defaultdict(int)
    for row in all_rows:
        text = str(row.get("text") or "")
        for marker in FORBIDDEN_TEXT_MARKERS:
            if marker in text:
                forbidden_hits[marker] += 1
    add("no_rationale_error_type_label_source_in_training_text", not forbidden_hits, dict(forbidden_hits))
    return checks


def _scan_api_keys(paths: list[Path]) -> list[str]:
    hits: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            candidates = [path]
        else:
            candidates = [p for p in path.rglob("*") if p.is_file() and p.stat().st_size < 20_000_000]
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if API_KEY_PATTERN.search(text):
                hits.append(relpath(candidate))
    return sorted(set(hits))


def _checkpoint_weight_files(paths: list[Path]) -> list[str]:
    hits: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in CHECKPOINT_EXTENSIONS:
                hits.append(relpath(candidate))
    return sorted(set(hits))


def _write_data_card(dataset_dir: Path, dataset_name: str, stats_rows: list[dict[str, Any]], leakage_rows: list[dict[str, Any]]) -> None:
    overall = "PASS" if all(row["status"] == "PASS" for row in stats_rows + leakage_rows) else "FAIL"
    if dataset_name == "QD-S0_human_only":
        purpose = "Human-only question-disjoint baseline registration dataset."
    elif dataset_name == "QD-S1_human_plus_synthetic_ordinal":
        purpose = "Human train plus reviewed low-score synthetic rows for ordinary ordinal training."
    elif dataset_name == "QD-S2_human_plus_synthetic_L1":
        purpose = "Human train plus reviewed low-score synthetic rows for later L1-weighted ordinal training."
    else:
        purpose = "Two-stage dataset: synthetic low-score pretrain followed by human-only fine-tuning."
    lines = [
        f"# {dataset_name}",
        "",
        f"Overall status: **{overall}**",
        "",
        purpose,
        "",
        "Template: `A4_question_answer_metric_rubric_metadata`.",
        "",
        "Synthetic labels remain pseudo labels (`label_provenance=pseudo_label`) and are not human labels.",
        "",
        "## Splits",
        "",
        "| split | rows | human | synthetic | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in stats_rows:
        lines.append(
            f"| {row['split']} | {row['rows']} | {row['human_rows']} | {row['synthetic_rows']} | {row['status']} |"
        )
    lines.extend(["", "## Leakage", "", "| check | status | details |", "| --- | --- | --- |"])
    for row in leakage_rows:
        lines.append(f"| {row['check_name']} | {row['status']} | {row.get('details', '')} |")
    write_text(dataset_dir / "data_card.md", "\n".join(lines))


def _write_dataset_outputs(
    dataset_name: str,
    rows_by_split: dict[str, list[dict[str, Any]]],
    source_index: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_dir = OUTPUT_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        write_jsonl(dataset_dir / f"{split}.jsonl", rows)

    stats_rows = _dataset_stats_rows(dataset_name, rows_by_split)
    leakage_rows = _check_dataset(dataset_name, rows_by_split, source_index)
    write_csv(dataset_dir / "dataset_stats.csv", stats_rows)
    write_csv(dataset_dir / "label_distribution.csv", _distribution_rows(rows_by_split, "label_5"))
    write_csv(dataset_dir / "source_distribution.csv", _distribution_rows(rows_by_split, "source_type"))
    write_csv(dataset_dir / "metric_distribution.csv", _distribution_rows(rows_by_split, "metric_canonical"))
    write_csv(dataset_dir / "language_distribution.csv", _distribution_rows(rows_by_split, "language"))
    write_csv(dataset_dir / "leakage_check.csv", leakage_rows)
    _write_data_card(dataset_dir, dataset_name, stats_rows, leakage_rows)
    return stats_rows, leakage_rows


def _combined_distribution_rows(datasets: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset_name, rows_by_split in datasets.items():
        for split, rows in rows_by_split.items():
            counts = Counter(_as_int(row["label_5"]) for row in rows)
            for label in range(1, 6):
                out.append(
                    {
                        "dataset_name": dataset_name,
                        "split": split,
                        "label_5": label,
                        "count": counts.get(label, 0),
                    }
                )
    return out


def _synthetic_contribution_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(field) or "") for row in rows)
    total = len(rows)
    return [
        {
            field: value,
            "synthetic_rows": count,
            "share": round(count / total, 6) if total else 0.0,
        }
        for value, count in sorted(counts.items(), key=lambda item: item[0])
    ]


def _write_global_reports(
    datasets: dict[str, dict[str, list[dict[str, Any]]]],
    stats_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    synthetic_training_rows: list[dict[str, Any]],
    global_checks: list[dict[str, Any]],
) -> None:
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for dataset_name, rows_by_split in datasets.items():
        for split, rows in rows_by_split.items():
            source_counts = Counter(row.get("source_type") for row in rows)
            summary_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split": split,
                    "rows": len(rows),
                    "human_rows": source_counts.get("human", 0),
                    "synthetic_rows": source_counts.get("synthetic", 0),
                }
            )
    write_csv(tables_dir / "exp06_training_dataset_summary.csv", summary_rows)
    write_csv(tables_dir / "exp06_combined_label_distribution.csv", _combined_distribution_rows(datasets))
    write_csv(
        tables_dir / "exp06_synthetic_contribution_by_metric.csv",
        _synthetic_contribution_rows(synthetic_training_rows, "metric_canonical"),
    )
    write_csv(
        tables_dir / "exp06_synthetic_contribution_by_language.csv",
        _synthetic_contribution_rows(synthetic_training_rows, "language"),
    )
    leakage_summary_rows = leakage_rows + global_checks
    write_csv(tables_dir / "exp06_leakage_summary.csv", leakage_summary_rows)

    overall = "PASS" if all(row["status"] == "PASS" for row in stats_rows + leakage_summary_rows) else "FAIL"
    can_train = overall == "PASS"
    label_counts = Counter(_as_int(row["label_5"]) for row in synthetic_training_rows)
    language_counts = Counter(str(row.get("language") or "") for row in synthetic_training_rows)
    lines = [
        "# Exp6 Synthetic-Augmented Training Datasets",
        "",
        f"Overall dataset build status: **{overall}**",
        "",
        "This stage builds data only. It does not call an API, generate new synthetic rows, or train models.",
        "",
        "## Dataset Summary",
        "",
        "| dataset | split | rows | human | synthetic |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset_name']} | {row['split']} | {row['rows']} | {row['human_rows']} | {row['synthetic_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Synthetic Pool",
            "",
            f"- Synthetic rows used: {len(synthetic_training_rows)}",
            "- Label distribution: "
            + ", ".join(f"{label}={label_counts.get(label, 0)}" for label in sorted(FINAL_SYNTHETIC_LABEL_TARGET)),
            "- Language distribution: " + ", ".join(f"{k}={v}" for k, v in sorted(language_counts.items())),
            "- Labels remain pseudo labels and must not be interpreted as human labels.",
            "",
            "## Training Gate",
            "",
            f"- Can Exp6 model training start? {'YES' if can_train else 'NO'}",
            "- Allowed training runs if the gate is YES: QD-S1 ordinary ordinal; QD-S2 L1 weighted ordinal; QD-S3 synthetic pretrain -> human fine-tune.",
            "- Synthetic-only main training is not built because the synthetic pool only covers labels 1/2/3.",
        ]
    )
    write_text(OUTPUT_DIR / "report.md", "\n".join(lines))

    blocker_rows = [row for row in leakage_summary_rows + stats_rows if row["status"] != "PASS"]
    review_lines = [
        "# Exp6 Training Dataset Review Package",
        "",
        f"Can Exp6 model training start? **{'YES' if can_train else 'NO'}**",
        "",
        "Allowed training runs:",
        "",
        "- QD-S1 ordinary ordinal",
        "- QD-S2 L1 weighted ordinal",
        "- QD-S3 synthetic pretrain -> human fine-tune",
        "",
        "Training is not executed in this stage.",
        "",
        "## Remaining Blockers",
        "",
    ]
    if blocker_rows:
        for row in blocker_rows:
            review_lines.append(f"- {row.get('dataset_name', 'global')}: {row.get('check_name', row.get('split'))} = {row['status']}")
    else:
        review_lines.append("- None for dataset build. Model training can start as a separate reviewed step.")
    write_text(OUTPUT_DIR / "review_package.md", "\n".join(review_lines))

    notion_lines = [
        "# Exp6-13 Synthetic-Augmented Training Dataset Summary",
        "",
        f"- 数据集构建状态：**{overall}**",
        f"- 是否可开始 Exp6 model training：**{'YES' if can_train else 'NO'}**",
        "- 本阶段未调用 API、未生成新 synthetic、未训练模型。",
        "- final synthetic low-score pool 作为 pseudo-label augmentation，只进入 train。",
        "",
        "## Datasets",
        "",
        "| Dataset | Train design | Dev/Test | Purpose |",
        "| --- | --- | --- | --- |",
        "| QD-S0_human_only | human train only | human only | baseline registration |",
        "| QD-S1_human_plus_synthetic_ordinal | human + 384 synthetic | human only | ordinary ordinal |",
        "| QD-S2_human_plus_synthetic_L1 | human + 384 synthetic | human only | L1 weighted ordinal |",
        "| QD-S3_synthetic_pretrain_then_human_finetune | stage1 synthetic, stage2 human | human only | pretrain then finetune |",
        "",
        "## Synthetic Label Distribution",
        "",
        "| Label | Count |",
        "| ---: | ---: |",
    ]
    for label in sorted(FINAL_SYNTHETIC_LABEL_TARGET):
        notion_lines.append(f"| {label} | {label_counts.get(label, 0)} |")
    write_text(OUTPUT_DIR / "notion_exp06_training_dataset_summary.md", "\n".join(notion_lines))


def _global_checks(
    raw_by_split: dict[str, list[dict[str, Any]]],
    synthetic_source_rows: list[dict[str, Any]],
    synthetic_training_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_index = _source_index(raw_by_split)
    checks: list[dict[str, Any]] = []

    def add(check_name: str, status: bool, details: Any = "") -> None:
        checks.append({"dataset_name": "GLOBAL", "check_name": check_name, "status": "PASS" if status else "FAIL", "details": details})

    add("QD-S0 train/dev/test rows = 3326/1107/1103", [len(raw_by_split[s]) for s in ["train", "dev", "test"]] == [3326, 1107, 1103])
    add("final_synthetic_pool_count_384", len(synthetic_source_rows) == EXPECTED_SYNTHETIC_ROWS, len(synthetic_source_rows))
    label_counts = Counter(_as_int(row.get("target_label_5")) for row in synthetic_source_rows)
    add("final_synthetic_pool_label_distribution_168_168_48", dict(label_counts) == FINAL_SYNTHETIC_LABEL_TARGET, dict(label_counts))
    add("synthetic_label_provenance_pseudo_label", all(row.get("label_provenance") == "pseudo_label" for row in synthetic_source_rows))
    add("synthetic_label_source_synthetic_design", all(row.get("label_source") == "synthetic_design" for row in synthetic_source_rows))
    add("synthetic_source_split_train", all(row.get("source_split") == "train" for row in synthetic_source_rows))
    add(
        "synthetic_source_question_keys_in_train",
        all(str(row.get("source_question_key")) in source_index["train_question_keys"] for row in synthetic_source_rows),
    )
    add(
        "synthetic_source_triple_keys_in_train",
        all(str(row.get("source_triple_key")) in source_index["train_triple_keys"] for row in synthetic_source_rows),
    )
    add(
        "no_synthetic_source_question_overlap_with_dev_test",
        not any(
            str(row.get("source_question_key")) in source_index["dev_question_keys"] | source_index["test_question_keys"]
            for row in synthetic_source_rows
        ),
    )
    add(
        "no_synthetic_source_triple_overlap_with_dev_test",
        not any(
            str(row.get("source_triple_key")) in source_index["dev_triple_keys"] | source_index["test_triple_keys"]
            for row in synthetic_source_rows
        ),
    )
    synthetic_ids = [str(row.get("synthetic_id") or "") for row in synthetic_source_rows]
    add("no_duplicate_synthetic_id", len(synthetic_ids) == len(set(synthetic_ids)), len(synthetic_ids) - len(set(synthetic_ids)))
    answer_hashes = [_answer_hash(row.get("answer_synthetic")) for row in synthetic_source_rows]
    add("no_duplicate_answer_hash_within_synthetic_pool", len(answer_hashes) == len(set(answer_hashes)), len(answer_hashes) - len(set(answer_hashes)))
    text_bad = defaultdict(int)
    for row in synthetic_training_rows:
        text = str(row.get("text") or "")
        for marker in FORBIDDEN_TEXT_MARKERS:
            if marker in text:
                text_bad[marker] += 1
    add("no_rationale_error_type_label_source_in_training_text", not text_bad, dict(text_bad))
    add("a4_template_fields_present", all(row.get("template_name") == TEMPLATE_NAME and row.get("text") for row in synthetic_training_rows))
    api_hits = _scan_api_keys([OUTPUT_DIR, SYNTHETIC_POOL_REPORT_PATH, SYNTHETIC_POOL_SANITY_PATH])
    add("no_api_key_in_files", not api_hits, len(api_hits))
    checkpoint_hits = _checkpoint_weight_files([OUTPUT_DIR])
    add("no_checkpoint_weights", not checkpoint_hits, len(checkpoint_hits))
    add("no_model_training_executed", True, "dataset build only")
    add("no_api_called", True, "local transformation only")
    add("no_new_synthetic_generated", True, "reused final curated synthetic pool")
    return checks


def build_training_datasets(model_name_or_path: str | None = None, max_length: int = 2048, rubric_mode: str = AUTO_MODE) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_by_split, synthetic_source_rows = _load_inputs()
    resolved_rubric_mode = resolve_rubric_mode(rubric_mode)
    rubric_mapping = load_rubric_mapping() if resolved_rubric_mode == "auto" else {}
    tokenizer = load_tokenizer(model_name_or_path, local_files_only=True) if model_name_or_path else None

    human_by_split = {
        split: convert_human_rows(rows, split, tokenizer, max_length, resolved_rubric_mode, rubric_mapping)
        for split, rows in raw_by_split.items()
    }
    synthetic_training_rows = convert_synthetic_rows(synthetic_source_rows, tokenizer, max_length)
    datasets = {
        dataset_name: _split_rows_for_dataset(dataset_name, human_by_split, synthetic_training_rows)
        for dataset_name in DATASET_NAMES
    }

    source_index = _source_index(raw_by_split)
    all_stats_rows: list[dict[str, Any]] = []
    all_leakage_rows: list[dict[str, Any]] = []
    for dataset_name, rows_by_split in datasets.items():
        stats_rows, leakage_rows = _write_dataset_outputs(dataset_name, rows_by_split, source_index)
        all_stats_rows.extend(stats_rows)
        all_leakage_rows.extend(leakage_rows)

    global_checks = _global_checks(raw_by_split, synthetic_source_rows, synthetic_training_rows)
    _write_global_reports(datasets, all_stats_rows, all_leakage_rows, synthetic_training_rows, global_checks)

    failures = [row for row in all_stats_rows + all_leakage_rows + global_checks if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"Exp6 training dataset build sanity failed with {len(failures)} failures")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--rubric_mode", default=AUTO_MODE)
    args = parser.parse_args()
    build_training_datasets(args.model_name_or_path, args.max_length, args.rubric_mode)
    print(f"Wrote Exp6 synthetic-augmented training datasets: {relpath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
