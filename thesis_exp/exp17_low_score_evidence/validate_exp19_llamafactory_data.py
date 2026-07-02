"""Validate Exp19-S0/S0b LLaMA-Factory dataset artifacts.

The validator is intentionally lightweight: it checks JSON/schema compatibility,
path resolution from ``dataset_dir``, basic target fields, and the no-rationale
leakage guardrail. It does not train or load a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.prepare_exp19_sft_dpo_datasets import (  # noqa: E402
    FAILURE_VOCAB,
    OPENAI_TAGS,
    clean,
)
from thesis_exp.src.edujudge.utils.io import write_text  # noqa: E402


DEFAULT_DATASET_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp19_sft_dpo_datasets_seed42")
DEFAULT_A0_CANDIDATES = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp17_a0_train_hidden_failure_signals_seed42/"
    "train_hidden_failure_candidates.csv"
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_target(payload: Any, errors: list[str], where: str) -> None:
    if not isinstance(payload, dict):
        errors.append(f"{where}: target is not an object")
        return
    score = payload.get("score")
    if not isinstance(score, int) or score < 1 or score > 5:
        errors.append(f"{where}: score must be int in [1,5], got {score!r}")
    if "major_failures" in payload:
        failures = payload.get("major_failures")
        if not isinstance(failures, list) or not failures:
            errors.append(f"{where}: major_failures must be a non-empty list")
        else:
            unknown = [item for item in failures if item not in FAILURE_VOCAB]
            if unknown:
                errors.append(f"{where}: unknown major_failures {unknown}")
    if "score_cap" in payload:
        cap = payload.get("score_cap")
        if cap is not None and (not isinstance(cap, int) or cap < 1 or cap > 5):
            errors.append(f"{where}: score_cap must be null or int in [1,5], got {cap!r}")
    if "rubric_satisfied" in payload and not isinstance(payload.get("rubric_satisfied"), bool):
        errors.append(f"{where}: rubric_satisfied must be boolean")


def parse_assistant_json(message: Any, errors: list[str], where: str) -> None:
    if not isinstance(message, dict):
        errors.append(f"{where}: assistant message is not an object")
        return
    if message.get("role") != "assistant":
        errors.append(f"{where}: assistant message role is {message.get('role')!r}")
    content = message.get("content")
    if not isinstance(content, str):
        errors.append(f"{where}: assistant content is not string")
        return
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        errors.append(f"{where}: assistant content is not valid JSON: {exc}")
        return
    validate_target(payload, errors, where)


def validate_messages(messages: Any, errors: list[str], where: str, require_assistant: bool) -> str:
    if not isinstance(messages, list) or not messages:
        errors.append(f"{where}: messages must be a non-empty list")
        return ""
    user_parts: list[str] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"{where}: message {idx} is not an object")
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            errors.append(f"{where}: message {idx} has invalid role {role!r}")
        if not isinstance(content, str):
            errors.append(f"{where}: message {idx} content is not string")
        if role == "user" and isinstance(content, str):
            user_parts.append(content)
    if require_assistant:
        parse_assistant_json(messages[-1], errors, f"{where}.messages[-1]")
    return "\n".join(user_parts)


def recovered_reasons(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for row in read_csv_rows(path):
        reason = clean(row.get("recovered_reason_summary"))
        if len(reason) >= 24:
            out.append(reason)
    return out


def validate_dataset_info(dataset_dir: Path, info: dict[str, Any], errors: list[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name, entry in info.items():
        if not isinstance(entry, dict):
            errors.append(f"{name}: dataset_info entry is not object")
            continue
        if entry.get("formatting") != "sharegpt":
            errors.append(f"{name}: formatting must be sharegpt")
        if entry.get("tags") != OPENAI_TAGS:
            errors.append(f"{name}: OpenAI role/content tags are missing or different")
        columns = entry.get("columns")
        if not isinstance(columns, dict) or columns.get("messages") != "messages":
            errors.append(f"{name}: columns.messages must be messages")
        if entry.get("ranking") is True:
            if columns.get("chosen") != "chosen" or columns.get("rejected") != "rejected":
                errors.append(f"{name}: DPO entry must map chosen/rejected")
        file_name = entry.get("file_name")
        if not isinstance(file_name, str):
            errors.append(f"{name}: file_name missing")
            continue
        path = dataset_dir / file_name
        if not path.exists():
            errors.append(f"{name}: file does not exist from dataset_dir: {path}")
        files[name] = path
    return files


def validate_records(
    name: str,
    path: Path,
    ranking: bool,
    reason_strings: list[str],
    max_records: int,
) -> tuple[int, int, Counter[str], list[str]]:
    errors: list[str] = []
    data = load_json(path)
    if not isinstance(data, list):
        return 0, 0, Counter(), [f"{name}: dataset root must be a list"]
    leakage_count = 0
    label_counts: Counter[str] = Counter()
    records = data if max_records <= 0 else data[:max_records]
    for idx, record in enumerate(records):
        where = f"{name}[{idx}]"
        if not isinstance(record, dict):
            errors.append(f"{where}: record is not object")
            continue
        user_prompt = validate_messages(record.get("messages"), errors, where, require_assistant=not ranking)
        if ranking:
            parse_assistant_json(record.get("chosen"), errors, f"{where}.chosen")
            parse_assistant_json(record.get("rejected"), errors, f"{where}.rejected")
        else:
            try:
                payload = json.loads(record["messages"][-1]["content"])
                label_counts.update([str(payload.get("score"))])
            except Exception:
                pass
        if any(reason in user_prompt for reason in reason_strings):
            leakage_count += 1
            errors.append(f"{where}: recovered human rationale appears in user prompt")
    return len(data), leakage_count, label_counts, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Exp19 LLaMA-Factory dataset files.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset-info", type=Path, default=None)
    parser.add_argument("--a0-candidates", type=Path, default=DEFAULT_A0_CANDIDATES)
    parser.add_argument("--max-records", type=int, default=0, help="0 means validate all records.")
    args = parser.parse_args()

    dataset_info_path = args.dataset_info or (args.dataset_dir / "dataset_info_snippet.json")
    errors: list[str] = []
    info = load_json(dataset_info_path)
    if not isinstance(info, dict):
        raise ValueError(f"dataset_info must be an object: {dataset_info_path}")
    files = validate_dataset_info(args.dataset_dir, info, errors)
    reasons = recovered_reasons(args.a0_candidates)

    rows = []
    total_leaks = 0
    for name, path in files.items():
        ranking = bool(info[name].get("ranking"))
        count, leaks, labels, record_errors = validate_records(name, path, ranking, reasons, args.max_records)
        total_leaks += leaks
        errors.extend(record_errors)
        rows.append(
            {
                "dataset": name,
                "count": count,
                "ranking": ranking,
                "leakage_count": leaks,
                "label_counts": dict(sorted(labels.items())),
            }
        )

    report = [
        "# Exp19 LLaMA-Factory Dataset Validation",
        "",
        f"- dataset_dir: `{args.dataset_dir}`",
        f"- dataset_info: `{dataset_info_path}`",
        f"- datasets checked: {len(rows)}",
        f"- recovered rationale strings checked: {len(reasons)}",
        f"- total user-prompt leakage count: {total_leaks}",
        f"- errors: {len(errors)}",
        "",
        "## Datasets",
        "",
    ]
    for row in rows:
        report.append(
            f"- {row['dataset']}: count={row['count']}, ranking={row['ranking']}, leakage={row['leakage_count']}, labels={row['label_counts']}"
        )
    if errors:
        report.extend(["", "## Errors", ""])
        report.extend(f"- {error}" for error in errors[:200])
    write_text(args.dataset_dir / "reports" / "exp19_s0_llamafactory_validation_report.md", "\n".join(report))
    print(json.dumps({"datasets": len(rows), "errors": len(errors), "leakage_count": total_leaks}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
