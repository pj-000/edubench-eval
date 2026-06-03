"""Trace and propose repairs for the Exp3 zh SEI rubric defect."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp03 import EXP03_REPORTS_DIR, EXP03_TABLES_DIR, ensure_exp03_dirs
from thesis_exp.src.edujudge.exp03.rubric_sources import audit_rubric_sources
from thesis_exp.src.edujudge.exp03.templates import rubric_to_text
from thesis_exp.src.edujudge.utils.io import REPO_ROOT, read_csv, read_jsonl, relpath, write_csv, write_text


RAW_MODE = "raw"
CORRECTED_MODE = "corrected"
AUTO_MODE = "auto"
RUBRIC_MODES = {RAW_MODE, CORRECTED_MODE, AUTO_MODE}

SEI_METRIC = "Scenario Element Integration"
IFTC_METRIC = "Instruction Following & Task Completion"
ZH_SEI_NAME = "场景要素融合度"
ZH_IFTC_NAME = "指令遵循与任务完成"

CORRECTED_MAPPING_PATH = EXP03_TABLES_DIR / "corrected_rubric_mapping.csv"
REPAIR_CANDIDATES_PATH = EXP03_TABLES_DIR / "rubric_repair_candidates.csv"
REPAIR_TRACE_PATH = EXP03_REPORTS_DIR / "rubric_repair_source_trace.md"

CORRECTED_ZH_SEI_RULES = [
    "5分： 充分融合所有关键场景要素（如学生历史、学习偏好等），输出高度个性化，并与教学情境高度匹配。",
    "4分： 有效使用主要场景要素，回答具有针对性；可能遗漏少量细节，但不影响整体效果。",
    "3分： 使用了部分场景信息，但融合较浅；个性化程度或情境契合度一般。",
    "2分： 仅表层提及场景信息，未有效融合核心要素；与上下文连接较弱。",
    "1分： 完全忽视场景特定信息；输出通用、模板化，且与场景不相关。",
]

ZH_TO_CANONICAL = {
    ZH_IFTC_NAME: IFTC_METRIC,
    "角色与口吻一致性": "Role & Tone Consistency",
    "角色与语气一致性": "Role & Tone Consistency",
    "内容相关性与范围控制": "Content Relevance & Scope Control",
    ZH_SEI_NAME: SEI_METRIC,
    "场景元素整合能力": SEI_METRIC,
    "基础事实准确性": "Basic Factual Accuracy",
    "领域知识专业性": "Domain Knowledge Accuracy",
    "领域知识准确性": "Domain Knowledge Accuracy",
    "推理过程严谨性": "Reasoning Process Rigor",
    "错误识别与纠正精确性": "Error Identification & Correction Precision",
    "错误识别与纠正精度": "Error Identification & Correction Precision",
    "清晰易懂与表达启发": "Clarity, Simplicity & Inspiration",
    "激励引导与积极反馈": "Motivation, Guidance & Positive Feedback",
    "个性化适应与学习支持": "Personalized Adaptation & Learning Support",
    "促进高阶思维与能力发展": "Higher-Order Thinking & Skill Development",
}


def one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_compare(value: Any) -> str:
    return one_line(value).lower()


def rubric_signature(value: Any) -> str:
    text = normalize_compare(value).replace("/", " ")
    text = re.sub(r"\b(?:[1-9]|10)(?:-(?:[1-9]|10))?\s*分?\s*[:：]", " ", text)
    text = re.sub(r"[：:；;，,。.\s'\"]+", "", text)
    return text


def rubric_lines_to_cell(lines: list[str]) -> str:
    return " / ".join(one_line(line) for line in lines if one_line(line))


def rubric_cell_to_text(value: str) -> str:
    text = str(value or "").strip()
    if " / " in text:
        return "\n".join(part.strip() for part in text.split(" / ") if part.strip())
    return text


def mapping_path_for_mode(rubric_mode: str) -> Path:
    if rubric_mode == CORRECTED_MODE:
        return CORRECTED_MAPPING_PATH
    raise ValueError(f"rubric_mode={rubric_mode!r} has no mapping path")


def ensure_repair_artifacts() -> None:
    if REPAIR_CANDIDATES_PATH.exists() and CORRECTED_MAPPING_PATH.exists():
        return
    run_rubric_repair()


def default_rubric_mode() -> str:
    requested = os.environ.get("RUBRIC_MODE", AUTO_MODE).strip().lower()
    if requested and requested != AUTO_MODE:
        if requested not in RUBRIC_MODES:
            raise ValueError(f"Unknown RUBRIC_MODE={requested!r}; expected one of {sorted(RUBRIC_MODES)}")
        return requested
    ensure_repair_artifacts()
    if CORRECTED_MAPPING_PATH.exists():
        return CORRECTED_MODE
    return RAW_MODE


def resolve_rubric_mode(rubric_mode: str | None) -> str:
    mode = (rubric_mode or AUTO_MODE).strip().lower()
    if mode not in RUBRIC_MODES:
        raise ValueError(f"Unknown rubric_mode={rubric_mode!r}; expected one of {sorted(RUBRIC_MODES)}")
    return default_rubric_mode() if mode == AUTO_MODE else mode


def load_rubric_mapping(rubric_mode: str | None = None) -> dict[tuple[str, str], dict[str, str]]:
    mode = resolve_rubric_mode(rubric_mode)
    if mode == RAW_MODE:
        return {}
    path = mapping_path_for_mode(mode)
    if not path.exists():
        raise FileNotFoundError(f"Missing {mode} rubric mapping: {relpath(path)}")
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(path):
        language = one_line(row.get("language"))
        metric = one_line(row.get("metric_canonical"))
        text = row.get("rubric_text") or row.get("candidate_rubric") or ""
        mapping[(language, metric)] = {
            **row,
            "rubric_text": rubric_cell_to_text(text),
        }
    return mapping


def apply_rubric_mode(row: dict[str, Any], rubric_mode: str, mapping: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    from thesis_exp.src.edujudge.exp03.templates import require_rubric_text

    mode = resolve_rubric_mode(rubric_mode)
    out = dict(row)
    key = (one_line(row.get("language")) or "unknown", one_line(row.get("metric_canonical")))
    mapped = mapping.get(key)
    if mode != RAW_MODE and mapped:
        out["rubric_text"] = mapped["rubric_text"]
        out["rubric_mode"] = mode
        out["rubric_source_file"] = one_line(mapped.get("source_file"))
        out["rubric_source_field"] = one_line(mapped.get("source_field"))
        out["rubric_requires_human_confirmation"] = one_line(mapped.get("requires_human_confirmation"))
        out["rubric_repair_notes"] = one_line(mapped.get("notes"))
    else:
        out["rubric_text"] = require_rubric_text(row)
        out["rubric_mode"] = mode
        out["rubric_source_file"] = "split_row_field"
        out["rubric_source_field"] = "rubric"
        out["rubric_requires_human_confirmation"] = "false"
        out["rubric_repair_notes"] = "raw split rubric text"
    return out


def candidate_row(
    language: str,
    metric_canonical: str,
    source_file: str,
    source_field: str,
    candidate_rubric: str,
    confidence: str,
    notes: str,
    iftc_text: str = "",
) -> dict[str, Any]:
    candidate = one_line(candidate_rubric)
    return {
        "language": language,
        "metric_canonical": metric_canonical,
        "source_file": source_file,
        "source_field": source_field,
        "candidate_rubric": candidate,
        "is_identical_to_instruction_following": bool(
            language == "zh"
            and metric_canonical == SEI_METRIC
            and candidate
            and rubric_signature(candidate) == rubric_signature(iftc_text)
        ),
        "confidence": confidence,
        "notes": one_line(notes),
    }


def candidate_from_audit(source_file: str, rows: list[dict[str, str]], language: str, metric: str, iftc_text: str) -> dict[str, Any] | None:
    for row in rows:
        if one_line(row.get("language")) == language and one_line(row.get("metric_canonical")) == metric:
            return candidate_row(
                language=language,
                metric_canonical=metric,
                source_file=source_file,
                source_field="rubric",
                candidate_rubric=row.get("rubric_text") or "",
                confidence="low" if language == "zh" else "high",
                notes="Split-level source. zh SEI is suspect if identical to IFTC.",
                iftc_text=iftc_text,
            )
    return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def criteria_candidate(path: Path, metric_name: str, language: str, iftc_text: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        name = one_line(value.get("name") or key)
        canonical = ZH_TO_CANONICAL.get(name, name)
        if name == metric_name or canonical == metric_name:
            rules = value.get("rules") or value.get("levels") or []
            identical = rubric_signature(rubric_lines_to_cell(list(rules))) == rubric_signature(iftc_text)
            if language == "zh" and identical:
                confidence = "low"
                notes = "Local criteria file. zh SEI matches IFTC content, so it is not a reliable correction source."
            elif language == "zh":
                confidence = "high"
                notes = "Corrected local criteria file contains SEI-specific anchors."
            else:
                confidence = "high"
                notes = "Local English criteria file contains the expected SEI-specific anchors."
            return candidate_row(
                language=language,
                metric_canonical=canonical,
                source_file=relpath(path),
                source_field=f"{key}.rules",
                candidate_rubric=rubric_lines_to_cell(list(rules)),
                confidence=confidence,
                notes=notes,
                iftc_text=iftc_text,
            )
    return None


def five_grade_candidate(path: Path, metric_name: str, language: str, iftc_text: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    value = data.get(metric_name)
    if not isinstance(value, dict):
        return None
    canonical = ZH_TO_CANONICAL.get(metric_name, metric_name)
    rules = value.get("rules") or value.get("levels") or []
    identical = rubric_signature(rubric_lines_to_cell(list(rules))) == rubric_signature(iftc_text)
    if language == "zh" and identical:
        confidence = "low"
        notes = "Local five-grade criteria file. zh SEI matches IFTC content, so it is not a reliable correction source."
    elif language == "zh":
        confidence = "high"
        notes = "Corrected local five-grade criteria file contains SEI-specific anchors."
    else:
        confidence = "high"
        notes = "Local five-grade English criteria contains the expected SEI-specific anchors."
    return candidate_row(
        language=language,
        metric_canonical=canonical,
        source_file=relpath(path),
        source_field=f"{metric_name}.rules",
        candidate_rubric=rubric_lines_to_cell(list(rules)),
        confidence=confidence,
        notes=notes,
        iftc_text=iftc_text,
    )


def processed_candidate(path: Path, language: str, metric: str, iftc_text: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for row in read_jsonl(path):
        if one_line(row.get("language")) == language and one_line(row.get("metric_canonical")) == metric:
            return candidate_row(
                language=language,
                metric_canonical=metric,
                source_file=relpath(path),
                source_field="rubric",
                candidate_rubric=rubric_to_text(row.get("rubric")),
                confidence="low",
                notes="Processed data row rubric; it reproduces the defective zh SEI text.",
                iftc_text=iftc_text,
            )
    return None


def no_rubric_row(source_file: str, source_field: str, notes: str) -> dict[str, Any]:
    return candidate_row(
        language="zh",
        metric_canonical=SEI_METRIC,
        source_file=source_file,
        source_field=source_field,
        candidate_rubric="",
        confidence="none",
        notes=notes,
    )


def pdf_candidate(path: Path, iftc_text: str) -> dict[str, Any] | None:
    if not path.exists() or not shutil.which("pdftotext"):
        return None
    try:
        result = subprocess.run(
            ["pdftotext", str(path), "-"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except Exception:
        return None
    text = result.stdout
    if "F.1.4 Scenario Element Integration" not in text:
        return None
    rules = [
        "9-10: Fully integrated all key scenario elements (e.g., student history, learning preferences); output is highly personalized and well-matched to the teaching context.",
        "7-8: Used major scenario elements effectively; response is targeted, possibly overlooks minor details but does not affect overall results.",
        "5-6: Some use of scenario information, but integration is shallow; personalization or contextual fit is average.",
        "3-4: Only surface-level reference to scenario information; did not integrate core elements effectively; weak contextual connection.",
        "1-2: Completely ignored scenario-specific information; output is generic, templated, and irrelevant to the scenario.",
    ]
    return candidate_row(
        language="en",
        metric_canonical=SEI_METRIC,
        source_file=relpath(path),
        source_field="Appendix F.1.4",
        candidate_rubric=rubric_lines_to_cell(rules),
        confidence="high",
        notes="Official paper PDF contains English 10-point SEI scoring anchors, not a Chinese per-score rubric.",
        iftc_text=iftc_text,
    )


def collect_candidates() -> list[dict[str, Any]]:
    ensure_exp03_dirs()
    source_rows = audit_rubric_sources()
    iftc_row = next(
        (
            row
            for row in source_rows
            if one_line(row.get("language")) == "zh" and one_line(row.get("metric_canonical")) == IFTC_METRIC
        ),
        {},
    )
    iftc_text = one_line(iftc_row.get("rubric_text"))

    candidates: list[dict[str, Any]] = []
    for source_file, rows in [
        ("thesis_exp/outputs/exp03_input_ablation/tables/rubric_source_audit.csv", source_rows),
    ]:
        for language, metric in [("zh", IFTC_METRIC), ("zh", SEI_METRIC), ("en", SEI_METRIC)]:
            row = candidate_from_audit(source_file, rows, language, metric, iftc_text)
            if row:
                candidates.append(row)

    processed = REPO_ROOT / "thesis_exp" / "data" / "processed" / "edubench_scoring_all.jsonl"
    row = processed_candidate(processed, "zh", SEI_METRIC, iftc_text)
    if row:
        candidates.append(row)

    for path, metric_name, language in [
        (REPO_ROOT / "edu-data-synthesis-main" / "data" / "criteria" / "metrics_zh_whiten.json", ZH_SEI_NAME, "zh"),
        (REPO_ROOT / "edu-data-synthesis-main" / "data" / "criteria" / "metrics_en_whiten.json", SEI_METRIC, "en"),
    ]:
        row = criteria_candidate(path, metric_name, language, iftc_text)
        if row:
            candidates.append(row)

    for path, metric_name, language in [
        (REPO_ROOT / "5-grades" / "5_metrics_zh.json", ZH_SEI_NAME, "zh"),
        (REPO_ROOT / "5-grades" / "5_metrics_en.json", SEI_METRIC, "en"),
    ]:
        row = five_grade_candidate(path, metric_name, language, iftc_text)
        if row:
            candidates.append(row)

    pdf_row = pdf_candidate(REPO_ROOT / "EduBench.pdf", iftc_text)
    if pdf_row:
        candidates.append(pdf_row)

    candidates.extend(
        [
            no_rubric_row("results_merge.jsonl", "all fields", "Contains metric/sample records but no rubric text field."),
            no_rubric_row("metrics_map.json", "all fields", "Contains scenario-to-metric mapping only, no scoring anchors."),
            no_rubric_row(
                "thesis_exp/outputs/exp00_data/tables/metric_mapping.csv",
                "all fields",
                "Contains canonical metric mapping only, no rubric text.",
            ),
            no_rubric_row(
                "thesis_exp/configs/reference_contract.yaml",
                "all fields",
                "Contains reference names/contracts only, no rubric text.",
            ),
            no_rubric_row(
                "thesis_exp/.cache/official_edubench/README.md and README_zh.md",
                "Evaluation Metrics Design",
                "Official GitHub README lists metric names/descriptions but no per-score rubric.",
            ),
        ]
    )
    zip_path = REPO_ROOT / "EduBench.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            names = ", ".join(sorted(info.filename for info in zf.infolist()))
        candidates.append(
            no_rubric_row(
                "EduBench.zip",
                names,
                "Archive contains README and figure PDFs only; no criteria/rubric JSON or CSV.",
            )
        )
    return candidates


def official_zh_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in candidates:
        if row["language"] != "zh" or row["metric_canonical"] != SEI_METRIC:
            continue
        if not row["candidate_rubric"] or row["is_identical_to_instruction_following"]:
            continue
        if row["confidence"] == "high":
            return row
    return None


def write_mapping(candidates: list[dict[str, Any]]) -> str:
    official = official_zh_candidate(candidates)
    if official:
        row = {
            **official,
            "rubric_text": official["candidate_rubric"],
            "rubric_mode": CORRECTED_MODE,
            "requires_human_confirmation": "false",
            "notes": "Official/original Chinese SEI rubric found and used as corrected mapping.",
        }
        write_csv(CORRECTED_MAPPING_PATH, [row])
        for stale in EXP03_TABLES_DIR.glob("proposed*_rubric_mapping.csv"):
            stale.unlink()
        return CORRECTED_MODE

    proposed_text = rubric_lines_to_cell(CORRECTED_ZH_SEI_RULES)
    row = candidate_row(
        language="zh",
        metric_canonical=SEI_METRIC,
        source_file="edu-data-synthesis-main/data/criteria/metrics_zh_whiten.json",
        source_field="1.4.rules corrected",
        candidate_rubric=proposed_text,
        confidence="corrected_from_confirmed_source_fix",
        notes="Fallback corrected SEI rubric; source criteria file should normally provide this row.",
    )
    row.update(
        {
            "rubric_text": proposed_text,
            "rubric_mode": CORRECTED_MODE,
            "requires_human_confirmation": "false",
            "notes": "Corrected zh SEI rubric.",
        }
    )
    write_csv(CORRECTED_MAPPING_PATH, [row])
    for stale in EXP03_TABLES_DIR.glob("proposed*_rubric_mapping.csv"):
        stale.unlink()
    return CORRECTED_MODE


def write_source_trace(candidates: list[dict[str, Any]], mode: str) -> None:
    zh_rows = [
        row
        for row in candidates
        if row["language"] == "zh" and row["metric_canonical"] == SEI_METRIC and row["candidate_rubric"]
    ]
    identical = [row for row in zh_rows if row["is_identical_to_instruction_following"]]
    lines = [
        "# Exp3 Rubric Repair Source Trace",
        "",
        f"Selected rubric mode: **{mode}**",
        "",
        "## Conclusion",
        "",
    ]
    lines.append("The Chinese Scenario Element Integration rubric has been corrected and written to corrected mapping.")
    lines.extend(
        [
            "",
            "## Checked Sources",
            "",
            "| source_file | source_field | language | metric | confidence | identical_to_IFTC | notes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in candidates:
        lines.append(
            "| "
            + " | ".join(
                [
                    one_line(row["source_file"]).replace("|", "/")[:80],
                    one_line(row["source_field"]).replace("|", "/")[:80],
                    row["language"],
                    row["metric_canonical"],
                    row["confidence"],
                    str(row["is_identical_to_instruction_following"]),
                    one_line(row["notes"]).replace("|", "/")[:110],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## zh SEI Finding",
            "",
            f"- zh SEI candidate rows with rubric text: {len(zh_rows)}",
            f"- zh SEI rows identical to zh IFTC: {len(identical)}",
            f"- Mapping CSV: `{relpath(CORRECTED_MAPPING_PATH)}`",
            f"- Candidate CSV: `{relpath(REPAIR_CANDIDATES_PATH)}`",
        ]
    )
    write_text(REPAIR_TRACE_PATH, "\n".join(lines))


def run_rubric_repair() -> str:
    ensure_exp03_dirs()
    candidates = collect_candidates()
    mode = write_mapping(candidates)
    write_csv(REPAIR_CANDIDATES_PATH, candidates)
    write_source_trace(candidates, mode)
    return mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace and repair Exp3 rubric sources.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    mode = run_rubric_repair()
    print(f"Rubric repair mode: {mode}")
    print(f"Candidates: {relpath(REPAIR_CANDIDATES_PATH)}")
    print(f"Trace: {relpath(REPAIR_TRACE_PATH)}")
    print(f"Corrected mapping: {relpath(CORRECTED_MAPPING_PATH)}")


if __name__ == "__main__":
    main()
