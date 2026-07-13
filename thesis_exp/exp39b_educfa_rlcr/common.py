"""Shared train-only I/O, API, hashing, and locked-span helpers for Exp39B."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import jsonschema
except ModuleNotFoundError:
    from thesis_exp.exp17_low_score_evidence import json_schema_compat as jsonschema

ROOT = Path("thesis_exp/exp39b_educfa_rlcr/outputs/exp39b_rlcr_pilot_seed43")
R1_ROOT = Path("thesis_exp/exp39b_educfa_rlcr/outputs/exp39b_r1_response_disjoint_pilot_seed44")
TRAIN_PATH = Path("thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl")
EXP39A_LOCK = Path("thesis_exp/exp39_educfa/outputs/exp39a_educfa_seed42/configs/exp39a_source_lock.json")
EXP39A_PACKETS = Path(
    "thesis_exp/exp39_educfa/outputs/exp39a_educfa_seed42/private/source_packets/exp39a_source_anchor_packets.jsonl"
)
PROMPT_DIR = Path("thesis_exp/exp39b_educfa_rlcr/prompts")
SCHEMA_DIR = Path("thesis_exp/exp39b_educfa_rlcr/schemas")

BAND_COUNTS = {"severe_low": 20, "moderate_low": 25, "boundary": 15}
BANDS = {"severe_low": [1, 2], "moderate_low": [2, 3], "boundary": [3, 3]}
OPERATORS = (
    "delete_required_span",
    "replace_with_local_contradiction",
    "delete_supporting_reasoning",
    "insert_local_scope_drift",
    "violate_explicit_constraint",
)


def reject_eval_path(path: Path) -> None:
    normalized = "/" + str(path).replace("\\", "/").lower().strip("/") + "/"
    if path.name.lower() in {"dev.jsonl", "test.jsonl", "dev.json", "test.json"}:
        raise ValueError(f"Exp39B forbids evaluation path: {path}")
    if "/paper_like_triple_seed42/dev." in normalized or "/paper_like_triple_seed42/test." in normalized:
        raise ValueError(f"Exp39B forbids paper-like dev/test: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    reject_eval_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    fields = list(fieldnames or [])
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def index_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = sample_id(row)
        if sid in indexed:
            raise ValueError(f"Duplicate sample_id: {sid}")
        indexed[sid] = row
    return indexed


def require_prepare_go(out_dir: Path) -> dict[str, Any]:
    decision_path = out_dir / "decision/exp39b_protocol_prepare_decision.json"
    if not decision_path.exists():
        raise SystemExit(f"Run Exp39B preparation first: {decision_path}")
    decision = read_json(decision_path)
    if decision.get("status") != "PROTOCOL_PREPARE_GO":
        raise SystemExit(
            "Exp39B API stage blocked by pre-registered source gate: "
            f"status={decision.get('status')} reason={decision.get('failure_reason')}"
        )
    return decision


def model_from_protocol(out_dir: Path, key: str) -> str:
    return str(read_json(out_dir / "configs/exp39b_protocol_lock.json")[key])


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def write_stage_summary(out_dir: Path, stage: str, summary: dict[str, Any]) -> None:
    write_json(out_dir / f"configs/exp39b_{stage}_summary.json", summary)


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("sample_id") or row.get("record_id") or row.get("id")
    if not value:
        raise ValueError("Missing sample ID")
    return str(value)


def score_value(row: dict[str, Any], key: str) -> int:
    return int(float(row.get(f"{key}_5", row.get(key))))


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", str(text)))


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().lower()


def text_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalize_answer(text)))


def character_ngrams(text: str, n: int = 5) -> set[str]:
    normalized = normalize_answer(text)
    return {normalized[index:index + n] for index in range(max(0, len(normalized) - n + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_layout(out_dir: Path) -> None:
    for name in (
        "configs", "tables", "reports", "decision", "hashes", "prompts", "schemas", "raw_api",
        "private/source_packets", "private/clause_plans", "private/first_pass_edits",
        "private/first_pass_candidates", "private/critic_feedback", "private/revisions",
        "private/final_candidates", "private/final_verifications", "logs_private",
    ):
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def parse_content(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text or "").strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Stage output is not a JSON object")
    return value


def call_api(base_url: str, api_key: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc


def run_json_stage(
    *,
    rows: list[dict[str, Any]],
    prompt_path: Path,
    schema_path: Path,
    parsed_path: Path,
    raw_path: Path,
    provider: str,
    model: str,
    max_tokens: int,
    build_user: Callable[[dict[str, Any], dict[str, Any]], str],
    semantic_errors: Callable[[dict[str, Any], dict[str, Any]], list[str]],
    dry_run: bool,
    max_rows: int | None,
    workers: int,
    timeout: int,
    retries: int,
    stage_name: str,
    max_failures: int = 0,
) -> dict[str, Any]:
    selected = rows[:max_rows] if max_rows else rows
    if not selected:
        raise ValueError(f"{stage_name} has no input rows")
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    first_user = build_user(selected[0], schema)
    if dry_run or os.environ.get("RUN_API") != "1":
        summary = {
            "status": "DRY_RUN", "stage": stage_name, "provider": provider, "model": model,
            "rows": len(selected), "api_called": False,
            "request_hash": stable_hash({"prompt": prompt, "schema": schema, "user": first_user}),
            "dev_access_count": 0, "test_access_count": 0,
        }
        print(json.dumps(summary, sort_keys=True))
        return summary

    env_key = "QWEN_API_KEY" if provider == "qwen" else "DEEPSEEK_API_KEY"
    api_key = os.environ.get(env_key, "")
    if not api_key:
        raise SystemExit(f"Missing {env_key}")
    base_url = os.environ.get(
        "QWEN_BASE_URL" if provider == "qwen" else "DEEPSEEK_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1" if provider == "qwen" else "https://api.deepseek.com",
    )
    existing = read_jsonl(parsed_path) if parsed_path.exists() else []
    selected_ids = {sample_id(row) for row in selected}
    existing_by_id = {sample_id(row): row for row in existing if sample_id(row) in selected_ids}
    pending = [row for row in selected if sample_id(row) not in existing_by_id]
    started = time.time()

    def worker(row: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], list[str]]:
        sid = sample_id(row)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": build_user(row, schema)},
        ]
        body: dict[str, Any] = {
            "model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if provider == "qwen":
            body["enable_thinking"] = False
        else:
            body["thinking"] = {"type": "disabled"}
        for attempt in range(1, retries + 1):
            content = ""
            try:
                response = call_api(base_url, api_key, body, timeout)
                choice = (response.get("choices") or [{}])[0]
                content = str((choice.get("message") or {}).get("content") or "")
                if not content.strip():
                    raise ValueError(f"empty_final_content finish_reason={choice.get('finish_reason')}")
                value = parse_content(content)
                normalized = []
                for field in ("sample_id", "source_sample_id"):
                    if field in value and field in row and str(value[field]) != str(row[field]):
                        value[field] = str(row[field])
                        normalized.append(field)
                errors = [error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(value)]
                errors.extend(semantic_errors(value, row))
                if errors:
                    raise ValueError("; ".join(errors))
                return sid, response, value, normalized
            except Exception as exc:
                if attempt == retries:
                    raise RuntimeError(f"{stage_name} failed for {sid}: {type(exc).__name__}: {exc}") from exc
                if content:
                    body["messages"] = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": (
                            f"The previous JSON failed validation: {type(exc).__name__}: {exc}. "
                            "Correct only the validation failure and return exactly one schema-valid JSON object."
                        )},
                    ]
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    completed = set(existing_by_id)
    normalized_count = 0
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(worker, row): sample_id(row) for row in pending}
        session_done = 0
        for future in as_completed(futures):
            try:
                sid, response, value, normalized = future.result()
            except Exception as exc:
                sid = futures[future]
                failures.append((sid, f"{type(exc).__name__}: {exc}"))
                print(f"[{stage_name}] deferred_failure sample_id={sid}", flush=True)
                continue
            append_jsonl(raw_path, {"sample_id": sid, "response": response})
            append_jsonl(parsed_path, value)
            completed.add(sid)
            normalized_count += bool(normalized)
            session_done += 1
            elapsed = time.time() - started
            eta = elapsed / max(session_done, 1) * max(len(pending) - session_done, 0)
            print(
                f"[{stage_name}] {len(completed)}/{len(selected)} workers={max(1, workers)} "
                f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )
    if len(failures) > max_failures:
        preview = "; ".join(f"{sid}: {error}" for sid, error in failures[:5])
        raise RuntimeError(f"{stage_name} has {len(failures)} failures; rerun to resume. {preview}")
    return {
        "status": "COMPLETED", "stage": stage_name, "provider": provider, "model": model,
        "rows": len(completed), "expected_rows": len(selected),
        "schema_success_rate": len(completed) / len(selected),
        "failure_count": len(failures), "max_failures": max_failures,
        "routing_identity_normalization_count": normalized_count,
        "dev_access_count": 0, "test_access_count": 0,
    }


def locate_occurrence(text: str, span: str, occurrence: int) -> tuple[int, int]:
    if not span:
        raise ValueError("Empty source span")
    starts = [match.start() for match in re.finditer(re.escape(span), text)]
    if occurrence < 0 or occurrence >= len(starts):
        raise ValueError(f"Span occurrence {occurrence} not found; matches={len(starts)}")
    start = starts[occurrence]
    return start, start + len(span)


def patch_locked_span(source: str, span: str, occurrence: int, replacement: str) -> dict[str, Any]:
    start, end = locate_occurrence(source, span, occurrence)
    candidate = source[:start] + replacement + source[end:]
    outside_identity = candidate[:start] == source[:start] and candidate[start + len(replacement):] == source[end:]
    return {"candidate": candidate, "start": start, "end": end, "outside_span_identity": outside_identity}


def edit_budget(source: str, span: str, replacement: str, operator: str) -> dict[str, Any]:
    source_tokens = max(token_count(source), 1)
    span_tokens = token_count(span)
    replacement_tokens = token_count(replacement)
    final_tokens = source_tokens - span_tokens + replacement_tokens
    final_ratio = final_tokens / source_tokens
    span_ratio = span_tokens / source_tokens
    inserted_ratio = max(0, replacement_tokens - span_tokens) / source_tokens
    if operator == "replace_with_local_contradiction":
        passed = span_ratio <= 0.20 and 0.85 <= final_ratio <= 1.15
    elif operator in {"delete_required_span", "delete_supporting_reasoning"}:
        passed = 0.03 <= span_ratio <= 0.25 and 0.75 <= final_ratio <= 1.00
    elif operator == "insert_local_scope_drift":
        passed = inserted_ratio <= 0.15 and 1.00 <= final_ratio <= 1.15
    elif operator == "violate_explicit_constraint":
        passed = max(span_tokens, replacement_tokens) / source_tokens <= 0.15 and 0.85 <= final_ratio <= 1.15
    else:
        passed = False
    return {
        "source_tokens": source_tokens, "span_tokens": span_tokens, "replacement_tokens": replacement_tokens,
        "span_token_ratio": span_ratio, "inserted_token_ratio": inserted_ratio,
        "final_token_length_ratio": final_ratio, "edit_budget_pass": passed,
    }


def range_intersects(left: Iterable[int], right: Iterable[int]) -> bool:
    a, b = [int(value) for value in left]
    c, d = [int(value) for value in right]
    return max(a, c) <= min(b, d)


def soft_target(minimum: int, maximum: int, center: int) -> list[float]:
    weights = [math.exp(-abs(score - center)) if minimum <= score <= maximum else 0.0 for score in range(1, 6)]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Invalid final verifier interval")
    return [weight / total for weight in weights]
