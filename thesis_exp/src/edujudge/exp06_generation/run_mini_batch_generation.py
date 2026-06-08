"""Run or dry-run the Exp6-3 mini-batch generation step.

Dry-run is the default and never calls an API. Generation mode requires
EXP6_RUN_GENERATION=1 plus GENERATION_MODEL and either an API key or a local endpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis_exp.src.edujudge.exp06_generation import MINI_BATCH_TOTAL_TARGET, ensure_mini_batch_dirs
from thesis_exp.src.edujudge.exp06_generation.mini_batch_common import (
    mini_generated_path,
    mini_prompt_path,
    mini_report_path,
    read_jsonl_if_exists,
    write_mini_jsonl,
)
from thesis_exp.src.edujudge.utils.io import relpath, write_text


RAW_FIELDS = [
    "synthetic_plan_id",
    "source_record_id",
    "raw_prompt",
    "raw_response",
    "generation_model",
    "generation_status",
    "error_message",
    "created_at",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def prompt_rows(max_items: int) -> list[dict[str, Any]]:
    rows = read_jsonl_if_exists(mini_prompt_path("mini_batch_prompts.jsonl"))
    if len(rows) > MINI_BATCH_TOTAL_TARGET:
        rows = rows[:MINI_BATCH_TOTAL_TARGET]
    return rows[:max_items]


def generation_env() -> dict[str, str]:
    return {
        "model": os.getenv("GENERATION_MODEL", ""),
        "endpoint": os.getenv("GENERATION_ENDPOINT", ""),
        "api_key": os.getenv("GENERATION_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
    }


def write_dry_run_report(max_items: int) -> None:
    prompts = prompt_rows(max_items)
    report = f"""# Exp6-3 Mini-batch Dry Run

Mode: **DRY_RUN**

No API was called and no synthetic answers were generated.

- Prompt rows available: **{len(prompts)}**
- Prompt path: `{relpath(mini_prompt_path("mini_batch_prompts.jsonl"))}`
- Raw generation file written: **NO**
- Next action: review prompts, set generation environment only if approved, then run generation for at
  most 24 rows.
"""
    write_text(mini_report_path("DRY_RUN_NO_GENERATION.md"), report)


def write_blocked_report(reason: str) -> None:
    report = f"""# BLOCKED: Exp6-3 Mini-batch Generation Did Not Run

Generation mode was requested, but no synthetic answers were generated.

Reason: **{reason}**

Required environment:

- `EXP6_RUN_GENERATION=1`
- `GENERATION_MODEL`
- `GENERATION_API_KEY` or `DEEPSEEK_API_KEY`, unless `GENERATION_ENDPOINT` is a local endpoint that
  needs no key

The runner is capped at 24 prompt rows and never logs API keys.
"""
    write_text(mini_report_path("BLOCKED_NO_GENERATION.md"), report)


class HardTimeoutError(TimeoutError):
    """Raised when a generation request exceeds the hard wall-clock timeout."""


@contextlib.contextmanager
def hard_timeout(seconds: int):
    def handler(signum: int, frame: Any) -> None:
        raise HardTimeoutError(f"request exceeded {seconds}s hard timeout")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(max(1, int(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def max_tokens_from_env(default: int = 2048) -> int:
    raw = os.getenv("GENERATION_MAX_TOKENS", "")
    if not raw:
        return default
    try:
        return max(256, int(raw))
    except ValueError:
        return default


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("model content is JSON but not an object")
    return parsed


def validate_generated_content(text: str) -> None:
    if not text.strip():
        raise ValueError("empty model content")
    parsed = parse_json_object(text)
    answer = str(parsed.get("answer_synthetic") or "").strip()
    if not answer:
        raise ValueError("missing answer_synthetic")
    if "target_label_5" not in parsed:
        raise ValueError("missing target_label_5")
    if "error_type" not in parsed:
        raise ValueError("missing error_type")


def existing_row_is_usable(row: dict[str, Any]) -> bool:
    if row.get("generation_status") != "generated":
        return False
    try:
        validate_generated_content(str(row.get("raw_response") or ""))
        return True
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


def post_chat_completion(endpoint: str, api_key: str, model: str, prompt: str, timeout: int = 45) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": max_tokens_from_env(),
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with hard_timeout(timeout):
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices") or []
    if not choices:
        return json.dumps(body, ensure_ascii=False)
    message = choices[0].get("message") or {}
    return str(message.get("content") or choices[0].get("text") or "")


def load_existing_by_plan(raw_path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not raw_path.exists():
        return existing
    for row in read_jsonl_if_exists(raw_path):
        plan_id = str(row.get("synthetic_plan_id") or "")
        if plan_id:
            existing[plan_id] = row
    return existing


def rewrite_raw_file(raw_path: Path, rows_by_plan: dict[str, dict[str, Any]], plan_order: list[str]) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as handle:
        for plan_id in plan_order:
            row = rows_by_plan.get(plan_id)
            if row:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_generation(max_items: int, retries: int = 2, request_timeout: int = 45, resume: bool = True) -> list[dict[str, Any]]:
    env = generation_env()
    model = env["model"]
    endpoint = env["endpoint"] or "https://api.deepseek.com/chat/completions"
    api_key = env["api_key"]
    if not model:
        write_blocked_report("missing GENERATION_MODEL")
        raise SystemExit(2)
    if not api_key and not env["endpoint"]:
        write_blocked_report("missing API key or local GENERATION_ENDPOINT")
        raise SystemExit(2)

    rows = prompt_rows(max_items)
    if not rows:
        write_blocked_report("mini_batch_prompts.jsonl is missing or empty")
        raise SystemExit(2)
    if len(rows) > MINI_BATCH_TOTAL_TARGET:
        rows = rows[:MINI_BATCH_TOTAL_TARGET]

    raw_path = mini_generated_path("raw_generations.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    plan_order = [str(row.get("synthetic_plan_id") or "") for row in rows]
    rows_by_plan = load_existing_by_plan(raw_path) if resume else {}
    rows_by_plan = {plan_id: row for plan_id, row in rows_by_plan.items() if plan_id in set(plan_order)}
    rewrite_raw_file(raw_path, rows_by_plan, plan_order)
    for index, row in enumerate(rows, start=1):
        plan_id = str(row.get("synthetic_plan_id") or "")
        existing = rows_by_plan.get(plan_id)
        if existing and existing_row_is_usable(existing):
            print(f"generation_progress row={index}/{len(rows)} status=skipped_existing", flush=True)
            continue
        prompt = row.get("prompt_text", "")
        status = "failed"
        raw_response = ""
        error_message = ""
        for attempt in range(retries + 1):
            try:
                raw_response = post_chat_completion(endpoint, api_key, model, prompt, timeout=request_timeout)
                validate_generated_content(raw_response)
                status = "generated"
                error_message = ""
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, HardTimeoutError, json.JSONDecodeError, OSError, ValueError) as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    time.sleep(2**attempt)
        output_row = {
            "synthetic_plan_id": row.get("synthetic_plan_id", ""),
            "source_record_id": row.get("source_record_id", ""),
            "raw_prompt": prompt,
            "raw_response": raw_response,
            "generation_model": model,
            "generation_status": status,
            "error_message": error_message,
            "created_at": now_iso(),
        }
        rows_by_plan[plan_id] = output_row
        rewrite_raw_file(raw_path, rows_by_plan, plan_order)
        print(f"generation_progress row={index}/{len(rows)} status={status}", flush=True)
    return [rows_by_plan[plan_id] for plan_id in plan_order if plan_id in rows_by_plan]


def run(mode: str = "dry_run", max_items: int = MINI_BATCH_TOTAL_TARGET, request_timeout: int = 45, resume: bool = True) -> str:
    ensure_mini_batch_dirs()
    if max_items > MINI_BATCH_TOTAL_TARGET:
        raise SystemExit("Mini-batch generation is capped at 24 rows.")
    requested = mode == "generate" or os.getenv("EXP6_RUN_GENERATION") == "1"
    if not requested:
        write_dry_run_report(max_items)
        return "DRY_RUN"
    rows = run_generation(max_items=max_items, request_timeout=request_timeout, resume=resume)
    write_text(
        mini_report_path("GENERATION_RUN_SUMMARY.md"),
        f"""# Exp6-3 Mini-batch Generation Run Summary

- Raw output path: `{relpath(mini_generated_path("raw_generations.jsonl"))}`
- Requested rows: **{min(max_items, MINI_BATCH_TOTAL_TARGET)}**
- Rows written: **{len(rows)}**
- Generated rows: **{sum(1 for row in rows if row.get("generation_status") == "generated")}**
- Failed rows: **{sum(1 for row in rows if row.get("generation_status") != "generated")}**
""",
    )
    return "GENERATED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry_run", "generate"], default="dry_run")
    parser.add_argument("--max-items", "--max_items", dest="max_items", type=int, default=MINI_BATCH_TOTAL_TARGET)
    parser.add_argument("--request-timeout", type=int, default=45)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    result = run(mode=args.mode, max_items=args.max_items, request_timeout=args.request_timeout, resume=args.resume)
    print(f"Mini-batch generation runner finished with mode={result}")


if __name__ == "__main__":
    main()
