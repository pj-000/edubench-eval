"""Dry-run or execute the frozen answer-blind Qwen rubric compiler."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jsonschema
except ModuleNotFoundError:
    from thesis_exp.exp17_low_score_evidence import json_schema_compat as jsonschema

from thesis_exp.exp41_rubric_bridge.common import (  # noqa: E402
    FORBIDDEN_UNIT_FIELDS, ROOT, append_jsonl, lexical_tokens, normalize_text, read_jsonl,
    stable_hash, write_csv, write_json, write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("API_WORKERS", "8")))
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    return parser.parse_args()


def parse_content(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Compiler output is not a JSON object")
    return value


def call_api(base_url: str, key: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:1000]}") from exc


def user_text(unit: dict[str, Any], schema: dict[str, Any]) -> str:
    payload = {
        "rubric_unit_id": unit["rubric_unit_id"], "question": unit["question"], "metric": unit["metric"],
        "raw_rubric": unit["raw_rubric"], "language": unit["language"], "metadata": unit["metadata"],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if any(f'"{field}"' in serialized for field in FORBIDDEN_UNIT_FIELDS):
        raise RuntimeError("Compiler request contains forbidden answer/label fields")
    return "Compilation input:\n" + serialized + "\n\nJSON schema:\n" + json.dumps(schema, ensure_ascii=False, sort_keys=True)


def deterministic_response_errors(unit: dict[str, Any], value: dict[str, Any]) -> list[str]:
    errors = []
    raw = normalize_text(unit["raw_rubric"])
    criteria = value.get("criteria") if isinstance(value.get("criteria"), list) else []
    ids = [str(item.get("criterion_id")) for item in criteria if isinstance(item, dict)]
    quotes = [normalize_text(item.get("rubric_quote", "")) for item in criteria if isinstance(item, dict)]
    if len(ids) != len(set(ids)):
        errors.append("criterion IDs must be unique")
    if len(quotes) != len(set(quotes)):
        errors.append("criterion rubric_quote values must be unique")
    invalid_quotes = [quote for quote in quotes if not quote or quote not in raw]
    if invalid_quotes:
        errors.append(f"rubric_quote is not an exact normalized continuous raw-rubric substring: {invalid_quotes[:2]}")
    if any(not str(item.get("check_question", "")).strip().endswith(("?", "？")) for item in criteria if isinstance(item, dict)):
        errors.append("every check_question must end with ? or ？")
    raw_tokens = lexical_tokens(raw)
    covered = set().union(*(lexical_tokens(quote) for quote in quotes)) if quotes else set()
    coverage = len(raw_tokens & covered) / max(len(raw_tokens), 1)
    if coverage < 0.60:
        errors.append(f"exact-quote lexical coverage must be >=0.60, got {coverage:.4f}")
    id_set = set(ids)
    for rule in value.get("score_level_rules", []):
        quote = normalize_text(rule.get("rubric_quote", ""))
        level = int(rule.get("score_level", 0) or 0)
        if quote not in raw:
            errors.append(f"score-level quote is not an exact raw-rubric substring: {quote[:80]}")
        if not re.search(rf"(?:^|\D){level}\s*(?:分|[:：.)、-])", quote):
            errors.append(f"score-level {level} is not explicitly supported by its quote")
        if not set(map(str, rule.get("required_criteria", []))) <= id_set:
            errors.append(f"score-level {level} references an unknown criterion")
    return errors


def compile_one(unit: dict[str, Any], *, prompt: str, schema: dict[str, Any], model: str, base_url: str, key: str,
                timeout: int, retries: int, max_output_tokens: int) -> tuple[str, dict[str, Any], dict[str, Any], bool]:
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_text(unit, schema)}]
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_output_tokens,
            "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}}
    for attempt in range(1, retries + 1):
        response_content = ""
        try:
            response = call_api(base_url, key, body, timeout)
            response_content = str(response["choices"][0]["message"]["content"] or "")
            if not response_content.strip():
                raise ValueError("empty_final_content")
            value = parse_content(response_content)
            normalized_identity = str(value.get("rubric_unit_id")) != str(unit["rubric_unit_id"])
            if normalized_identity:
                value["rubric_unit_id"] = unit["rubric_unit_id"]
            errors = [error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(value)]
            errors.extend(deterministic_response_errors(unit, value))
            if errors:
                raise ValueError("; ".join(errors))
            return unit["rubric_unit_id"], response, value, normalized_identity
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Qwen rubric compilation failed for {unit['rubric_unit_id']}: {type(exc).__name__}: {exc}") from exc
            if response_content:
                corrective = (
                    f"The previous JSON failed deterministic validation: {type(exc).__name__}: {exc}.\n"
                    "Rebuild the JSON. The ONLY legal source for every rubric_quote is the literal raw_rubric block below; "
                    "never quote or paraphrase the question. Remove every question-derived criterion. Copy each rubric_quote "
                    "character-for-character from this block. It is valid to return an empty score_level_rules list; if a score "
                    "rule is used, its rubric_quote must retain the explicit level marker such as '5:' or '2:'. Ensure the union "
                    "of exact quotes covers at least 60% of unique raw-rubric tokens. Return JSON only.\n"
                    f"<raw_rubric>\n{unit['raw_rubric']}\n</raw_rubric>"
                )
                body["messages"] = messages + [
                    {"role": "assistant", "content": response_content},
                    {"role": "user", "content": corrective},
                ]
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    lock_path = args.out_dir / "configs/exp41a_compiler_protocol_lock.json"
    unit_path = args.out_dir / "private/rubric_units/exp41a_rubric_units.jsonl"
    if not lock_path.exists() or not unit_path.exists():
        raise FileNotFoundError("Run prepare_exp41a_rubric_units.py first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    units = read_jsonl(unit_path)
    if args.max_units:
        units = units[: args.max_units]
    prompt = (args.out_dir / "prompts/exp41a_answer_blind_rubric_compiler.md").read_text(encoding="utf-8")
    schema = json.loads((args.out_dir / "schemas/exp41a_compiled_rubric_schema.json").read_text(encoding="utf-8"))
    request_preview = user_text(units[0], schema)
    dry_run = args.dry_run or os.environ.get("RUN_API") != "1"
    if dry_run:
        print(json.dumps({"status": "DRY_RUN", "units": len(units), "model": lock["model"], "api_called": False,
                          "answer_blind": True, "label_blind": True,
                          "request_hash": stable_hash({"prompt": prompt, "request": request_preview})}, indent=2, sort_keys=True))
        return
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing QWEN_API_KEY")
    base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    parsed_path = args.out_dir / "private/compiled_rubrics/exp41a_qwen_compiled_rubrics.jsonl"
    raw_path = args.out_dir / "raw_api/exp41a_qwen_rubric_compiler.jsonl"
    normalization_path = args.out_dir / "private/compiled_rubrics/exp41a_routing_identity_normalizations.jsonl"
    existing = read_jsonl(parsed_path) if parsed_path.exists() else []
    by_id = {str(row["rubric_unit_id"]): row for row in existing}
    normalizations = read_jsonl(normalization_path) if normalization_path.exists() else []
    normalized_ids = {str(row["rubric_unit_id"]) for row in normalizations}
    pending = [unit for unit in units if unit["rubric_unit_id"] not in by_id]
    started = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {executor.submit(compile_one, unit, prompt=prompt, schema=schema, model=lock["model"], base_url=base_url,
                                   key=api_key, timeout=args.timeout, retries=args.retries,
                                   max_output_tokens=args.max_output_tokens): unit["rubric_unit_id"] for unit in pending}
        session_done = 0
        for future in as_completed(futures):
            unit_id = futures[future]
            try:
                unit_id, response, value, normalized = future.result()
            except Exception as exc:
                failures.append((unit_id, f"{type(exc).__name__}: {exc}"))
                print(f"[exp41a-compiler] deferred_failure unit={unit_id} error={type(exc).__name__}:{exc}", flush=True)
                continue
            append_jsonl(raw_path, {"rubric_unit_id": unit_id, "response": response})
            by_id[unit_id] = value
            append_jsonl(parsed_path, value)
            if normalized and unit_id not in normalized_ids:
                append_jsonl(normalization_path, {"rubric_unit_id": unit_id, "normalization_reason": "routing_identity_only"})
                normalized_ids.add(unit_id)
            session_done += 1
            elapsed = time.time() - started
            completed = len(units) - len(pending) + session_done
            eta = elapsed / max(session_done, 1) * max(len(pending) - session_done, 0)
            if session_done == 1 or session_done % 10 == 0 or session_done == len(pending):
                print(f"[exp41a-compiler] {completed}/{len(units)} workers={args.workers} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    write_jsonl(parsed_path, [by_id[key] for key in sorted(by_id)])
    if failures:
        raise RuntimeError(f"Exp41A has {len(failures)} deferred failures; rerun to resume. {failures[:3]}")
    selected_ids = {unit["rubric_unit_id"] for unit in units}
    completed = sum(unit_id in by_id for unit_id in selected_ids)
    summary = {"status": "COMPLETED" if completed == len(units) else "INCOMPLETE", "provider": "qwen", "model": lock["model"],
               "expected_outputs": len(units), "completed_outputs": completed, "schema_success_rate": completed / max(len(units), 1),
               "routing_identity_normalization_count": len(normalized_ids & selected_ids), "answer_blind": True, "label_blind": True,
               "dev_access_count": 0, "test_access_count": 0}
    write_json(args.out_dir / "decision/exp41a_compiler_api_summary.json", summary)
    write_csv(args.out_dir / "tables/exp41a_compiler_completion.csv", [{"stage": "qwen_compilation", **summary}])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
