"""Dry-run or execute frozen Qwen counterfactual generation for Exp39A."""

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

from thesis_exp.exp39_educfa.common import (  # noqa: E402
    PROMPT_DIR,
    ROOT,
    SCHEMA_DIR,
    append_jsonl,
    read_jsonl,
    sample_id,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)

PROMPT_PATH = PROMPT_DIR / "exp39a_qwen_counterfactual_generator.md"
SCHEMA_PATH = SCHEMA_DIR / "exp39a_counterfactual_generation_schema.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--model")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("API_WORKERS", "4")))
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
        raise ValueError("Generator output is not a JSON object")
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
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:1000]}") from exc


def user_text(packet: dict[str, Any], schema: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"Sample ID: {packet['sample_id']}",
            f"Source sample ID: {packet['source_sample_id']}",
            "<CONTEXT_ONLY_ORIGINAL_TASK>\n" + str(packet["question"]) + "\n</CONTEXT_ONLY_ORIGINAL_TASK>",
            "<ORIGINAL_HIGH_SCORE_OUTPUT>\n" + str(packet["original_answer"]) + "\n</ORIGINAL_HIGH_SCORE_OUTPUT>",
            f"Evaluation dimension: {packet['metric']}",
            "Rubric: " + json.dumps(packet["rubric"], ensure_ascii=False),
            "Non-label metadata: " + json.dumps(packet["metadata"], ensure_ascii=False, sort_keys=True),
            f"Assigned operator: {packet['assigned_operator']}",
            f"Target score: {packet['assigned_target_score']}",
            "JSON schema: " + json.dumps(schema, ensure_ascii=False, sort_keys=True),
        ]
    )


def semantic_errors(value: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    errors = []
    if str(value.get("sample_id")) != str(packet["sample_id"]):
        errors.append("sample_id_mismatch")
    if str(value.get("source_sample_id")) != str(packet["source_sample_id"]):
        errors.append("source_sample_id_mismatch")
    if value.get("operator") != packet["assigned_operator"]:
        errors.append("operator_mismatch")
    if value.get("target_score") != packet["assigned_target_score"]:
        errors.append("target_score_mismatch")
    score_range = value.get("target_score_range") or []
    if len(score_range) == 2 and not (score_range[0] <= value.get("target_score", 0) <= score_range[1]):
        errors.append("target_not_in_generator_range")
    if str(value.get("counterfactual_answer") or "").strip() == str(packet["original_answer"]).strip():
        errors.append("counterfactual_equals_source")
    return errors


def generate_one(
    packet: dict[str, Any],
    *,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    max_tokens: int,
    base_url: str,
    api_key: str,
    timeout: int,
    retries: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    sid = str(packet["sample_id"])
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_text(packet, schema)},
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(1, retries + 1):
        response_content = ""
        try:
            response = call_api(base_url, api_key, body, timeout)
            response_content = response["choices"][0]["message"]["content"]
            value = parse_content(response_content)
            errors = [error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(value)]
            errors.extend(semantic_errors(value, packet))
            if errors:
                raise ValueError("; ".join(errors))
            return sid, response, value
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Qwen generation failed for {sid}: {type(exc).__name__}: {exc}") from exc
            if response_content:
                body["messages"] = messages + [
                    {"role": "assistant", "content": response_content},
                    {
                        "role": "user",
                        "content": (
                            "The previous JSON failed automatic validation: "
                            f"{type(exc).__name__}: {exc}. Correct only that failure, execute the already "
                            "assigned counterfactual operator, and return one schema-valid JSON object."
                        ),
                    },
                ]
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    lock = json.loads((args.out_dir / "configs/exp39a_generation_protocol_lock.json").read_text(encoding="utf-8"))
    packets = read_jsonl(args.out_dir / "private/source_packets/exp39a_source_anchor_packets.jsonl")
    if args.max_rows:
        packets = packets[: args.max_rows]
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    model = args.model or lock["qwen_model"]
    if model != lock["qwen_model"]:
        raise RuntimeError(f"Qwen model differs from frozen protocol: {model}")
    dry_run = args.dry_run or os.environ.get("RUN_API") != "1"
    if dry_run:
        first = packets[0]
        print(json.dumps({
            "status": "DRY_RUN", "rows": len(packets), "model": model, "api_called": False,
            "sample_id": first["sample_id"],
            "request_hash": stable_hash({"prompt": prompt, "schema": schema, "user": user_text(first, schema)}),
        }, indent=2, sort_keys=True))
        return
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing QWEN_API_KEY")
    base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    raw_path = args.out_dir / "raw_api/exp39a_qwen_counterfactual_generation.jsonl"
    parsed_path = args.out_dir / "private/generated_candidates/exp39a_qwen_generated_candidates.jsonl"
    existing = read_jsonl(parsed_path) if parsed_path.exists() else []
    packet_ids = {str(packet["sample_id"]) for packet in packets}
    all_existing_by_id = {sample_id(row): row for row in existing}
    if len(existing) != len(all_existing_by_id) and args.max_rows is None:
        write_jsonl(parsed_path, [all_existing_by_id[key] for key in sorted(all_existing_by_id)])
    existing_by_id = {sid: row for sid, row in all_existing_by_id.items() if sid in packet_ids}
    completed = set(existing_by_id)
    pending = [packet for packet in packets if str(packet["sample_id"]) not in completed]
    workers = max(1, args.workers)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                generate_one,
                packet,
                model=model,
                prompt=prompt,
                schema=schema,
                max_tokens=int(lock["qwen_max_tokens"]),
                base_url=base_url,
                api_key=api_key,
                timeout=args.timeout,
                retries=args.retries,
            ): str(packet["sample_id"])
            for packet in pending
        }
        session_completed = 0
        try:
            for future in as_completed(futures):
                sid, response, value = future.result()
                append_jsonl(raw_path, {"sample_id": sid, "response": response})
                append_jsonl(parsed_path, value)
                completed.add(sid)
                session_completed += 1
                elapsed = time.time() - started
                eta = elapsed / max(session_completed, 1) * max(len(pending) - session_completed, 0)
                print(
                    f"[exp39a-qwen] {len(completed)}/{len(packets)} workers={workers} "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
        except Exception:
            for future in futures:
                future.cancel()
            raise
    summary = {
        "status": "COMPLETED", "provider": "qwen", "model": model, "rows": len(completed),
        "expected_rows": len(packets), "schema_success_rate": 1.0,
        "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp39a_qwen_generation_api_summary.json", summary)
    write_csv(args.out_dir / "tables/exp39a_generation_completion.csv", [summary])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
