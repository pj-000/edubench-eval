"""Dry-run or execute the frozen Exp38A Qwen score-range protocol."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jsonschema
except ModuleNotFoundError:
    from thesis_exp.exp17_low_score_evidence import json_schema_compat as jsonschema

from thesis_exp.exp38_hails_score.common import ROOT, append_jsonl, read_jsonl, sample_id, stable_hash, write_json, write_jsonl

PROMPT_PATH = Path("thesis_exp/exp38_hails_score/prompts/exp38a_qwen_score_range_prompt.md")
SCHEMA_PATH = Path("thesis_exp/exp38_hails_score/schemas/exp38a_qwen_score_range_schema.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("qualification", "all_train"), required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--model")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def parse_content(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("API output is not a JSON object")
    return value


def semantic_errors(value: dict[str, Any], expected_id: str) -> list[str]:
    errors = []
    if str(value.get("sample_id")) != expected_id:
        errors.append("sample_id_mismatch")
    scores = [value.get("minimum_plausible_score"), value.get("most_plausible_score"), value.get("maximum_plausible_score")]
    if all(isinstance(score, int) for score in scores) and not (scores[0] <= scores[1] <= scores[2]):
        errors.append("range_order_invalid")
    return errors


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


def main() -> None:
    args = parse_args()
    lock = json.loads((args.out_dir / "configs/exp38a_range_protocol_lock.json").read_text(encoding="utf-8"))
    if args.split == "all_train":
        decision = json.loads((args.out_dir / "decision/exp38a_range_qualification_decision.json").read_text(encoding="utf-8"))
        if not decision.get("recommend_full_train_range_annotation"):
            raise RuntimeError("Full-train API is blocked by qualification NO-GO")
    packet_name = "exp38a_qwen_range_qualification_packets.jsonl" if args.split == "qualification" else "exp38a_qwen_range_all_train_packets.jsonl"
    packets = read_jsonl(args.out_dir / "private_reference" / packet_name)
    if args.max_rows:
        packets = packets[: args.max_rows]
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    model = args.model or lock["model"]
    if model != lock["model"]:
        raise RuntimeError(f"Model differs from frozen protocol: {model} != {lock['model']}")
    dry_run = args.dry_run or os.environ.get("RUN_API") != "1"
    if dry_run:
        first = packets[0]
        summary = {
            "status": "DRY_RUN", "split": args.split, "rows": len(packets), "model": model,
            "request_hash": stable_hash({"prompt": prompt, "schema": schema, "packet": first}),
            "sample_id": sample_id(first), "api_called": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing QWEN_API_KEY")
    base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    raw_path = args.out_dir / f"raw_api/exp38a_qwen_ranges_{args.split}.jsonl"
    parsed_path = args.out_dir / f"parsed_ranges_private/exp38a_qwen_ranges_{args.split}.jsonl"
    existing_rows = read_jsonl(parsed_path) if parsed_path.exists() else []
    existing_by_id = {sample_id(row): row for row in existing_rows}
    if len(existing_by_id) != len(existing_rows):
        write_jsonl(parsed_path, [existing_by_id[sid] for sid in sorted(existing_by_id)])
    completed = set(existing_by_id)
    start = time.time()
    for index, packet in enumerate(packets, 1):
        sid = sample_id(packet)
        if sid in completed:
            continue
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": packet["review_text"] + "\n\nJSON schema:\n" + json.dumps(schema, ensure_ascii=False, sort_keys=True)},
        ]
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": int(lock["max_tokens"]),
            "enable_thinking": bool(lock["enable_thinking"]),
            "response_format": {"type": "json_object"},
        }
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                response = call_api(base_url, api_key, body, args.timeout)
                content = response["choices"][0]["message"]["content"]
                value = parse_content(content)
                schema_errors = [error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(value)]
                schema_errors.extend(semantic_errors(value, sid))
                if schema_errors:
                    raise ValueError("; ".join(schema_errors))
                append_jsonl(raw_path, {"sample_id": sid, "response": response})
                append_jsonl(parsed_path, value)
                completed.add(sid)
                break
            except Exception as exc:  # network/schema failures are retried and surfaced
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == args.retries:
                    raise RuntimeError(f"Qwen range failed for {sid}: {last_error}") from exc
                time.sleep(2 ** attempt)
        elapsed = time.time() - start
        done = len(completed)
        eta = elapsed / max(done, 1) * max(len(packets) - done, 0)
        print(f"[exp38a-range] {args.split} {done}/{len(packets)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    write_json(args.out_dir / f"decision/exp38a_{args.split}_api_summary.json", {
        "status": "COMPLETED", "split": args.split, "rows": len(completed), "model": model,
        "prompt_sha256": lock["prompt_sha256"], "schema_sha256": lock["schema_sha256"],
        "temperature": 0, "enable_thinking": lock["enable_thinking"], "max_tokens": lock["max_tokens"],
        "dev_access_count": 0, "test_access_count": 0,
    })
    print(json.dumps({"status": "COMPLETED", "split": args.split, "rows": len(completed)}, sort_keys=True))


if __name__ == "__main__":
    main()
