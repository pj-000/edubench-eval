"""Run resumable Qwen Batch annotation for all 2,654 paper-train rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import jsonschema
except ModuleNotFoundError:
    from thesis_exp.exp17_low_score_evidence import json_schema_compat as jsonschema

from thesis_exp.exp17_low_score_evidence.run_exp28b_teacher_protocol_api import (
    SCHEMA_PATH,
    build_messages,
    normalize_annotation_structure,
    parse_json,
    read_jsonl,
    response_text,
)


PACKETS = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28b_paper_train_annotation_inventory_seed42/"
    "private/exp28b_blind_teacher_packets_2654.jsonl"
)
OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42"
)
PROTOCOL = "p0_holistic_zero_shot"
MODEL = "qwen3.7-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def batch_body(messages: list[dict[str, str]], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 8192,
        "enable_thinking": True,
    }


def prepare_input(args: argparse.Namespace, packets: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    rows = []
    for packet in packets:
        rows.append(
            {
                "custom_id": str(packet["sample_id"]),
                "method": "POST",
                "url": args.endpoint,
                "body": batch_body(build_messages(packet, args.protocol, schema), args.model),
            }
        )
    write_jsonl(args.batch_input, rows)


def client_for(args: argparse.Namespace) -> OpenAI:
    key = os.environ.get("QWEN_API_KEY", "")
    if not key:
        raise SystemExit("Missing QWEN_API_KEY; supply it through the environment")
    return OpenAI(api_key=key, base_url=args.base_url)


def load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    write_json(path, state)


def download_file(client: OpenAI, file_id: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    content.write_to_file(path)


def parse_results(
    args: argparse.Namespace,
    packets: list[dict[str, Any]],
    schema: dict[str, Any],
    batch_id: str,
) -> dict[str, Any]:
    raw_rows = read_jsonl(args.batch_result)
    raw_by_id = {str(row.get("custom_id")): row for row in raw_rows}
    validator = jsonschema.Draft202012Validator(schema)
    parsed_rows: list[dict[str, Any]] = []
    valid = 0
    for packet in packets:
        sid = str(packet["sample_id"])
        raw = raw_by_id.get(sid) or {}
        response = raw.get("response") or {}
        body = response.get("body") or {}
        status_code = response.get("status_code")
        annotation = None
        parse_error = ""
        actions: list[str] = []
        schema_errors: list[str] = []
        if status_code == 200:
            annotation, parse_error = parse_json(response_text(body))
            annotation, actions = normalize_annotation_structure(annotation, sid, schema)
            if annotation is not None:
                schema_errors = [error.message for error in validator.iter_errors(annotation)]
                if annotation.get("sample_id") != sid:
                    schema_errors.append("sample_id mismatch")
                failures = annotation.get("major_failures") or []
                if "no_major_failure" in failures and len(failures) > 1:
                    schema_errors.append("no_major_failure cannot coexist with another failure")
        else:
            parse_error = f"batch_status_{status_code}: {raw.get('error')}"
        if annotation is not None and not schema_errors:
            valid += 1
        messages = build_messages(packet, args.protocol, schema)
        choices = body.get("choices") or []
        parsed_rows.append(
            {
                "sample_id": sid,
                "question_key": packet["question_key"],
                "protocol_subset": "all_train",
                "provider": "qwen",
                "protocol": args.protocol,
                "annotation": annotation,
                "parse_error": parse_error,
                "schema_errors": schema_errors,
                "request_hash": hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "response_meta": {
                    "model": body.get("model", args.model),
                    "usage": body.get("usage"),
                    "finish_reason": choices[0].get("finish_reason") if choices else None,
                    "batch_id": batch_id,
                    "batch_status_code": status_code,
                    "max_tokens": 8192,
                    "thinking": "enabled",
                    "normalization_actions": actions,
                },
                "reference_status": "teacher_model_annotation_not_human_review",
            }
        )
    write_jsonl(args.normalized_output, parsed_rows)
    return {
        "expected_rows": len(packets),
        "batch_result_rows": len(raw_rows),
        "normalized_rows": len(parsed_rows),
        "valid_rows": valid,
        "invalid_rows": len(parsed_rows) - valid,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    packets = read_jsonl(args.packets)
    if args.max_rows > 0:
        packets = packets[: args.max_rows]
    if args.max_rows <= 0 and (
        len(packets) != 2654 or len({str(row["sample_id"]) for row in packets}) != 2654
    ):
        raise ValueError("Exp28D Batch requires exactly 2,654 unique paper-train packets")
    if len({str(row["sample_id"]) for row in packets}) != len(packets):
        raise ValueError("Batch packets must have unique sample_id values")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not args.batch_input.exists():
        prepare_input(args, packets, schema)
    if not args.run_api:
        return {
            "status": "BATCH_INPUT_READY",
            "rows": len(packets),
            "batch_input": str(args.batch_input),
            "paper_dev_or_test_read": False,
        }

    client = client_for(args)
    state = load_state(args.state)
    if not state.get("input_file_id"):
        with args.batch_input.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        state["input_file_id"] = uploaded.id
        save_state(args.state, state)
        print(f"[exp28-batch] uploaded input_file_id={uploaded.id}", flush=True)
    if not state.get("batch_id"):
        batch = client.batches.create(
            input_file_id=state["input_file_id"],
            endpoint=args.endpoint,
            completion_window="24h",
        )
        state["batch_id"] = batch.id
        state["submitted_model"] = args.model
        state["submitted_protocol"] = args.protocol
        save_state(args.state, state)
        print(f"[exp28-batch] created batch_id={batch.id}", flush=True)

    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(state["batch_id"])
        state.update(
            {
                "status": batch.status,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "request_counts": (
                    batch.request_counts.model_dump()
                    if getattr(batch, "request_counts", None) is not None
                    else None
                ),
            }
        )
        save_state(args.state, state)
        print(
            f"[exp28-batch] status={batch.status} counts={state.get('request_counts')}",
            flush=True,
        )
        if batch.status in terminal:
            break
        if not args.wait:
            return {"status": batch.status, "batch_id": state["batch_id"]}
        time.sleep(args.poll_seconds)
    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status={batch.status}: {getattr(batch, 'errors', None)}")
    if batch.output_file_id and not args.batch_result.exists():
        download_file(client, batch.output_file_id, args.batch_result)
    if batch.error_file_id and not args.batch_errors.exists():
        download_file(client, batch.error_file_id, args.batch_errors)

    summary = parse_results(args, packets, schema, state["batch_id"])
    decision = {
        "status": (
            "READY_FOR_REALTIME_REPAIR"
            if summary["invalid_rows"]
            else "READY_FOR_SECONDARY_ROUTE"
        ),
        "provider": "qwen",
        "model": args.model,
        "thinking": "enabled",
        "protocol": args.protocol,
        "batch_id": state["batch_id"],
        **summary,
        "paper_dev_or_test_read": False,
        "reference_status": "teacher_model_annotation_not_human_review",
    }
    write_json(args.decision, decision)
    return decision


def parse_args() -> argparse.Namespace:
    private = OUT_DIR / "private" / "batch" / "qwen3_7_max_thinking_all_train"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, default=PACKETS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch-input", type=Path, default=private / "input.jsonl")
    parser.add_argument("--batch-result", type=Path, default=private / "result.jsonl")
    parser.add_argument("--batch-errors", type=Path, default=private / "errors.jsonl")
    parser.add_argument("--state", type=Path, default=private / "state.json")
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=OUT_DIR / "private" / "qwen" / PROTOCOL / "all_train.jsonl",
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=OUT_DIR / "decision" / "exp28d_qwen_batch_annotation_decision.json",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
