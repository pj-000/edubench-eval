"""Run Exp28 teacher-protocol annotations with OpenAI-compatible APIs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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


DEFAULT_INVENTORY_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28b_paper_train_annotation_inventory_seed42"
)
DEFAULT_PACKETS = DEFAULT_INVENTORY_DIR / "private" / "exp28b_blind_teacher_packets_2654.jsonl"
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42"
)
SCHEMA_PATH = Path(
    "thesis_exp/exp17_low_score_evidence/schemas/exp28_teacher_annotation_schema.json"
)
PROMPT_PATHS = {
    "p0_holistic_zero_shot": Path(
        "thesis_exp/exp17_low_score_evidence/prompts/exp28_p0_holistic_zero_shot.md"
    ),
    "p1_rubric_first": Path(
        "thesis_exp/exp17_low_score_evidence/prompts/exp28_p1_rubric_first.md"
    ),
    "p2_rubric_verify_then_score": Path(
        "thesis_exp/exp17_low_score_evidence/prompts/exp28_p2_rubric_verify_then_score.md"
    ),
}
PROVIDERS = {
    "qwen": {
        "key_env": "QWEN_API_KEY",
        "base_env": "QWEN_BASE_URL",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_env": "QWEN_TEACHER_MODEL",
        "model": "qwen3.7-max",
    },
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "base_env": "DEEPSEEK_BASE_URL",
        "base_url": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_TEACHER_MODEL",
        "model": "deepseek-v4-pro",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = strip_code_fence(text)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None, f"json_decode_error: {exc}"
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as second:
            return None, f"json_decode_error_after_extract: {second}"
    if not isinstance(value, dict):
        return None, "parsed_json_is_not_object"
    return value, ""


def normalize_annotation_structure(
    annotation: dict[str, Any] | None,
    sid: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize transport-only JSON differences without changing semantic fields."""
    if annotation is None:
        return None, []
    normalized = dict(annotation)
    actions: list[str] = []
    required = set(schema.get("required") or [])
    if "content" in normalized and not (required & set(normalized)):
        wrapped = normalized.get("content")
        if isinstance(wrapped, dict):
            normalized = dict(wrapped)
            actions.append("unwrap_content_object")
        elif isinstance(wrapped, str):
            parsed, _ = parse_json(wrapped)
            if parsed is not None:
                normalized = parsed
                actions.append("unwrap_content_json_string")
    if "sample_id" not in normalized:
        normalized["sample_id"] = sid
        actions.append("restore_request_sample_id")
    allowed = set((schema.get("properties") or {}).keys())
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        normalized = {key: value for key, value in normalized.items() if key in allowed}
        actions.extend(f"drop_unexpected_top_level:{key}" for key in unexpected)
    return normalized, actions


def provider_config(provider: str, model: str | None, dry_run: bool) -> dict[str, str]:
    spec = PROVIDERS[provider]
    key = os.environ.get(spec["key_env"], "")
    if not dry_run and not key:
        raise SystemExit(f"Missing {spec['key_env']}; keys must be supplied through environment variables")
    return {
        "api_key": key if not dry_run else "<dry-run>",
        "base_url": os.environ.get(spec["base_env"], spec["base_url"]).rstrip("/"),
        "model": model or os.environ.get(spec["model_env"], spec["model"]),
    }


def input_text(packet: dict[str, Any]) -> str:
    teacher_input = packet["teacher_input"]
    return "\n\n".join(
        [
            f"Sample ID: {packet['sample_id']}",
            f"Question:\n{teacher_input['question']}",
            f"Answer:\n{teacher_input['answer']}",
            f"Evaluation metric:\n{teacher_input['metric']}",
            f"Rubric:\n{json.dumps(teacher_input['rubric'], ensure_ascii=False)}",
            f"Metadata:\n{json.dumps(teacher_input['metadata'], ensure_ascii=False, sort_keys=True)}",
        ]
    )


def build_messages(packet: dict[str, Any], protocol: str, schema: dict[str, Any]) -> list[dict[str, str]]:
    system = PROMPT_PATHS[protocol].read_text(encoding="utf-8")
    user = "\n\n".join(
        [
            input_text(packet),
            "Return exactly one JSON object. Copy the supplied sample_id exactly.",
            f"Required schema:\n{json.dumps(schema, ensure_ascii=False, sort_keys=True)}",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_completion(
    cfg: dict[str, str],
    messages: list[dict[str, str]],
    timeout: int,
    max_tokens: int,
    thinking: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    body = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    if cfg["provider"] == "deepseek":
        body["thinking"] = {"type": thinking}
        if thinking == "enabled":
            body["reasoning_effort"] = reasoning_effort
    elif cfg["provider"] == "qwen":
        body["enable_thinking"] = thinking == "enabled"
    request = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def successful_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    successful = []
    for row in read_jsonl(path):
        if isinstance(row.get("annotation"), dict) and not row.get("schema_errors"):
            successful.append(row)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in successful),
        encoding="utf-8",
    )
    return {str(row["sample_id"]) for row in successful}


def annotate_one(
    packet: dict[str, Any],
    args: argparse.Namespace,
    cfg: dict[str, str],
    schema: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], bool]:
    sid = str(packet["sample_id"])
    messages = build_messages(packet, args.protocol, schema)
    validator = jsonschema.Draft202012Validator(schema)
    annotation = None
    parse_error = ""
    schema_errors: list[str] = []
    response_meta: dict[str, Any] = {}
    normalization_actions: list[str] = []
    for attempt in range(args.retries + 1):
        try:
            response = chat_completion(
                cfg,
                messages,
                args.timeout,
                max_tokens,
                args.thinking,
                args.reasoning_effort,
            )
            annotation, parse_error = parse_json(response_text(response))
            annotation, normalization_actions = normalize_annotation_structure(
                annotation, sid, schema
            )
            schema_errors = []
            if annotation is not None:
                schema_errors = [error.message for error in validator.iter_errors(annotation)]
                if annotation.get("sample_id") != sid:
                    schema_errors.append("sample_id mismatch")
                failures = annotation.get("major_failures") or []
                if "no_major_failure" in failures and len(failures) > 1:
                    schema_errors.append("no_major_failure cannot coexist with another failure")
            choices = response.get("choices") or []
            response_meta = {
                "model": cfg["model"],
                "usage": response.get("usage"),
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "max_tokens": max_tokens,
                "thinking": args.thinking,
                "reasoning_effort": (
                    args.reasoning_effort
                    if args.provider == "deepseek" and args.thinking == "enabled"
                    else "not_applicable"
                ),
                "attempt": attempt + 1,
                "normalization_actions": normalization_actions,
            }
            if annotation is not None and not schema_errors:
                break
            if attempt < args.retries:
                time.sleep(args.retry_sleep)
        except Exception as exc:  # noqa: BLE001
            parse_error = f"api_error_attempt_{attempt + 1}: {exc}"
            if attempt < args.retries:
                time.sleep(args.retry_sleep)

    ok = annotation is not None and not schema_errors
    request_hash = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return (
        {
            "sample_id": sid,
            "question_key": packet["question_key"],
            "protocol_subset": args.subset,
            "provider": args.provider,
            "protocol": args.protocol,
            "annotation": annotation,
            "parse_error": parse_error,
            "schema_errors": schema_errors,
            "request_hash": request_hash,
            "response_meta": response_meta,
            "reference_status": "teacher_model_annotation_not_human_review",
        },
        ok,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    dry_run = not args.run_api
    cfg = provider_config(args.provider, args.model, dry_run)
    cfg["provider"] = args.provider
    max_tokens = args.max_tokens or (8192 if args.provider == "deepseek" else 4096)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    all_packets = read_jsonl(args.packets)
    packets = (
        all_packets
        if args.subset == "all_train"
        else [row for row in all_packets if row.get("protocol_subset") == args.subset]
    )
    if args.max_rows > 0:
        packets = packets[: args.max_rows]
    output = args.out_dir / "private" / args.provider / args.protocol / f"{args.subset}.jsonl"
    done = successful_ids(output) if args.resume else set()
    if args.overwrite and output.exists():
        output.unlink()
        done = set()

    pending = [(index, packet) for index, packet in enumerate(packets, start=1) if str(packet["sample_id"]) not in done]
    attempted = passed = failed = 0
    start = time.time()
    if dry_run and pending:
        messages = build_messages(pending[0][1], args.protocol, schema)
        print(json.dumps({"provider": args.provider, "protocol": args.protocol, "messages": messages}, ensure_ascii=False, indent=2))
    elif pending:
        attempted = len(pending)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(annotate_one, packet, args, cfg, schema, max_tokens): source_index
                for source_index, packet in pending
            }
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                source_index = futures[future]
                row, ok = future.result()
                append_jsonl(output, row)
                if ok:
                    passed += 1
                else:
                    failed += 1
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed else 0.0
                eta = (attempted - completed) / rate if rate else 0.0
                print(
                    f"[exp28] {args.provider}/{args.protocol}/{args.subset} "
                    f"completed={completed}/{attempted} source={source_index}/{len(packets)} "
                    f"passed={passed} failed={failed} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )

    summary = {
        "provider": args.provider,
        "model": cfg["model"],
        "protocol": args.protocol,
        "subset": args.subset,
        "packet_rows": len(packets),
        "attempted": attempted,
        "passed": passed,
        "failed": failed,
        "workers": args.workers,
        "max_tokens": max_tokens,
        "thinking": args.thinking,
        "reasoning_effort": (
            args.reasoning_effort
            if args.provider == "deepseek" and args.thinking == "enabled"
            else "not_applicable"
        ),
        "dry_run": dry_run,
        "output": str(output),
    }
    summary_path = args.out_dir / "decision" / f"exp28_{args.provider}_{args.protocol}_{args.subset}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--protocol", choices=sorted(PROMPT_PATHS), required=True)
    parser.add_argument(
        "--subset",
        choices=[
            "protocol_demo_reference",
            "protocol_development",
            "sealed_qualification",
            "full_annotation_pool",
            "all_train",
            "secondary_route",
        ],
        default="protocol_development",
    )
    parser.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
