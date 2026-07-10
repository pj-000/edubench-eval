"""Run or dry-run Exp27D teacher-audit API calls.

Default behavior is a dry-run message check. Real API calls require --run-api
and keys supplied only through environment variables.
"""

from __future__ import annotations

import argparse
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
except ModuleNotFoundError:  # Local macOS runner may not have the optional dependency.
    from thesis_exp.exp17_low_score_evidence import json_schema_compat as jsonschema

from thesis_exp.exp17_low_score_evidence.prepare_exp27d_teacher_audit_v4_packets import (  # noqa: E402
    PROMPT_VERSION,
    SCHEMA_VERSION,
)
from thesis_exp.src.edujudge.utils.io import write_json, write_text  # noqa: E402


DEFAULT_OUT_DIR = Path("thesis_exp/exp17_low_score_evidence/outputs/exp27d_teacher_audit_v4_seed42")
DEFAULT_PACKETS = DEFAULT_OUT_DIR / "packets" / "exp27d_v4_repilot_blind_packets.jsonl"
DEFAULT_AUDIT_REFERENCE = DEFAULT_OUT_DIR / "packets" / "exp27d_v4_repilot_audit_reference_private.jsonl"

PROVIDERS = {
    "qwen": {
        "env_key": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_env": "QWEN_TEACHER_MODEL",
        "model": "qwen3.7-max",
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url": "https://api.deepseek.com",
        "model_env": "DEEPSEEK_TEACHER_MODEL",
        "model": "deepseek-v4-pro",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json_file(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_text(text: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None, f"json_decode_error: {exc}"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc2:
            return None, f"json_decode_error_after_object_extract: {exc2}"
    if not isinstance(parsed, dict):
        return None, "parsed_json_is_not_object"
    return parsed, ""


def teacher_input_text(packet: dict[str, Any]) -> str:
    teacher_input = packet["teacher_input"]
    return "\n\n".join(
        [
            f"Sample ID: {packet['sample_id']}",
            f"Question:\n{teacher_input.get('question', '')}",
            f"Answer:\n{teacher_input.get('answer', '')}",
            f"Evaluation metric:\n{teacher_input.get('metric', '')}",
            f"Rubric:\n{teacher_input.get('rubric', '')}",
            f"Metadata:\n{teacher_input.get('metadata', '')}",
        ]
    )


def load_protocol(out_dir: Path) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    blind_prompt = read_text(out_dir / "prompts" / "exp27d_blind_teacher_prompt_v4.md")
    audit_prompt = read_text(out_dir / "prompts" / "exp27d_label_audit_prompt_v4.md")
    blind_schema = read_json(out_dir / "schema" / "exp27d_teacher_blind_schema_v4.json")
    audit_schema = read_json(out_dir / "schema" / "exp27d_teacher_audit_schema_v4.json")
    return blind_prompt, audit_prompt, blind_schema, audit_schema


def schema_for_stage(out_dir: Path, stage: str) -> dict[str, Any]:
    _, _, blind_schema, audit_schema = load_protocol(out_dir)
    return blind_schema if stage == "blind" else audit_schema


def build_blind_messages(packet: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    blind_prompt, _, blind_schema, _ = load_protocol(out_dir)
    user = "\n\n".join(
        [
            teacher_input_text(packet),
            "Return this exact top-level shape:",
            '{"sample_id":"...","blind":{...}}',
            f"Blind output JSON schema:\n{json.dumps(blind_schema, ensure_ascii=False, sort_keys=True)}",
        ]
    )
    return [{"role": "system", "content": blind_prompt}, {"role": "user", "content": user}]


def build_audit_messages(
    packet: dict[str, Any],
    blind_payload: dict[str, Any],
    audit_ref: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, str]]:
    _, audit_prompt, _, audit_schema = load_protocol(out_dir)
    blind_annotation_id = f"{packet['sample_id']}:blind"
    blind_annotation_hash = sha1_text(json.dumps(blind_payload, ensure_ascii=False, sort_keys=True))
    user = "\n\n".join(
        [
            teacher_input_text(packet),
            f"Original human score: {audit_ref['original_score']}",
            f"Blind annotation id: {blind_annotation_id}",
            f"Blind annotation hash: {blind_annotation_hash}",
            f"Previous blind teacher output:\n{json.dumps(blind_payload, ensure_ascii=False, sort_keys=True)}",
            "Return this exact top-level shape:",
            '{"sample_id":"...","blind_annotation_id":"...","blind_annotation_hash":"...","audit":{...}}',
            f"Audit output JSON schema:\n{json.dumps(audit_schema, ensure_ascii=False, sort_keys=True)}",
        ]
    )
    return [{"role": "system", "content": audit_prompt}, {"role": "user", "content": user}]


def build_schema_repair_messages(
    messages: list[dict[str, str]],
    parsed: dict[str, Any],
    schema_errors: list[str],
    stage_schema: dict[str, Any],
) -> list[dict[str, str]]:
    repair_instruction = "\n\n".join(
        [
            "Your previous JSON object violated the required schema.",
            "Fix only the JSON structure and enum values. Do not change the scoring judgment unless required by the schema.",
            "Return exactly one corrected JSON object. Do not wrap it in Markdown. Do not add extra keys.",
            f"Schema errors:\n{json.dumps(schema_errors[:20], ensure_ascii=False, indent=2)}",
            f"Previous JSON:\n{json.dumps(parsed, ensure_ascii=False, sort_keys=True)}",
            f"Required JSON schema:\n{json.dumps(stage_schema, ensure_ascii=False, sort_keys=True)}",
        ]
    )
    return messages + [{"role": "user", "content": repair_instruction}]


def provider_config(provider: str, model: str | None) -> dict[str, str]:
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["env_key"], "")
    if not api_key:
        raise SystemExit(f"Missing API key environment variable: {cfg['env_key']}")
    base_url = os.environ.get(cfg["base_url_env"], cfg["base_url"]).rstrip("/")
    resolved_model = model or os.environ.get(cfg["model_env"], cfg["model"])
    return {"api_key": api_key, "base_url": base_url, "model": resolved_model}


def dry_provider_config(provider: str, model: str | None) -> dict[str, str]:
    cfg = PROVIDERS[provider]
    return {
        "api_key": "<dry-run>",
        "base_url": os.environ.get(cfg["base_url_env"], cfg["base_url"]).rstrip("/"),
        "model": model or os.environ.get(cfg["model_env"], cfg["model"]),
    }


def chat_completion(
    provider: str,
    cfg: dict[str, str],
    messages: list[dict[str, str]],
    timeout: int,
    thinking: str,
    temperature: float,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if thinking != "omit":
        enabled = thinking == "on"
        if provider == "qwen":
            body["enable_thinking"] = enabled
        elif provider == "deepseek":
            body["thinking"] = {"type": "enabled" if enabled else "disabled"}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1200]}") from exc


def output_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return ""


def successful_output_row(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("sample_id"))
        and isinstance(row.get("parsed"), dict)
        and not row.get("parse_error")
        and not row.get("schema_errors")
    )


def retain_successful_outputs(path: Path) -> set[str]:
    """Drop stale failed rows before resume so only failed samples are retried."""
    if not path.exists():
        return set()
    successful = [row for row in read_jsonl(path) if successful_output_row(row)]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in successful),
        encoding="utf-8",
    )
    return {str(row["sample_id"]) for row in successful}


def load_blind_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Blind output file is required for audit stage: {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else None
        if parsed is None:
            continue
        sid = str(row.get("sample_id") or parsed.get("sample_id") or "")
        if sid:
            out[sid] = parsed
    return out


def load_audit_ref_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Audit reference file is required for audit stage: {path}")
    return {str(row["sample_id"]): row for row in read_jsonl(path) if row.get("sample_id")}


def teacher_score_from_parsed(parsed: dict[str, Any] | None) -> int | None:
    if not isinstance(parsed, dict):
        return None
    blind = parsed.get("blind")
    audit = parsed.get("audit")
    source = blind if isinstance(blind, dict) else audit if isinstance(audit, dict) else {}
    score = source.get("teacher_score") if isinstance(source, dict) else None
    return score if isinstance(score, int) else None


def major_failures_from_parsed(parsed: dict[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(parsed, dict):
        return tuple()
    blind = parsed.get("blind")
    if not isinstance(blind, dict):
        return tuple()
    failures = blind.get("major_failures")
    if not isinstance(failures, list):
        return tuple()
    return tuple(str(item) for item in failures)


def score_cap_from_parsed(parsed: dict[str, Any] | None) -> int | None:
    if not isinstance(parsed, dict):
        return None
    blind = parsed.get("blind")
    if not isinstance(blind, dict):
        return None
    cap = blind.get("score_cap")
    return cap if isinstance(cap, int) else None


def failure_visibility_from_parsed(parsed: dict[str, Any] | None) -> str:
    if not isinstance(parsed, dict):
        return ""
    blind = parsed.get("blind")
    if not isinstance(blind, dict):
        return ""
    value = blind.get("failure_visibility")
    return str(value) if value is not None else ""


def run(args: argparse.Namespace) -> dict[str, Any]:
    do_api = args.run_api and not args.dry_run
    cfg = provider_config(args.provider, args.model) if do_api else dry_provider_config(args.provider, args.model)
    packets = read_jsonl(args.packets)
    if args.max_rows > 0:
        packets = packets[: args.max_rows]
    output_suffix = args.output_suffix or ""
    if output_suffix and not re.fullmatch(r"[_A-Za-z0-9.-]+", output_suffix):
        raise SystemExit(f"Unsafe output suffix: {output_suffix}")
    output_path = args.out_dir / "annotations" / "parsed" / args.provider / f"exp27d_{args.stage}_outputs{output_suffix}.jsonl"
    if args.overwrite and output_path.exists():
        output_path.unlink()
    done = retain_successful_outputs(output_path) if args.resume else set()
    blind_by_id: dict[str, dict[str, Any]] = {}
    audit_ref_by_id: dict[str, dict[str, Any]] = {}
    if args.stage == "audit":
        blind_path = args.blind_output or (
            args.out_dir / "annotations" / "parsed" / args.provider / "exp27d_blind_outputs.jsonl"
        )
        if not blind_path.exists():
            raise SystemExit(f"Blind parsed output is required for audit stage: {blind_path}")
        blind_by_id = load_blind_by_id(blind_path)
        audit_ref_by_id = load_audit_ref_by_id(args.audit_reference)
    stage_schema = schema_for_stage(args.out_dir, args.stage)
    schema_validator = jsonschema.Draft202012Validator(stage_schema)

    start = time.time()
    attempted = 0
    parsed_ok = 0
    schema_ok = 0
    failed = 0
    consecutive_failed = 0
    repair_attempt_rows = 0
    repair_success_rows = 0
    repair_changed_teacher_score_count = 0
    repair_changed_major_failures_count = 0
    repair_changed_score_cap_count = 0
    repair_changed_failure_visibility_count = 0
    for idx, packet in enumerate(packets, start=1):
        sid = str(packet["sample_id"])
        if sid in done and not args.overwrite:
            continue
        if args.stage == "blind":
            messages = build_blind_messages(packet, args.out_dir)
        else:
            blind_payload = blind_by_id.get(sid)
            audit_ref = audit_ref_by_id.get(sid)
            if blind_payload is None or audit_ref is None:
                failed += 1
                append_jsonl(
                    output_path,
                    {
                        "sample_id": sid,
                        "provider": args.provider,
                        "stage": args.stage,
                        "parse_error": "missing_blind_payload_or_audit_reference",
                        "parsed": None,
                        "schema_errors": ["missing_blind_payload_or_audit_reference"],
                    },
                )
                continue
            messages = build_audit_messages(packet, blind_payload, audit_ref, args.out_dir)

        if not do_api:
            print(json.dumps({"provider": args.provider, "stage": args.stage, "messages": messages}, ensure_ascii=False, indent=2))
            break

        attempted += 1
        repair_attempts_used = 0

        def request_and_parse(request_messages: list[dict[str, str]], repair_attempt: int) -> tuple[
            dict[str, Any] | None,
            str,
            list[str],
            dict[str, Any],
            str,
            str,
        ]:
            request_hash = sha1_text(json.dumps(request_messages, ensure_ascii=False, sort_keys=True))
            raw_path = (
                args.out_dir
                / "annotations"
                / "raw_api"
                / args.provider
                / args.stage
                / f"{args.provider}_{args.stage}_{sid}_{request_hash[:12]}_repair{repair_attempt}.json"
            )
            parse_error = ""
            parsed: dict[str, Any] | None = None
            response_meta: dict[str, Any] = {}
            for attempt in range(1, args.retries + 2):
                try:
                    response = chat_completion(
                        args.provider, cfg, request_messages, args.timeout, args.thinking, args.temperature
                    )
                    raw_text = output_content(response)
                    parsed, parse_error = parse_json_text(raw_text)
                    response_meta = {
                        "model": cfg["model"],
                        "provider": args.provider,
                        "prompt_version": PROMPT_VERSION,
                        "schema_version": SCHEMA_VERSION,
                        "created": response.get("created"),
                        "usage": response.get("usage"),
                        "repair_attempt": repair_attempt,
                    }
                    write_json_file(
                        raw_path,
                        {
                            "sample_id": sid,
                            "provider": args.provider,
                            "stage": args.stage,
                            "request_hash": request_hash,
                            "repair_attempt": repair_attempt,
                            "response": response,
                        },
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    parse_error = f"api_error_attempt_{attempt}: {exc}"
                    if attempt <= args.retries:
                        time.sleep(args.retry_sleep)
            schema_errors: list[str] = []
            if parsed is not None:
                schema_errors = [err.message for err in schema_validator.iter_errors(parsed)]
            return parsed, parse_error, schema_errors, response_meta, str(raw_path) if raw_path.exists() else "", request_hash

        parsed, parse_error, schema_errors, response_meta, raw_api_path, request_hash = request_and_parse(messages, 0)
        initial_teacher_score = teacher_score_from_parsed(parsed)
        initial_major_failures = major_failures_from_parsed(parsed)
        initial_score_cap = score_cap_from_parsed(parsed)
        initial_failure_visibility = failure_visibility_from_parsed(parsed)
        while parsed is not None and schema_errors and repair_attempts_used < args.schema_repair_retries:
            repair_attempts_used += 1
            repair_messages = build_schema_repair_messages(messages, parsed, schema_errors, stage_schema)
            parsed, parse_error, schema_errors, response_meta, raw_api_path, request_hash = request_and_parse(
                repair_messages, repair_attempts_used
            )
        repair_changed_teacher_score = (
            repair_attempts_used > 0 and teacher_score_from_parsed(parsed) != initial_teacher_score
        )
        repair_changed_major_failures = (
            repair_attempts_used > 0 and major_failures_from_parsed(parsed) != initial_major_failures
        )
        repair_changed_score_cap = repair_attempts_used > 0 and score_cap_from_parsed(parsed) != initial_score_cap
        repair_changed_failure_visibility = (
            repair_attempts_used > 0 and failure_visibility_from_parsed(parsed) != initial_failure_visibility
        )
        if repair_attempts_used > 0:
            repair_attempt_rows += 1
            if parsed is not None and not schema_errors:
                repair_success_rows += 1
            if repair_changed_teacher_score:
                repair_changed_teacher_score_count += 1
            if repair_changed_major_failures:
                repair_changed_major_failures_count += 1
            if repair_changed_score_cap:
                repair_changed_score_cap_count += 1
            if repair_changed_failure_visibility:
                repair_changed_failure_visibility_count += 1
        if parsed is None:
            failed += 1
            consecutive_failed += 1
        else:
            parsed_ok += 1
            if not schema_errors:
                schema_ok += 1
                consecutive_failed = 0
            else:
                failed += 1
                consecutive_failed += 1
        append_jsonl(
            output_path,
            {
                "sample_id": sid,
                "batch_id": packet.get("batch_id"),
                "provider": args.provider,
                "stage": args.stage,
                "thinking": args.thinking,
                "parsed": parsed,
                "parse_error": parse_error,
                "schema_errors": schema_errors,
                "schema_repair_attempts": repair_attempts_used,
                "repair_changed_teacher_score": repair_changed_teacher_score,
                "repair_changed_major_failures": repair_changed_major_failures,
                "repair_changed_score_cap": repair_changed_score_cap,
                "repair_changed_failure_visibility": repair_changed_failure_visibility,
                "raw_api_path": raw_api_path,
                "request_hash": request_hash,
                "response_meta": response_meta,
                "source_meta": packet.get("source_meta"),
            },
        )
        elapsed = time.time() - start
        rate = attempted / elapsed if elapsed > 0 else 0.0
        remaining = max(0, len(packets) - idx)
        eta = remaining / rate if rate > 0 else 0.0
        print(
            f"[exp27d] {args.provider}/{args.stage} {idx}/{len(packets)} "
            f"parsed={parsed_ok} schema_ok={schema_ok} failed={failed} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )
        if consecutive_failed > args.max_consecutive_failures:
            raise SystemExit(
                f"Stopping after {consecutive_failed} consecutive failed rows for {args.provider}/{args.stage}"
            )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary = {
        "provider": args.provider,
        "stage": args.stage,
        "model": cfg["model"],
        "packets": len(packets),
        "attempted": attempted,
        "parsed_ok": parsed_ok,
        "schema_ok": schema_ok,
        "failed": failed,
        "output_path": str(output_path),
        "dry_run": not do_api,
        "schema_repair_retries": args.schema_repair_retries,
        "schema_repair_attempt_rows": repair_attempt_rows,
        "repair_success_rows": repair_success_rows,
        "repair_changed_teacher_score_count": repair_changed_teacher_score_count,
        "repair_changed_major_failures_count": repair_changed_major_failures_count,
        "repair_changed_score_cap_count": repair_changed_score_cap_count,
        "repair_changed_failure_visibility_count": repair_changed_failure_visibility_count,
    }
    summary_suffix = output_suffix or ""
    write_json(args.out_dir / "decision" / f"exp27d_{args.provider}_{args.stage}{summary_suffix}_api_summary.json", summary)
    write_text(
        args.out_dir / "reports" / f"exp27d_{args.provider}_{args.stage}{summary_suffix}_api_summary.md",
        "\n".join(
            [
                f"# Exp27D {args.provider} {args.stage} API Summary",
                "",
                f"- model: `{cfg['model']}`",
                f"- packets: {len(packets)}",
                f"- attempted: {attempted}",
                f"- parsed_ok: {parsed_ok}",
                f"- schema_ok: {schema_ok}",
                f"- failed: {failed}",
                f"- dry_run: {not do_api}",
                f"- schema_repair_retries: {args.schema_repair_retries}",
                f"- schema_repair_attempt_rows: {repair_attempt_rows}",
                f"- repair_success_rows: {repair_success_rows}",
                f"- repair_changed_teacher_score_count: {repair_changed_teacher_score_count}",
                f"- repair_changed_major_failures_count: {repair_changed_major_failures_count}",
                f"- repair_changed_score_cap_count: {repair_changed_score_cap_count}",
                f"- repair_changed_failure_visibility_count: {repair_changed_failure_visibility_count}",
                f"- output: `{output_path}`",
            ]
        ),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or dry-run Exp27D teacher-audit API calls.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--stage", choices=["blind", "audit"], default="blind")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--blind-output", type=Path, default=None)
    parser.add_argument("--audit-reference", type=Path, default=DEFAULT_AUDIT_REFERENCE)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--schema-repair-retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--thinking", choices=["omit", "off", "on"], default="omit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
