"""Resumable blind criterion verifier for private Exp48A packets."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .common import MODULE, PRIVATE, read_jsonl, write_jsonl

PROVIDERS = {
    "qwen": {
        "key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-max",
    },
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
    },
}


def parse_json(text: str) -> dict[str, Any]:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Verifier response must be one JSON object")
    return parsed


def request_json(
    *, url: str, api_key: str, model: str, messages: list[dict], timeout: int, max_tokens: int
) -> tuple[dict, dict]:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    choice = payload["choices"][0]
    content = choice["message"].get("content") or ""
    if not content.strip():
        reasoning = choice["message"].get("reasoning_content") or ""
        raise ValueError(
            f"empty_content finish_reason={choice.get('finish_reason')} "
            f"reasoning_chars={len(reasoning)} usage={payload.get('usage', {})}"
        )
    return parse_json(content), payload.get("usage", {})


def validate_output(output: dict, packet: dict, *, verifier_id: str, model: str, session_id: str) -> list[str]:
    errors: list[str] = []
    if output.get("family_id") != packet["family_id"]:
        errors.append("family_id mismatch")
    if output.get("packet_id") != packet["packet_id"]:
        errors.append("packet_id mismatch")
    provenance = output.get("verifier_provenance")
    if not isinstance(provenance, dict):
        errors.append("missing verifier_provenance")
    else:
        expected = {"verifier_id": verifier_id, "model_family": model, "session_id": session_id}
        for key, value in expected.items():
            if provenance.get(key) != value:
                errors.append(f"verifier_provenance.{key} mismatch")
    answers = output.get("answers")
    if not isinstance(answers, list):
        return errors + ["answers must be a list"]
    expected_answers = {str(row["anonymous_answer_id"]) for row in packet["answers"]}
    actual_answers = {str(row.get("anonymous_answer_id")) for row in answers if isinstance(row, dict)}
    if actual_answers != expected_answers:
        errors.append("anonymous answer IDs mismatch")
    expected_criteria = {str(row["id"]) for row in packet["criteria"]}
    allowed_statuses = {"satisfied", "partial", "violated", "unclear"}
    allowed_uncertainty = {"low", "medium", "high"}
    for answer in answers:
        if not isinstance(answer, dict):
            errors.append("answer output must be an object")
            continue
        answer_id = str(answer.get("anonymous_answer_id"))
        if answer.get("uncertainty") not in allowed_uncertainty:
            errors.append(f"{answer_id}: invalid uncertainty")
        criteria = answer.get("criteria")
        if not isinstance(criteria, list):
            errors.append(f"{answer_id}: criteria must be a list")
            continue
        actual_criteria = {str(row.get("criterion_id")) for row in criteria if isinstance(row, dict)}
        if actual_criteria != expected_criteria:
            errors.append(f"{answer_id}: criterion IDs mismatch")
        for criterion in criteria:
            if not isinstance(criterion, dict):
                errors.append(f"{answer_id}: criterion output must be an object")
                continue
            status = criterion.get("status")
            if status not in allowed_statuses:
                errors.append(f"{answer_id}/{criterion.get('criterion_id')}: invalid status")
            evidence = str(criterion.get("evidence_span", "")).strip()
            missing = str(criterion.get("missing_reason", "")).strip()
            if status in {"satisfied", "partial"} and not evidence:
                errors.append(f"{answer_id}/{criterion.get('criterion_id')}: evidence_span required")
            if status in {"violated", "unclear"} and not missing:
                errors.append(f"{answer_id}/{criterion.get('criterion_id')}: missing_reason required")
    forbidden = {"score", "intended_score", "gold_label", "target_score"}
    serialized = json.dumps(output, ensure_ascii=False).lower()
    for field in forbidden:
        if f'"{field}"' in serialized:
            errors.append(f"forbidden field present: {field}")
    return sorted(set(errors))


def verify_one(
    packet: dict,
    *,
    model: str,
    base_url: str,
    api_key: str,
    prompt: str,
    schema: dict,
    verifier_id: str,
    session_id: str,
    timeout: int,
    retries: int,
    max_tokens: int,
    diagnostic_dir: Path,
) -> tuple[dict, dict]:
    blind_packet = {key: value for key, value in packet.items() if key != "score_program"}
    user = (
        "Verify every criterion for every anonymous answer in this packet. Return exactly one JSON object. "
        "Do not output, infer, or discuss a score. For satisfied/partial use a short exact quote from the answer; "
        "for violated/unclear explain what evidence is missing.\n\n"
        f"VERIFIER ID: {verifier_id}\nMODEL FAMILY: {model}\nSESSION ID: {session_id}\n\n"
        f"BLIND PACKET:\n{json.dumps(blind_packet, ensure_ascii=False)}\n\n"
        f"REQUIRED JSON SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
    last_error = ""
    last_candidate = "The previous response could not be parsed as JSON."
    for attempt in range(retries + 1):
        try:
            output, usage = request_json(
                url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            last_candidate = json.dumps(output, ensure_ascii=False)
            output["family_id"] = packet["family_id"]
            output["packet_id"] = packet["packet_id"]
            output["verifier_provenance"] = {
                "verifier_id": verifier_id,
                "model_family": model,
                "session_id": session_id,
            }
            errors = validate_output(
                output, packet, verifier_id=verifier_id, model=model, session_id=session_id
            )
            if errors:
                raise ValueError("; ".join(errors))
            return output, {key: int(value) for key, value in usage.items() if isinstance(value, (int, float))}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= retries:
                break
            messages += [
                {"role": "assistant", "content": last_candidate},
                {
                    "role": "user",
                    "content": (
                        "Correct the JSON in place. Return the complete object with all three answers and every "
                        f"criterion. Do not add any score field. Validation errors: {last_error}"
                    ),
                },
            ]
            time.sleep(2**attempt)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    if last_candidate.startswith("{"):
        (diagnostic_dir / f"{packet['family_id']}_last_invalid.json").write_text(
            last_candidate + "\n", encoding="utf-8"
        )
    raise RuntimeError(f"{packet['family_id']} failed after {retries + 1} attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-id", choices=["a", "b"], required=True)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--packets", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=6144)
    args = parser.parse_args()

    config = PROVIDERS[args.provider]
    api_key = os.environ.get(config["key_env"], "")
    if not api_key:
        raise SystemExit(f"Missing {config['key_env']}; provide it only through the environment")
    model = args.model or config["model"]
    base_url = os.environ.get(config["base_url_env"], config["base_url"])
    verifier_id = args.verifier_id
    packets_path = args.packets or PRIVATE / f"verifier_packets/exp48a_verifier_{verifier_id}_packets.jsonl"
    output_path = args.output or PRIVATE / f"verifier_{verifier_id}/exp48a_verifier_{verifier_id}_outputs.jsonl"
    session_id = args.session_id or f"exp48a_{args.provider}_verifier_{verifier_id}_{time.strftime('%Y%m%d')}"

    packets = read_jsonl(packets_path)
    if args.max_rows is not None:
        packets = packets[: args.max_rows]
    existing = read_jsonl(output_path) if output_path.exists() else []
    by_family = {str(row["family_id"]): row for row in existing}
    pending = [row for row in packets if str(row["family_id"]) not in by_family]
    prompt = (MODULE / "prompts/exp48a_blind_verifier_session_prompt.md").read_text(encoding="utf-8")
    schema = json.loads(
        (MODULE / "schemas/exp48a_criterion_verification_schema.json").read_text(encoding="utf-8")
    )
    started = time.time()
    usage_totals: dict[str, int] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                verify_one,
                packet,
                model=model,
                base_url=base_url,
                api_key=api_key,
                prompt=prompt,
                schema=schema,
                verifier_id=verifier_id,
                session_id=session_id,
                timeout=args.timeout,
                retries=args.retries,
                max_tokens=args.max_tokens,
                diagnostic_dir=output_path.parent / "invalid_candidates",
            ): packet
            for packet in pending
        }
        for completed, future in enumerate(as_completed(futures), 1):
            packet = futures[future]
            try:
                output, usage = future.result()
                by_family[str(packet["family_id"])] = output
                for key, value in usage.items():
                    usage_totals[key] = usage_totals.get(key, 0) + value
                ordered = [by_family[str(row["family_id"])] for row in packets if str(row["family_id"]) in by_family]
                write_jsonl(output_path, ordered)
                elapsed = time.time() - started
                eta = elapsed / completed * (len(pending) - completed)
                print(
                    f"[exp48a-verifier-{verifier_id}] completed={completed}/{len(pending)} "
                    f"total_saved={len(ordered)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
            except Exception as exc:
                failures.append({"family_id": packet["family_id"], "error": str(exc)})
                print(
                    f"[exp48a-verifier-{verifier_id}] failed={packet['family_id']} error={type(exc).__name__}",
                    flush=True,
                )
    result = {
        "status": "COMPLETED" if not failures else "PARTIAL_FAILURE",
        "verifier_id": verifier_id,
        "provider": args.provider,
        "model": model,
        "session_id": session_id,
        "requested": len(packets),
        "existing_before": len(existing),
        "verified_this_run": len(pending) - len(failures),
        "total_saved": len(by_family),
        "failures": failures,
        "usage": usage_totals,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
