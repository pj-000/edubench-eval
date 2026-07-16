"""Resumable API generator for private Exp48A synthetic families."""

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

from .common import MODULE, OUT, PRIVATE, TRAIN, char_ngrams, jaccard, normalize_text, read_jsonl, tokens, validate_family, write_jsonl

PROVIDERS = {
    "qwen": {
        "key_env": "QWEN_API_KEY", "base_url_env": "QWEN_BASE_URL",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen3.7-max",
    },
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY", "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url": "https://api.deepseek.com", "model": "deepseek-v4-pro",
    },
}

NOVELTY_CONTEXTS = (
    "an Antarctic algae habitat with intermittent communications",
    "a lunar archive restoration project using fragile historical media",
    "an underwater robotics festival for mixed-age student teams",
    "a fictional Mars greenhouse facing an unexpected resource constraint",
    "a community museum digitizing oral histories after a flood",
    "a remote island microgrid planning workshop with competing priorities",
    "a multilingual public-health simulation aboard a research vessel",
    "a wildlife corridor design exercise using fictional sensor evidence",
)


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
        parsed = json.loads(value[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Generator response must be one JSON object")
    return parsed


def request_json(*, url: str, api_key: str, model: str, messages: list[dict], timeout: int, max_tokens: int) -> tuple[dict, dict]:
    body = json.dumps({
        "model": model, "messages": messages, "temperature": 0.35,
        "max_tokens": max_tokens, "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST",
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


def generate_one(packet: dict, *, provider: str, model: str, base_url: str, api_key: str, prompt: str, schema: dict, timeout: int, retries: int, max_tokens: int, diagnostic_dir: Path, train_novelty: list[tuple[str, set[str], set[str]]]) -> tuple[dict, dict]:
    safe_packet = {key: value for key, value in packet.items() if not key.startswith("private_")}
    blueprint_number = int(re.search(r"(\d+)$", str(packet["blueprint_id"])).group(1))
    novelty_context = NOVELTY_CONTEXTS[blueprint_number % len(NOVELTY_CONTEXTS)]
    user = (
        "Generate exactly one family for this source blueprint. Return one JSON object only.\n\n"
        "HARD LENGTH CONTROL: if language=zh, each answer must contain 180-220 normalized characters; "
        "if language=en, each answer must contain 100-130 words. Keep the three surface styles comparable.\n\n"
        f"MANDATORY NOVELTY CONTEXT: build the new question around {novelty_context}. Do not retain source entities or the original requested task.\n\n"
        f"SOURCE BLUEPRINT:\n{json.dumps(safe_packet, ensure_ascii=False)}\n\n"
        f"REQUIRED JSON SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
    last_error = ""
    last_candidate = "The previous response could not be parsed as JSON."
    total_usage: dict[str, int] = {}
    for attempt in range(retries + 1):
        try:
            family, usage = request_json(url=base_url, api_key=api_key, model=model, messages=messages, timeout=timeout, max_tokens=max_tokens)
            last_candidate = json.dumps(family, ensure_ascii=False)
            family["source_blueprint_id"] = packet["blueprint_id"]
            family["metric"] = packet["metric"]
            family["language"] = packet["language"]
            family.setdefault("subject", packet["subject"])
            family.setdefault("scenario", packet["scenario"])
            family.setdefault("education_level", packet["education_level"])
            family["generator_provenance"] = {
                "provider": provider, "model_family": model,
                "session_id": f"exp48a_{provider}_{time.strftime('%Y%m%d')}",
            }
            errors = validate_family(family)
            question = str(family.get("synthetic_question", ""))
            normalized = normalize_text(question)
            question_chars = char_ngrams(question)
            question_tokens = tokens(question)
            exact = any(normalized == candidate[0] for candidate in train_novelty)
            max_char = max((jaccard(question_chars, candidate[1]) for candidate in train_novelty), default=0.0)
            max_token = max((jaccard(question_tokens, candidate[2]) for candidate in train_novelty), default=0.0)
            if exact or max_char >= 0.80 or max_token >= 0.80:
                errors.append(f"question_novelty_failed:exact={int(exact)}:char5={max_char:.3f}:token={max_token:.3f}")
            if errors:
                raise ValueError("; ".join(errors))
            total_usage = {key: int(value) for key, value in usage.items() if isinstance(value, (int, float))}
            return family, total_usage
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= retries:
                break
            if "question_novelty_failed" in last_error:
                correction = (
                    "The synthetic question is too similar to a train question. Replace the synthetic question completely "
                    "with a new domain situation, entities, requested task, and wording while preserving only metric, "
                    "language, and education level. Update all criteria and answers to fit the new question. Do not "
                    "paraphrase or reuse distinctive source phrases. Then fix every other validation error and return "
                    f"the complete JSON family. Validation errors: {last_error}"
                )
            else:
                correction = f"Correct this JSON in place and return the complete corrected family. Do not restart with a different design. Validation errors: {last_error}"
            messages += [{"role": "assistant", "content": last_candidate}, {"role": "user", "content": correction}]
            time.sleep(2 ** attempt)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    if last_candidate.startswith("{"):
        (diagnostic_dir / f"{packet['blueprint_id']}_last_invalid.json").write_text(last_candidate + "\n", encoding="utf-8")
    raise RuntimeError(f"{packet['blueprint_id']} failed after {retries + 1} attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--packets", type=Path, default=PRIVATE / "source_packets/exp48a_generator_blueprints_60.jsonl")
    parser.add_argument("--output", type=Path, default=PRIVATE / "generated_families/exp48a_generated_families.jsonl")
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    config = PROVIDERS[args.provider]
    api_key = os.environ.get(config["key_env"], "")
    if not api_key:
        raise SystemExit(f"Missing {config['key_env']}; provide it only through the environment")
    model = args.model or config["model"]
    base_url = os.environ.get(config["base_url_env"], config["base_url"])
    packets = read_jsonl(args.packets)
    if args.max_rows is not None:
        packets = packets[:args.max_rows]
    existing = read_jsonl(args.output) if args.output.exists() else []
    by_blueprint = {str(row["source_blueprint_id"]): row for row in existing}
    pending = [row for row in packets if str(row["blueprint_id"]) not in by_blueprint]
    prompt = (MODULE / "prompts/exp48a_generator_session_prompt.md").read_text(encoding="utf-8")
    schema = json.loads((MODULE / "schemas/exp48a_synthetic_family_schema.json").read_text(encoding="utf-8"))
    train_questions = {normalize_text(row["question"]): str(row["question"]) for row in read_jsonl(args.train)}
    train_novelty = [(normalized, char_ngrams(text), tokens(text)) for normalized, text in train_questions.items()]
    started = time.time()
    usage_totals: dict[str, int] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(generate_one, packet, provider=args.provider, model=model, base_url=base_url, api_key=api_key, prompt=prompt, schema=schema, timeout=args.timeout, retries=args.retries, max_tokens=args.max_tokens, diagnostic_dir=args.output.parent / "invalid_candidates", train_novelty=train_novelty): packet
            for packet in pending
        }
        for completed, future in enumerate(as_completed(futures), 1):
            packet = futures[future]
            try:
                family, usage = future.result()
                by_blueprint[str(packet["blueprint_id"])] = family
                for key, value in usage.items():
                    usage_totals[key] = usage_totals.get(key, 0) + value
                ordered = [by_blueprint[key] for key in sorted(by_blueprint)]
                write_jsonl(args.output, ordered)
                elapsed = time.time() - started
                rate = elapsed / completed
                eta = rate * (len(pending) - completed)
                print(f"[exp48a-generator] completed={completed}/{len(pending)} total_saved={len(ordered)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
            except Exception as exc:
                failures.append({"blueprint_id": packet["blueprint_id"], "error": str(exc)})
                print(f"[exp48a-generator] failed={packet['blueprint_id']} error={type(exc).__name__}", flush=True)
    result = {
        "status": "COMPLETED" if not failures else "PARTIAL_FAILURE", "provider": args.provider, "model": model,
        "requested": len(packets), "existing_before": len(existing), "generated_this_run": len(pending) - len(failures),
        "total_saved": len(by_blueprint), "failures": failures, "usage": usage_totals,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
