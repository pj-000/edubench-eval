"""Resumable API generator for Exp48B metric-specific edit plans."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import MODULE, char_ngrams, jaccard, normalize_text, read_jsonl, tokens, write_jsonl
from .exp48b_common import OUT, PRIVATE, TRAIN, validate_plan
from .run_exp48a_generator_api import PROVIDERS, request_json

NOVELTY_CONTEXTS = (
    "a multilingual emergency-planning workshop at a mountain observatory",
    "a coastal heritage laboratory restoring storm-damaged archives",
    "a fictional orbital greenhouse managing a sensor anomaly",
    "a rural maker classroom coordinating a community water audit",
    "an underwater archaeology course interpreting incomplete records",
    "a public library designing an accessibility-focused science exhibit",
    "a polar field station preparing a mixed-discipline student briefing",
    "a city youth council evaluating a fictional transport proposal",
    "a cross-cultural museum seminar analyzing contested evidence",
    "a remote health-education team adapting materials after a network outage",
    "a robotics ethics class reviewing an autonomous rescue simulation",
    "a conservation studio planning a wildlife corridor with uncertain data",
)


def generate_one(packet: dict, *, provider: str, model: str, base_url: str, api_key: str, prompt: str, schema: dict, timeout: int, retries: int, max_tokens: int, diagnostic_dir: Path, train_novelty: list[tuple[str, set[str], set[str]]]) -> tuple[dict, dict]:
    public_packet = {key: value for key, value in packet.items() if not key.startswith("private_")}
    context = NOVELTY_CONTEXTS[int(packet["blueprint_id"].split("_")[-1]) - 1]
    user = (
        "Create one metric-specific edit plan. Return one JSON object only.\n\n"
        f"MANDATORY NEW CONTEXT: {context}. Do not retain source entities or task wording.\n\n"
        f"SOURCE BLUEPRINT:\n{json.dumps(public_packet, ensure_ascii=False)}\n\n"
        f"REQUIRED SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
    last_error = ""
    last_candidate = "The previous response could not be parsed as JSON."
    for attempt in range(retries + 1):
        try:
            plan, usage = request_json(url=base_url, api_key=api_key, model=model, messages=messages, timeout=timeout, max_tokens=max_tokens)
            last_candidate = json.dumps(plan, ensure_ascii=False)
            plan["source_blueprint_id"] = packet["blueprint_id"]
            plan["metric"] = packet["metric"]
            plan["language"] = packet["language"]
            plan["rubric_levels"] = packet["rubric_levels"]
            plan["generator_provenance"] = {"provider": provider, "model_family": model, "session_id": f"exp48b_{provider}_{time.strftime('%Y%m%d')}"}
            expected = {"D2": 2, "D3": 3, "H4": 4}
            for key, level in expected.items():
                contract = plan.setdefault("metric_contract", {})
                item = contract.setdefault(key, {})
                if item.get("rubric_quote") != packet["rubric_levels"][str(level)]:
                    item["rubric_quote"] = packet["rubric_levels"][str(level)]
                item["rubric_level"] = level
            errors = validate_plan(plan)
            question = str(plan.get("synthetic_question", ""))
            normalized = normalize_text(question)
            question_chars, question_tokens = char_ngrams(question), tokens(question)
            exact = any(normalized == row[0] for row in train_novelty)
            max_char = max((jaccard(question_chars, row[1]) for row in train_novelty), default=0.0)
            max_token = max((jaccard(question_tokens, row[2]) for row in train_novelty), default=0.0)
            if exact or max_char >= 0.80 or max_token >= 0.80:
                errors.append(f"question_novelty_failed:exact={int(exact)}:char5={max_char:.3f}:token={max_token:.3f}")
            if errors:
                raise ValueError("; ".join(errors))
            return plan, {key: int(value) for key, value in usage.items() if isinstance(value, (int, float))}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= retries:
                break
            messages += [
                {"role": "assistant", "content": last_candidate},
                {"role": "user", "content": f"Correct this plan in place and return the complete JSON object. Preserve the metric-specific rubric meaning. Validation errors: {last_error}"},
            ]
            time.sleep(2**attempt)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    if last_candidate.startswith("{"):
        (diagnostic_dir / f"{packet['blueprint_id']}_last_invalid.json").write_text(last_candidate + "\n", encoding="utf-8")
    raise RuntimeError(f"{packet['blueprint_id']} failed after {retries + 1} attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="deepseek")
    parser.add_argument("--model")
    parser.add_argument("--packets", type=Path, default=PRIVATE / "source_packets/exp48b_metric_rubric_blueprints_12.jsonl")
    parser.add_argument("--output", type=Path, default=PRIVATE / "generated_plans/exp48b_metric_specific_edit_plans.jsonl")
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=6144)
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
    by_source = {str(row["source_blueprint_id"]): row for row in existing}
    pending = [row for row in packets if str(row["blueprint_id"]) not in by_source]
    prompt = (MODULE / "prompts/exp48b_metric_specific_generator_prompt.md").read_text(encoding="utf-8")
    schema = json.loads((MODULE / "schemas/exp48b_metric_specific_edit_plan_schema.json").read_text(encoding="utf-8"))
    train_questions = {normalize_text(row["question"]): str(row["question"]) for row in read_jsonl(args.train)}
    train_novelty = [(key, char_ngrams(text), tokens(text)) for key, text in train_questions.items()]
    started = time.time()
    failures, usage_totals = [], {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(generate_one, packet, provider=args.provider, model=model, base_url=base_url, api_key=api_key, prompt=prompt, schema=schema, timeout=args.timeout, retries=args.retries, max_tokens=args.max_tokens, diagnostic_dir=args.output.parent / "invalid_candidates", train_novelty=train_novelty): packet for packet in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            packet = futures[future]
            try:
                plan, usage = future.result()
                by_source[str(packet["blueprint_id"])] = plan
                for key, value in usage.items():
                    usage_totals[key] = usage_totals.get(key, 0) + value
                write_jsonl(args.output, [by_source[str(row["blueprint_id"])] for row in packets if str(row["blueprint_id"]) in by_source])
                elapsed = time.time() - started
                eta = elapsed / completed * (len(pending) - completed)
                print(f"[exp48b-generator] completed={completed}/{len(pending)} saved={len(by_source)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
            except Exception as exc:
                failures.append({"blueprint_id": packet["blueprint_id"], "error": str(exc)})
                print(f"[exp48b-generator] failed={packet['blueprint_id']} error={type(exc).__name__}", flush=True)
    result = {"status": "COMPLETED" if not failures else "PARTIAL_FAILURE", "requested": len(packets), "existing_before": len(existing), "generated_this_run": len(pending) - len(failures), "total_saved": len(by_source), "failures": failures, "usage": usage_totals, "elapsed_seconds": time.time() - started, "provider": args.provider, "model": model}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
