"""Resumable API verifier for Exp48B blind metric-contract packets."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import MODULE, read_jsonl, write_jsonl
from .exp48b_common import PRIVATE, VERIFIER_STATES
from .run_exp48a_generator_api import PROVIDERS, request_json


def validate_output(output: dict, packet: dict, verifier_id: str, model: str, session_id: str) -> list[str]:
    errors = []
    if output.get("family_id") != packet["family_id"] or output.get("packet_id") != packet["packet_id"]:
        errors.append("packet identity mismatch")
    provenance = output.get("verifier_provenance", {})
    if provenance != {"verifier_id": verifier_id, "model_family": model, "session_id": session_id}:
        errors.append("verifier provenance mismatch")
    answers = output.get("answers")
    if not isinstance(answers, list):
        return errors + ["answers must be a list"]
    packet_answers = {row["anonymous_answer_id"]: row["text"] for row in packet["answers"]}
    if {row.get("anonymous_answer_id") for row in answers if isinstance(row, dict)} != set(packet_answers):
        errors.append("anonymous answer IDs mismatch")
    for answer in answers:
        if not isinstance(answer, dict):
            errors.append("answer output is not an object")
            continue
        answer_id = answer.get("anonymous_answer_id")
        text = packet_answers.get(answer_id, "")
        if answer.get("uncertainty") not in {"low", "medium", "high"}:
            errors.append(f"{answer_id}: invalid uncertainty")
        contracts = answer.get("contracts")
        if not isinstance(contracts, list) or {row.get("contract_id") for row in contracts if isinstance(row, dict)} != {"D2", "D3", "H4"}:
            errors.append(f"{answer_id}: contract IDs mismatch")
            continue
        for row in contracts:
            contract_id, status = row.get("contract_id"), row.get("status")
            evidence, missing = str(row.get("evidence_span", "")), str(row.get("missing_reason", ""))
            if status not in VERIFIER_STATES:
                errors.append(f"{answer_id}/{contract_id}: invalid status")
            if status in {"entailed", "contradicted"} and (not evidence or evidence not in text):
                errors.append(f"{answer_id}/{contract_id}: evidence must be exact answer substring")
            if status in {"absent", "unclear"} and (evidence or not missing.strip()):
                errors.append(f"{answer_id}/{contract_id}: absent/unclear evidence fields invalid")
    serialized = json.dumps(output, ensure_ascii=False).lower()
    if any(f'"{field}"' in serialized for field in ("score", "intended_score", "target_score", "ranking")):
        errors.append("forbidden direct-score field")
    return sorted(set(errors))


def verify_one(packet: dict, *, base_url: str, api_key: str, model: str, verifier_id: str, session_id: str, prompt: str, schema: dict, timeout: int, retries: int, max_tokens: int, diagnostic_dir: Path) -> tuple[dict, dict]:
    user = (
        "Blindly verify all three metric-specific assertions for every anonymous answer. Return one JSON object only.\n\n"
        f"VERIFIER ID: {verifier_id}\nMODEL FAMILY: {model}\nSESSION ID: {session_id}\n\n"
        f"BLIND PACKET:\n{json.dumps(packet, ensure_ascii=False)}\n\n"
        f"REQUIRED SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
    last_error, last_candidate = "", "The previous response could not be parsed as JSON."
    for attempt in range(retries + 1):
        try:
            output, usage = request_json(url=base_url, api_key=api_key, model=model, messages=messages, timeout=timeout, max_tokens=max_tokens)
            last_candidate = json.dumps(output, ensure_ascii=False)
            output["family_id"], output["packet_id"] = packet["family_id"], packet["packet_id"]
            output["verifier_provenance"] = {"verifier_id": verifier_id, "model_family": model, "session_id": session_id}
            errors = validate_output(output, packet, verifier_id, model, session_id)
            if errors:
                raise ValueError("; ".join(errors))
            return output, {key: int(value) for key, value in usage.items() if isinstance(value, (int, float))}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= retries:
                break
            messages += [{"role": "assistant", "content": last_candidate}, {"role": "user", "content": f"Correct the complete JSON object. Do not output scores. Every entailed/contradicted evidence span must be copied exactly from its answer. Errors: {last_error}"}]
            time.sleep(2**attempt)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    if last_candidate.startswith("{"):
        (diagnostic_dir / f"{packet['family_id']}_last_invalid.json").write_text(last_candidate + "\n", encoding="utf-8")
    raise RuntimeError(f"{packet['family_id']} failed: {last_error}")


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
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=6144)
    args = parser.parse_args()
    config = PROVIDERS[args.provider]
    api_key = os.environ.get(config["key_env"], "")
    if not api_key:
        raise SystemExit(f"Missing {config['key_env']}; provide it only through the environment")
    model = args.model or config["model"]
    base_url = os.environ.get(config["base_url_env"], config["base_url"])
    packets_path = args.packets or PRIVATE / f"verifier_packets/exp48b_verifier_{args.verifier_id}_packets.jsonl"
    output_path = args.output or PRIVATE / f"verifier_{args.verifier_id}/exp48b_verifier_{args.verifier_id}_outputs.jsonl"
    session_id = args.session_id or f"exp48b_{args.provider}_{args.verifier_id}_{time.strftime('%Y%m%d')}"
    packets = read_jsonl(packets_path)
    if args.max_rows is not None:
        packets = packets[:args.max_rows]
    existing = read_jsonl(output_path) if output_path.exists() else []
    by_family = {row["family_id"]: row for row in existing}
    pending = [row for row in packets if row["family_id"] not in by_family]
    prompt = (MODULE / "prompts/exp48b_blind_verifier_prompt.md").read_text(encoding="utf-8")
    schema = json.loads((MODULE / "schemas/exp48b_blind_verification_schema.json").read_text(encoding="utf-8"))
    started, failures, usage_totals = time.time(), [], {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(verify_one, packet, base_url=base_url, api_key=api_key, model=model, verifier_id=args.verifier_id, session_id=session_id, prompt=prompt, schema=schema, timeout=args.timeout, retries=args.retries, max_tokens=args.max_tokens, diagnostic_dir=output_path.parent / "invalid_candidates"): packet for packet in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            packet = futures[future]
            try:
                output, usage = future.result()
                by_family[packet["family_id"]] = output
                for key, value in usage.items():
                    usage_totals[key] = usage_totals.get(key, 0) + value
                write_jsonl(output_path, [by_family[row["family_id"]] for row in packets if row["family_id"] in by_family])
                elapsed = time.time() - started
                eta = elapsed / completed * (len(pending) - completed)
                print(f"[exp48b-verifier-{args.verifier_id}] completed={completed}/{len(pending)} saved={len(by_family)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
            except Exception as exc:
                failures.append({"family_id": packet["family_id"], "error": str(exc)})
                print(f"[exp48b-verifier-{args.verifier_id}] failed={packet['family_id']}", flush=True)
    result = {"status": "COMPLETED" if not failures else "PARTIAL_FAILURE", "verifier_id": args.verifier_id, "provider": args.provider, "model": model, "session_id": session_id, "requested": len(packets), "existing_before": len(existing), "verified_this_run": len(pending) - len(failures), "total_saved": len(by_family), "failures": failures, "usage": usage_totals, "elapsed_seconds": time.time() - started}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
