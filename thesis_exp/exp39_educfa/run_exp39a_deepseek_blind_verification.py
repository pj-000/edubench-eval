"""Dry-run or execute target-blind DeepSeek verification for Exp39A."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
from thesis_exp.exp39_educfa.run_exp39a_qwen_counterfactual_generation import call_api, parse_content  # noqa: E402

PROMPT_PATH = PROMPT_DIR / "exp39a_deepseek_blind_counterfactual_verifier.md"
SCHEMA_PATH = SCHEMA_DIR / "exp39a_counterfactual_verification_schema.json"


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


def user_text(packet: dict, candidate: dict, schema: dict) -> str:
    # Assigned target and generator range are deliberately absent.
    return "\n\n".join(
        [
            f"Sample ID: {packet['sample_id']}",
            f"Source sample ID: {packet['source_sample_id']}",
            "<CONTEXT_ONLY_ORIGINAL_TASK>\n" + str(packet["question"]) + "\n</CONTEXT_ONLY_ORIGINAL_TASK>",
            "<ORIGINAL_HIGH_SCORE_OUTPUT>\n" + str(packet["original_answer"]) + "\n</ORIGINAL_HIGH_SCORE_OUTPUT>",
            "<COUNTERFACTUAL_OUTPUT_TO_VERIFY>\n" + str(candidate["counterfactual_answer"]) + "\n</COUNTERFACTUAL_OUTPUT_TO_VERIFY>",
            f"Evaluation dimension: {packet['metric']}",
            "Rubric: " + json.dumps(packet["rubric"], ensure_ascii=False),
            "Non-label metadata: " + json.dumps(packet["metadata"], ensure_ascii=False, sort_keys=True),
            f"Declared operator: {candidate['operator']}",
            "JSON schema: " + json.dumps(schema, ensure_ascii=False, sort_keys=True),
        ]
    )


def semantic_errors(value: dict, packet: dict) -> list[str]:
    errors = []
    if str(value.get("sample_id")) != str(packet["sample_id"]):
        errors.append("sample_id_mismatch")
    if str(value.get("source_sample_id")) != str(packet["source_sample_id"]):
        errors.append("source_sample_id_mismatch")
    for field in ("original_score_range", "counterfactual_score_range"):
        score_range = value.get(field) or []
        if len(score_range) == 2 and score_range[0] > score_range[1]:
            errors.append(f"{field}_order_invalid")
    score_range = value.get("counterfactual_score_range") or []
    center = value.get("most_plausible_counterfactual_score")
    if len(score_range) == 2 and isinstance(center, int) and not score_range[0] <= center <= score_range[1]:
        errors.append("plausible_score_outside_range")
    return errors


def verify_one(
    packet: dict,
    candidate: dict,
    *,
    model: str,
    prompt: str,
    schema: dict,
    max_tokens: int,
    base_url: str,
    api_key: str,
    timeout: int,
    retries: int,
) -> tuple[str, dict, dict]:
    sid = str(packet["sample_id"])
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_text(packet, candidate, schema)},
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
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
                raise RuntimeError(f"DeepSeek verification failed for {sid}: {type(exc).__name__}: {exc}") from exc
            if response_content:
                body["messages"] = messages + [
                    {"role": "assistant", "content": response_content},
                    {
                        "role": "user",
                        "content": (
                            "The previous JSON failed automatic validation: "
                            f"{type(exc).__name__}: {exc}. Correct only that failure and return one "
                            "schema-valid JSON object without inferring or adding any hidden target."
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
    generated_path = args.out_dir / "private/generated_candidates/exp39a_qwen_generated_candidates.jsonl"
    dry_run = args.dry_run or os.environ.get("RUN_API") != "1"
    generated = read_jsonl(generated_path) if generated_path.exists() else []
    by_generated = {sample_id(row): row for row in generated}
    if dry_run and not generated:
        by_generated = {
            str(packet["sample_id"]): {
                "sample_id": packet["sample_id"],
                "source_sample_id": packet["source_sample_id"],
                "operator": packet["assigned_operator"],
                "counterfactual_answer": str(packet["original_answer"]) + " [dry-run edit]",
            }
            for packet in packets
        }
    if not dry_run and set(by_generated) != {str(row["sample_id"]) for row in packets}:
        raise ValueError("Blind verification requires complete Qwen generation for all source packets")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    model = args.model or lock["deepseek_model"]
    if model != lock["deepseek_model"]:
        raise RuntimeError(f"DeepSeek model differs from frozen protocol: {model}")
    first = packets[0]
    first_text = user_text(first, by_generated[str(first["sample_id"])], schema)
    forbidden = [str(first["assigned_target_score"]), "assigned_target", "target_score_range"]
    if "Target score:" in first_text or "Assigned target" in first_text or "target_score_range" in first_text:
        raise RuntimeError(f"Verifier input leaked hidden target fields: {forbidden}")
    if dry_run:
        print(json.dumps({
            "status": "DRY_RUN", "rows": len(packets), "model": model, "api_called": False,
            "sample_id": first["sample_id"], "target_blind": True,
            "request_hash": stable_hash({"prompt": prompt, "schema": schema, "user": first_text}),
        }, indent=2, sort_keys=True))
        return
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    raw_path = args.out_dir / "raw_api/exp39a_deepseek_blind_verification.jsonl"
    parsed_path = args.out_dir / "private/verified_counterfactuals/exp39a_deepseek_verifications.jsonl"
    existing = read_jsonl(parsed_path) if parsed_path.exists() else []
    existing_by_id = {sample_id(row): row for row in existing}
    if len(existing) != len(existing_by_id):
        write_jsonl(parsed_path, [existing_by_id[key] for key in sorted(existing_by_id)])
    completed = set(existing_by_id)
    pending = [packet for packet in packets if str(packet["sample_id"]) not in completed]
    workers = max(1, args.workers)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                verify_one,
                packet,
                by_generated[str(packet["sample_id"])],
                model=model,
                prompt=prompt,
                schema=schema,
                max_tokens=int(lock["deepseek_max_tokens"]),
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
                    f"[exp39a-deepseek] {len(completed)}/{len(packets)} workers={workers} "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
        except Exception:
            for future in futures:
                future.cancel()
            raise
    summary = {
        "status": "COMPLETED", "provider": "deepseek", "model": model, "rows": len(completed),
        "expected_rows": len(packets), "schema_success_rate": 1.0, "target_blind": True,
        "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp39a_deepseek_verification_api_summary.json", summary)
    write_csv(args.out_dir / "tables/exp39a_verification_completion.csv", [summary])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
