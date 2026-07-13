"""Dry-run or execute the frozen Exp40A DeepSeek order-swapped verification."""

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

from thesis_exp.exp40_edupair_cf.common import (  # noqa: E402
    PROMPT_PATH,
    ROOT,
    SCHEMA_PATH,
    append_jsonl,
    read_jsonl,
    stable_hash,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--model")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("API_WORKERS", "4")))
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=5)
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
        raise ValueError("Pairwise judgment is not a JSON object")
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
            f"Pair ID: {packet['pair_id']}",
            f"Sample ID: {packet['sample_id']}",
            f"Order: {packet['order']}",
            "<ORIGINAL_TASK_CONTEXT>\n" + str(packet["question"]) + "\n</ORIGINAL_TASK_CONTEXT>",
            f"Evaluation dimension: {packet['metric']}",
            "Full rubric: " + json.dumps(packet["rubric"], ensure_ascii=False),
            "Targeted rubric clause: " + str(packet["targeted_rubric_clause"]),
            "Declared counterfactual operator: " + str(packet["declared_counterfactual_operator"]),
            "<ANSWER_A>\n" + str(packet["answer_a"]) + "\n</ANSWER_A>",
            "<ANSWER_B>\n" + str(packet["answer_b"]) + "\n</ANSWER_B>",
            "JSON schema: " + json.dumps(schema, ensure_ascii=False, sort_keys=True),
        ]
    )


def semantic_errors(value: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    errors = []
    for field in ("pair_id", "sample_id", "order"):
        if str(value.get(field)) != str(packet[field]):
            errors.append(f"{field}_mismatch")
    return errors


def judge_one(
    packet: dict[str, Any], *, model: str, prompt: str, schema: dict[str, Any], base_url: str,
    api_key: str, max_tokens: int, timeout: int, retries: int,
) -> tuple[tuple[str, str], dict[str, Any], dict[str, Any], list[str]]:
    identity = (str(packet["pair_id"]), str(packet["order"]))
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_text(packet, schema)}]
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    for attempt in range(1, retries + 1):
        response_content = ""
        try:
            response = call_api(base_url, api_key, body, timeout)
            response_content = response["choices"][0]["message"]["content"]
            if not str(response_content or "").strip():
                raise ValueError("empty_final_content")
            value = parse_content(response_content)
            normalized_identity_fields = []
            for field in ("pair_id", "sample_id", "order"):
                if str(value.get(field)) != str(packet[field]):
                    value[field] = packet[field]
                    normalized_identity_fields.append(field)
            errors = [error.message for error in jsonschema.Draft202012Validator(schema).iter_errors(value)]
            errors.extend(semantic_errors(value, packet))
            if errors:
                raise ValueError("; ".join(errors))
            return identity, response, value, normalized_identity_fields
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"DeepSeek pairwise verification failed for {identity}: {type(exc).__name__}: {exc}"
                ) from exc
            if response_content:
                body["messages"] = messages + [
                    {"role": "assistant", "content": response_content},
                    {
                        "role": "user",
                        "content": (
                            "The prior JSON failed automatic validation: "
                            f"{type(exc).__name__}: {exc}. Correct only the schema or routing-identity failure. "
                            "Do not infer an absolute score and return only one JSON object."
                        ),
                    },
                ]
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    lock_path = args.out_dir / "configs/exp40a_pairwise_protocol_lock.json"
    packet_path = args.out_dir / "private/pair_packets/exp40a_pairwise_verification_packets.jsonl"
    if not lock_path.exists() or not packet_path.exists():
        raise FileNotFoundError("Run Exp40A pair resolution and packet preparation first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    packets = read_jsonl(packet_path)
    if args.max_pairs:
        selected_ids = {row["pair_id"] for row in packets[: args.max_pairs * 2]}
        packets = [row for row in packets if row["pair_id"] in selected_ids]
    model = args.model or lock["model"]
    if model != lock["model"]:
        raise RuntimeError(f"DeepSeek model differs from frozen protocol: {model}")
    prompt_path = args.out_dir / "prompts/exp40a_deepseek_pairwise_judge.md"
    prompt = prompt_path.read_text(encoding="utf-8").rstrip("\n") + "\n\n"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    dry_run = args.dry_run or os.environ.get("RUN_API") != "1"
    first_text = user_text(packets[0], schema)
    forbidden_literals = ("assigned_target_score", "target_score_range", "human_1", "human_2", "human_3")
    if any(value in first_text for value in forbidden_literals):
        raise RuntimeError("Target-blind verification request contains a forbidden field")
    if dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN", "pairs": len({row['pair_id'] for row in packets}),
                    "judgments": len(packets), "model": model, "api_called": False, "target_blind": True,
                    "request_hash": stable_hash({"prompt": prompt, "schema": schema, "user": first_text}),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    parsed_path = args.out_dir / "private/pairwise_judgments/exp40a_deepseek_pairwise_judgments.jsonl"
    raw_path = args.out_dir / "raw_api/exp40a_deepseek_pairwise_verification.jsonl"
    normalization_path = args.out_dir / "private/pairwise_judgments/exp40a_routing_identity_normalizations.jsonl"
    existing = read_jsonl(parsed_path) if parsed_path.exists() else []
    all_existing = {(str(row["pair_id"]), str(row["order"])): row for row in existing}
    packet_keys = {(str(row["pair_id"]), str(row["order"])) for row in packets}
    existing_for_run = {key: row for key, row in all_existing.items() if key in packet_keys}
    existing_normalizations = read_jsonl(normalization_path) if normalization_path.exists() else []
    normalized_keys = {(str(row["pair_id"]), str(row["order"])) for row in existing_normalizations}
    pending = [row for row in packets if (str(row["pair_id"]), str(row["order"])) not in existing_for_run]
    started = time.time()
    failures = []
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                judge_one, packet, model=model, prompt=prompt, schema=schema, base_url=base_url,
                api_key=api_key, max_tokens=int(lock["max_tokens"]), timeout=args.timeout, retries=args.retries,
            ): (str(packet["pair_id"]), str(packet["order"]))
            for packet in pending
        }
        completed_session = 0
        for future in as_completed(futures):
            try:
                identity, response, value, normalized_fields = future.result()
            except Exception as exc:
                identity = futures[future]
                failures.append((identity, f"{type(exc).__name__}: {exc}"))
                print(f"[exp40a-pair] deferred_failure pair={identity[0]} order={identity[1]}", flush=True)
                continue
            append_jsonl(raw_path, {"pair_id": identity[0], "order": identity[1], "response": response})
            all_existing[identity] = value
            if normalized_fields and identity not in normalized_keys:
                append_jsonl(
                    normalization_path,
                    {
                        "pair_id": identity[0], "order": identity[1],
                        "normalized_fields": normalized_fields, "normalization_reason": "routing_identity_only",
                    },
                )
                normalized_keys.add(identity)
            completed_session += 1
            elapsed = time.time() - started
            total_completed = len(existing_for_run) + completed_session
            eta = elapsed / max(completed_session, 1) * max(len(pending) - completed_session, 0)
            print(
                f"[exp40a-pair] {total_completed}/{len(packets)} workers={workers} "
                f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )
    write_jsonl(parsed_path, [all_existing[key] for key in sorted(all_existing)])
    if failures:
        preview = "; ".join(f"{identity}: {message}" for identity, message in failures[:5])
        raise RuntimeError(f"Exp40A has {len(failures)} deferred API failures; rerun to resume. {preview}")
    completed = sum(key in all_existing for key in packet_keys)
    summary = {
        "status": "COMPLETED" if completed == len(packets) else "INCOMPLETE",
        "provider": "deepseek", "model": model, "pairs": len({row["pair_id"] for row in packets}),
        "expected_judgments": len(packets), "completed_judgments": completed,
        "schema_success_rate": completed / max(len(packets), 1), "target_blind": True,
        "routing_identity_normalization_count": len(normalized_keys & packet_keys),
        "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp40a_deepseek_pairwise_api_summary.json", summary)
    write_csv(args.out_dir / "tables/exp40a_pairwise_completion.csv", [{"stage": "api_verification", **summary}])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
