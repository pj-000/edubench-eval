"""Run independent, resumable Qwen3.7-Max rubric-only pointwise reviews."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .common import MODULE, read_jsonl, write_jsonl
from .exp48c_common import OUT, output_path, packet_path, validate_pointwise_output
from .run_exp48a_generator_api import parse_json


def request_one(*, base_url: str, api_key: str, model: str, packet: dict[str, Any], prompt: str, schema: dict[str, Any], timeout: int, max_tokens: int, session_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    user = (
        "Score this one answer independently. Return one JSON object only.\n\n"
        f"POINTWISE PACKET:\n{json.dumps(packet, ensure_ascii=False)}\n\n"
        f"OUTPUT SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    content = payload["choices"][0]["message"].get("content") or ""
    output = parse_json(content)
    output["packet_id"] = packet["packet_id"]
    output["anonymous_answer_id"] = packet["anonymous_answer_id"]
    output["verifier_provenance"] = {
        "verifier_id": "qwen", "model_family": "qwen3.7-max",
        "model_version": model, "session_id": session_id,
    }
    errors = validate_pointwise_output(output, packet)
    if errors:
        raise ValueError("; ".join(errors))
    return output, payload.get("usage", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, default=packet_path("qwen"))
    parser.add_argument("--output", type=Path, default=output_path("qwen"))
    parser.add_argument("--model", default="qwen3.7-max")
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    packets = read_jsonl(args.packets)
    if args.max_rows is not None:
        packets = packets[: args.max_rows]
    prompt = (MODULE / "prompts/exp48c_rubric_only_pointwise_verifier_prompt.md").read_text(encoding="utf-8")
    schema = json.loads((MODULE / "schemas/exp48c_rubric_only_score_schema.json").read_text(encoding="utf-8"))
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN", "packets": len(packets), "model": args.model,
            "temperature": 0, "thinking": "disabled", "independent_requests": True,
            "output": str(args.output),
        }, sort_keys=True))
        return
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing QWEN_API_KEY; provide it only through the environment")
    existing = read_jsonl(args.output) if args.output.exists() else []
    by_id = {row["packet_id"]: row for row in existing}
    pending = [row for row in packets if row["packet_id"] not in by_id]
    started, failures, usage = time.time(), [], {}

    def run(packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        last_error = ""
        for attempt in range(args.retries + 1):
            try:
                return request_one(
                    base_url=args.base_url, api_key=api_key, model=args.model,
                    packet=packet, prompt=prompt, schema=schema, timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    session_id=f"exp48c_qwen_{packet['packet_id']}",
                )
            except ValueError as exc:
                # A semantically invalid response is retained as a failure. Retrying it
                # could silently replace the model's original score and violate the lock.
                last_error = f"{type(exc).__name__}: {exc}"
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(last_error)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run, packet): packet for packet in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            packet = futures[future]
            try:
                output, row_usage = future.result()
                by_id[packet["packet_id"]] = output
                for key, value in row_usage.items():
                    if isinstance(value, (int, float)):
                        usage[key] = usage.get(key, 0) + value
                write_jsonl(args.output, [by_id[row["packet_id"]] for row in packets if row["packet_id"] in by_id])
            except Exception as exc:
                failures.append({"packet_id": packet["packet_id"], "error": str(exc)})
            elapsed = time.time() - started
            eta = elapsed / max(1, completed) * (len(pending) - completed)
            print(f"[exp48c-qwen] completed={completed}/{len(pending)} saved={len(by_id)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    summary = {
        "status": "COMPLETED" if not failures and len(by_id) == len(packets) else "PARTIAL_FAILURE",
        "requested": len(packets), "saved": len(by_id), "failures": failures,
        "model": args.model, "temperature": 0, "thinking": "disabled",
        "independent_requests": True, "usage": usage, "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["status"] != "COMPLETED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
