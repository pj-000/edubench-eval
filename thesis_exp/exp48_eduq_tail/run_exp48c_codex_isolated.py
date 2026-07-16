"""Run one ephemeral Codex context per Exp48C packet."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .common import MODULE, read_jsonl, write_jsonl
from .exp48c_common import output_path, packet_path, validate_pointwise_output


def review_one(packet: dict[str, Any], *, model: str, reasoning: str, prompt: str, schema_path: Path, timeout: int, scratch_root: Path) -> dict[str, Any]:
    packet_id = packet["packet_id"]
    session_id = f"exp48c_codex_isolated_{packet_id}"
    scratch = scratch_root / packet_id
    scratch.mkdir(parents=True, exist_ok=True)
    output_file = scratch / "output.json"
    instruction = (
        f"{prompt}\n\n"
        "This is a fresh pointwise context. Do not use tools, inspect files, browse, or seek any other answer. "
        "The only evidence available is the one packet below.\n\n"
        f"POINTWISE PACKET:\n{json.dumps(packet, ensure_ascii=False)}\n\n"
        "Return one raw JSON object only. Preserve packet_id and anonymous_answer_id. "
        f"Set verifier_provenance to exactly: {json.dumps({'verifier_id': 'codex', 'model_family': 'gpt-5.5', 'model_version': model, 'session_id': session_id})}"
    )
    command = [
        "codex", "exec", "--ephemeral", "--ignore-rules", "--skip-git-repo-check",
        "--sandbox", "read-only", "--color", "never", "-C", str(scratch),
        "-m", model, "-c", f'model_reasoning_effort="{reasoning}"',
        "--output-schema", str(schema_path), "--output-last-message", str(output_file), "-",
    ]
    completed = subprocess.run(
        command, input=instruction, text=True, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, timeout=timeout, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"codex exit={completed.returncode}: {completed.stderr[-1000:]}")
    if not output_file.exists():
        raise RuntimeError("Codex did not write its final output")
    output = json.loads(output_file.read_text(encoding="utf-8"))
    errors = validate_pointwise_output(output, packet)
    if errors:
        raise ValueError("; ".join(errors))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, default=packet_path("codex"))
    parser.add_argument("--output", type=Path, default=output_path("codex"))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning", choices=["low", "medium", "high", "xhigh"], default="high")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if shutil.which("codex") is None:
        raise SystemExit("Codex CLI is not available")
    packets = read_jsonl(args.packets)
    if args.max_rows is not None:
        packets = packets[: args.max_rows]
    existing = read_jsonl(args.output) if args.output.exists() else []
    by_id = {row["packet_id"]: row for row in existing}
    pending = [row for row in packets if row["packet_id"] not in by_id]
    started, failures = time.time(), []
    scratch_root = Path(tempfile.mkdtemp(prefix="exp48c_codex_"))
    schema = MODULE / "schemas/exp48c_rubric_only_score_schema.json"
    prompt = (MODULE / "prompts/exp48c_rubric_only_pointwise_verifier_prompt.md").read_text(encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(review_one, packet, model=args.model, reasoning=args.reasoning, prompt=prompt, schema_path=schema, timeout=args.timeout, scratch_root=scratch_root): packet
                for packet in pending
            }
            for completed_count, future in enumerate(as_completed(futures), 1):
                packet = futures[future]
                try:
                    output = future.result()
                    by_id[packet["packet_id"]] = output
                    write_jsonl(args.output, [by_id[row["packet_id"]] for row in packets if row["packet_id"] in by_id])
                except Exception as exc:
                    failures.append({"packet_id": packet["packet_id"], "error": str(exc)})
                elapsed = time.time() - started
                eta = elapsed / max(1, completed_count) * (len(pending) - completed_count)
                print(f"[exp48c-codex] completed={completed_count}/{len(pending)} saved={len(by_id)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
    summary = {
        "status": "COMPLETED" if not failures and len(by_id) == len(packets) else "PARTIAL_FAILURE",
        "requested": len(packets), "saved": len(by_id), "failures": failures,
        "model": args.model, "reasoning": args.reasoning,
        "ephemeral_contexts": True, "one_context_per_packet": True,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["status"] != "COMPLETED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
