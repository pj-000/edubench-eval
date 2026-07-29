"""Run one evaluator family on the train-only SORC rationale qualification."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    compact_json,
    read_jsonl,
    sha256_file,
)
from thesis_exp.exp54_rar_sft.run_minimal_rationale_audit_api import (
    CONFIG,
    call_api,
    parse_object,
    response_text,
    validate,
)


ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_pairs/rationale_qualification"
)
CANDIDATE_LOCK = ROOT / "candidate_lock.json"
PROMPT_ROOT = REPO_ROOT / "thesis_exp/exp54_rar_sft/prompts"
SCHEMA_ROOT = REPO_ROOT / "thesis_exp/exp54_rar_sft/schemas"
DEFAULT_MODELS = {
    "qwen": "qwen3.7-max",
    "deepseek": "deepseek-v4-pro",
}
EXPECTED_TASKS = 240


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(compact_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _paths(
    *,
    provider: str,
    model: str,
    stage: str,
) -> tuple[Path, Path]:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    stem = f"{provider}_{safe_model}_{stage}"
    return (
        ROOT / "private/evaluator_results" / f"{stem}.jsonl",
        ROOT / "raw_api" / f"{stem}.jsonl",
    )


def _validate_candidate() -> dict[str, Any]:
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if (
        lock.get("status")
        != "QUALIFICATION_PACKAGE_NOT_RUN_TRAINING_NOT_ALLOWED"
        or lock.get("rationale_blind_qualification_completed") is not False
        or lock.get("p3_preference_training_allowed") is not False
        or lock.get("dev_accessed") is not False
        or lock.get("test_accessed") is not False
    ):
        raise PermissionError("rationale qualification candidate differs")
    return lock


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock = _validate_candidate()
    stage = args.stage.replace("-", "_")
    tasks_path = ROOT / "private" / f"{stage}_tasks.jsonl"
    expected_item = lock["private_output_hashes"][f"{stage}_tasks"]
    if (
        sha256_file(tasks_path) != str(expected_item["sha256"])
        or int(expected_item["rows"]) != EXPECTED_TASKS
    ):
        raise ValueError("qualification task lock differs")
    prompt_path = PROMPT_ROOT / f"rationale_audit_{stage}_v2.txt"
    schema_path = SCHEMA_ROOT / f"rationale_audit_{stage}_v2.schema.json"
    expected_sources = lock["source_hashes"]
    if (
        sha256_file(prompt_path)
        != str(expected_sources[f"{stage}_prompt"])
        or sha256_file(schema_path)
        != str(expected_sources[f"{stage}_schema"])
    ):
        raise ValueError("qualification prompt/schema differs")
    tasks = read_jsonl(tasks_path)
    if (
        len(tasks) != EXPECTED_TASKS
        or len({str(row["presentation_id"]) for row in tasks})
        != EXPECTED_TASKS
    ):
        raise ValueError("qualification task inventory differs")
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    parsed_path, raw_path = _paths(
        provider=args.provider,
        model=args.model,
        stage=stage,
    )
    existing_rows = read_jsonl(parsed_path) if parsed_path.exists() else []
    existing = {
        str(row["presentation_id"]): row for row in existing_rows
    }
    if len(existing) != len(existing_rows):
        raise ValueError("existing evaluator results are duplicated")
    task_ids = {str(row["presentation_id"]) for row in tasks}
    if not set(existing).issubset(task_ids):
        raise ValueError("existing evaluator result has an unknown task")
    pending = [
        task
        for task in tasks
        if str(task["presentation_id"]) not in existing
    ]
    if not args.execute:
        return {
            "status": "SORC_RATIONALE_QUALIFICATION_API_DRY_RUN",
            "provider": args.provider,
            "model": args.model,
            "stage": stage,
            "tasks": len(tasks),
            "already_complete": len(existing),
            "pending": len(pending),
            "api_called": False,
            "dev_accessed": False,
            "test_accessed": False,
        }
    if not os.environ.get(CONFIG[args.provider]["key"]):
        raise RuntimeError(f"missing {CONFIG[args.provider]['key']}")

    started = time.time()

    def worker(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                response = call_api(
                    provider=args.provider,
                    model=args.model,
                    prompt=prompt,
                    schema=schema,
                    task=task,
                    timeout=args.timeout,
                )
                parsed = parse_object(response_text(response))
                validate(parsed, schema, str(task["presentation_id"]))
                return response, parsed
            except Exception as exc:
                last_error = exc
                if attempt < args.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"{task['presentation_id']}: {last_error}"
        ) from last_error

    completed = len(existing)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            response, parsed = future.result()
            parsed_with_identity = {
                **parsed,
                "stage": stage,
            }
            wrapper = {
                "schema_version": (
                    "exp54-sorc-rationale-qualification-evaluator-result-v1"
                ),
                "stage": stage,
                "presentation_id": str(task["presentation_id"]),
                "evaluator_family": args.provider,
                "model": args.model,
                "parsed_judgment": parsed_with_identity,
            }
            append_jsonl(
                raw_path,
                {
                    "stage": stage,
                    "presentation_id": str(task["presentation_id"]),
                    "evaluator_family": args.provider,
                    "model": args.model,
                    "response": response,
                },
            )
            append_jsonl(parsed_path, wrapper)
            completed += 1
            if completed % 20 == 0 or completed == len(tasks):
                elapsed = time.time() - started
                print(
                    f"[{args.provider}/{stage}] "
                    f"{completed}/{len(tasks)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return {
        "status": "SORC_RATIONALE_QUALIFICATION_EVALUATOR_COMPLETE",
        "provider": args.provider,
        "model": args.model,
        "stage": stage,
        "tasks": len(tasks),
        "completed": completed,
        "api_called": True,
        "parsed_result_sha256": sha256_file(parsed_path),
        "raw_response_sha256": sha256_file(raw_path),
        "dev_accessed": False,
        "test_accessed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(CONFIG), required=True)
    parser.add_argument("--model")
    parser.add_argument(
        "--stage",
        choices=("score-blind", "score-visible"),
        default="score-blind",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise ValueError("workers must be in 1..16")
    args.model = args.model or os.environ.get(
        f"{args.provider.upper()}_TEACHER_MODEL",
        DEFAULT_MODELS[args.provider],
    )
    return args


def main() -> None:
    print(
        json.dumps(
            run(parse_args()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
