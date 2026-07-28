"""Run one evaluator family for the Exp54 minimal rationale blind audit.

The command is a dry run unless --execute is supplied. API keys are read only
from QWEN_API_KEY or DEEPSEEK_API_KEY and are never written to disk.
"""

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

from thesis_exp.exp54_rar_sft import REPO_ROOT


OUTPUT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "rationale_blind_audit"
)
CONFIG = {
    "qwen": {
        "key": "QWEN_API_KEY",
        "base_key": "QWEN_BASE_URL",
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "deepseek": {
        "key": "DEEPSEEK_API_KEY",
        "base_key": "DEEPSEEK_BASE_URL",
        "base": "https://api.deepseek.com",
    },
}


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: expected JSON objects")
                rows.append(value)
    return rows


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(compact_json(value) + "\n")


def parse_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("evaluator output is not an object")
    return value


def validate(value: dict[str, Any], schema: dict[str, Any], task_id: str) -> None:
    if value.get("presentation_id") != task_id:
        raise ValueError("presentation_id differs")
    required = set(schema["required"])
    if set(value) != required:
        raise ValueError("response fields differ from the exact schema")
    properties = schema["properties"]
    for name, field in properties.items():
        current = value[name]
        if field.get("type") == "string" and not isinstance(current, str):
            raise ValueError(f"{name}: expected string")
        if "enum" in field and current not in field["enum"]:
            raise ValueError(f"{name}: value outside enum")
        if "maxLength" in field and len(current) > int(field["maxLength"]):
            raise ValueError(f"{name}: string too long")
        if "minLength" in field and len(current) < int(field["minLength"]):
            raise ValueError(f"{name}: string too short")
        if "pattern" in field and re.fullmatch(field["pattern"], current) is None:
            raise ValueError(f"{name}: pattern differs")


def call_api(
    *,
    provider: str,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    task: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    provider_config = CONFIG[provider]
    key = os.environ.get(provider_config["key"], "")
    if not key:
        raise RuntimeError(f"missing {provider_config['key']}")
    base = os.environ.get(
        provider_config["base_key"], provider_config["base"]
    ).rstrip("/")
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Evaluation input:\n"
                    + compact_json(task)
                    + "\n\nRequired JSON schema:\n"
                    + compact_json(schema)
                ),
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if provider == "qwen":
        body["enable_thinking"] = False
    else:
        body["thinking"] = {"type": "disabled"}
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc


def response_text(response: dict[str, Any]) -> str:
    choice = (response.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty evaluator response")
    return content


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage_name = args.stage.replace("-", "_")
    tasks_path = OUTPUT_ROOT / "private" / f"{stage_name}_tasks.jsonl"
    prompt_path = (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/prompts/"
        f"rationale_audit_{stage_name}_v2.txt"
    )
    schema_path = (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/schemas/"
        f"rationale_audit_{stage_name}_v2.schema.json"
    )
    tasks = read_jsonl(tasks_path)
    if len(tasks) != 480:
        raise ValueError(f"{tasks_path}: expected 480 presentations")
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    parsed_path = (
        OUTPUT_ROOT
        / "private"
        / "judgments"
        / f"{args.provider}_{safe_model}_{stage_name}.jsonl"
    )
    raw_path = (
        OUTPUT_ROOT
        / "raw_api"
        / f"{args.provider}_{safe_model}_{stage_name}.jsonl"
    )
    existing = {
        str(row["presentation_id"]): row for row in read_jsonl(parsed_path)
    }
    pending = [
        task
        for task in tasks
        if str(task["presentation_id"]) not in existing
    ]
    if not args.execute:
        return {
            "status": "DRY_RUN",
            "provider": args.provider,
            "model": args.model,
            "stage": stage_name,
            "tasks": len(tasks),
            "already_complete": len(existing),
            "pending": len(pending),
            "api_called": False,
        }

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
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"{task['presentation_id']}: {last_error}"
        ) from last_error

    completed = len(existing)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            response, parsed = future.result()
            append_jsonl(
                raw_path,
                {
                    "presentation_id": task["presentation_id"],
                    "response": response,
                },
            )
            append_jsonl(parsed_path, parsed)
            completed += 1
            if completed % 20 == 0 or completed == len(tasks):
                elapsed = time.time() - started
                print(
                    f"[{args.provider}/{stage_name}] "
                    f"{completed}/{len(tasks)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return {
        "status": "EVALUATOR_STAGE_COMPLETE",
        "provider": args.provider,
        "model": args.model,
        "stage": stage_name,
        "tasks": len(tasks),
        "completed": completed,
        "api_called": True,
        "parsed_path": parsed_path.relative_to(REPO_ROOT).as_posix(),
        "raw_path": raw_path.relative_to(REPO_ROOT).as_posix(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(CONFIG), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--stage",
        choices=("score-blind", "score-visible"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


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
