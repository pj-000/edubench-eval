"""Validate and describe the private bundle for isolated Codex pointwise reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import MODULE, read_jsonl, sha256_path, write_json
from .exp48c_common import OUT, packet_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    packets = read_jsonl(packet_path("codex"))
    if len(packets) != 36 or len({row["packet_id"] for row in packets}) != 36:
        raise SystemExit("Codex bundle must contain 36 unique pointwise packets")
    manifest = {
        "status": "CODEX_REVIEW_BUNDLE_READY",
        "packet_path": str(packet_path("codex")),
        "packet_sha256": sha256_path(packet_path("codex")),
        "packet_count": 36,
        "contexts_required": 36,
        "shared_context_forbidden": True,
        "model": "gpt-5.5",
        "verifier_prompt": str(MODULE / "prompts/exp48c_rubric_only_pointwise_verifier_prompt.md"),
        "dispatcher_prompt": str(MODULE / "prompts/exp48c_codex_one_session_prompt.md"),
        "schema": str(MODULE / "schemas/exp48c_rubric_only_score_schema.json"),
        "output": str(args.out_dir / "private/codex_outputs/exp48c_codex_rubric_only_outputs.jsonl"),
    }
    write_json(args.out_dir / "private/pointwise_packets_codex/exp48c_codex_review_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
