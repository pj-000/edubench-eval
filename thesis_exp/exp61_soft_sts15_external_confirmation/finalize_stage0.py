from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp61_soft_sts15_external_confirmation.audit_dataset import (
    OUTPUT_ROOT,
    sha256_bytes,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    audit_path = args.output_root / "audit/soft_sts15_stage0_audit.json"
    tokenizer_path = args.output_root / "audit/tokenizer_length_audit.json"
    data_decision_path = args.output_root / "decision/stage0_data_decision.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    data_decision = json.loads(data_decision_path.read_text(encoding="utf-8"))
    manifest_path = args.output_root / "data/split_manifest.jsonl"
    manifest_hash = sha256_bytes(manifest_path.read_bytes())

    gates = {
        "data_stage0_pass": (
            data_decision["status"] == "EXP61_REVISED_DATA_STAGE0_PASS_TOKENIZER_AUDIT_PENDING"
            and data_decision["all_data_gates_pass"] is True
        ),
        "data_gates_all_true": all(audit["data_gates"].values()),
        "manifest_gates_all_true": all(audit["manifest_gates"].values()),
        "split_gates_all_true": all(audit["frozen_component_split"]["gates"].values()),
        "manifest_hash_bound": manifest_hash == audit["split_manifest_sha256"],
        "tokenizer_audit_pass": tokenizer["status"] == "EXP61_TOKENIZER_LENGTH_AUDIT_PASS",
        "tokenizer_gates_all_true": all(tokenizer["gates"].values()),
        "tokenizer_manifest_bound": tokenizer["split_manifest_sha256"] == manifest_hash,
        "no_training_performed": (
            audit["model_training_performed"] is False
            and tokenizer["model_training_performed"] is False
        ),
        "no_gpu_used": audit["gpu_used"] is False and tokenizer["gpu_used"] is False,
    }
    failed = sorted(name for name, value in gates.items() if not value)
    if failed:
        raise RuntimeError("Stage 0 finalization failed closed: " + ", ".join(failed))

    decision = {
        "status": "EXP61_STAGE0_REVISED_GO_TO_PROTOCOL_FREEZE",
        "protocol_freeze_authorized": True,
        "formal_gpu_training_authorized": False,
        "gates": gates,
        "bindings": {
            "data_audit_sha256": sha256_bytes(audit_path.read_bytes()),
            "split_manifest_sha256": manifest_hash,
            "tokenizer_audit_sha256": sha256_bytes(tokenizer_path.read_bytes()),
            "selected_max_length": tokenizer["selected_max_length"],
            "input_template_sha256": tokenizer["input_template_sha256"],
        },
        "next_required_stage": (
            "freeze trainer, mapping, analysis, source lock, and real-model no-update preflight"
        ),
    }
    write_json(args.output_root / "decision/stage0_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
