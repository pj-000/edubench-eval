"""Validate the public Exp27N 54-row blind-review package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp17_low_score_evidence.exp27lr1_common import read_csv, read_jsonl  # noqa: E402


DEFAULT_OUT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp27n_selective_model_adjudication_seed42"
)


def validate(args: argparse.Namespace) -> dict[str, object]:
    out = args.out_dir
    paths = {
        "packets": out / "packets" / "exp27n_selective_adjudication_packets_54.jsonl",
        "template": out / "annotation_templates" / "exp27n_selective_adjudication_template_54.jsonl",
        "schema": out / "schemas" / "exp27n_selective_adjudication_schema.json",
        "prompt": out / "prompts" / "exp27n_gpt56pro_one_session_prompt.md",
        "summary": out / "tables" / "exp27n_selection_summary.csv",
        "distribution": out / "tables" / "exp27n_packet_distribution.csv",
        "leakage": out / "tables" / "exp27n_leakage_audit.csv",
        "decision": out / "decision" / "exp27n_selective_adjudication_prepare_decision.json",
        "report": out / "reports" / "exp27n_selective_adjudication_prepare_report.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    packets = read_jsonl(paths["packets"])
    templates = read_jsonl(paths["template"])
    if len(packets) != 54 or len({row["sample_id"] for row in packets}) != 54:
        raise ValueError("Exp27N must contain 54 unique blind packets")
    if len(templates) != 54 or {row["sample_id"] for row in templates} != {
        row["sample_id"] for row in packets
    }:
        raise ValueError("Exp27N template does not align to the packet")
    forbidden = {
        "original_human_score",
        "human_score",
        "qwen_score",
        "deepseek_score",
        "calibrated_score",
        "teacher_gap",
        "risk_probability",
    }
    for packet in packets:
        if forbidden & set(packet):
            raise ValueError(f"Packet exposes hidden scores: {packet['sample_id']}")
        content = packet["messages"][1]["content"]
        if "<CONTEXT_ONLY_ORIGINAL_TASK>" not in content or "<EVALUATOR_OUTPUT_TO_SCORE>" not in content:
            raise ValueError(f"Packet lacks target isolation: {packet['sample_id']}")
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    if schema.get("additionalProperties") is not False or len(schema.get("required", [])) != 13:
        raise ValueError("Exp27N output schema is incomplete")
    summary = {row["item"]: int(row["count"]) for row in read_csv(paths["summary"])}
    expected = {
        "teacher_audited_rows": 361,
        "direct_accept": 173,
        "weighted_accept": 118,
        "adjudication_required": 70,
        "adjudication_already_reviewed": 16,
        "adjudication_remaining_packet": 54,
    }
    if summary != expected:
        raise ValueError(f"Exp27N selection counts changed: {summary}")
    leakage = read_csv(paths["leakage"])
    if not leakage or any(int(row["count"]) != 0 for row in leakage):
        raise ValueError(f"Exp27N leakage audit failed: {leakage}")
    decision = json.loads(paths["decision"].read_text(encoding="utf-8"))
    if (
        decision.get("status") != "READY_FOR_ONE_GPT56PRO_SESSION"
        or decision.get("packet_rows") != 54
        or decision.get("human_external_validation") is not False
        or decision.get("teacher_api_calls") != 0
        or decision.get("gpu_required") is not False
        or decision.get("model_training_runs") != 0
        or decision.get("proceed_to_361_downstream_dataset_construction") is not False
    ):
        raise ValueError(f"Exp27N decision gate is invalid: {decision}")
    return {
        "status": "PASS",
        "packet_rows": 54,
        "hidden_score_fields": 0,
        "dev_test_overlap": 0,
        "gpu_required": False,
        "next_step": "one_gpt56pro_review_session",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), ensure_ascii=False, sort_keys=True))
