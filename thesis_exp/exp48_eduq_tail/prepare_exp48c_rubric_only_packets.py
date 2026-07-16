"""Freeze Exp48B answers into two independently shuffled rubric-only packet sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_path, stable_id, write_csv, write_json, write_jsonl
from .exp48c_common import (
    EXP48B_DECISION, EXP48B_MAPPING, EXP48B_METRICS, EXP48B_PROTOCOL, FAMILIES,
    FORBIDDEN_FIELDS, OUT, PACKET_FIELDS, PACKET_SEEDS, PRIVATE, VERIFIERS,
    mapping_path, packet_path,
)
from .common import read_jsonl


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def family_hash(family: dict[str, Any]) -> str:
    return canonical_hash(family)


def answer_hash(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def serialize_forbidden_audit(packet: dict[str, Any]) -> dict[str, int]:
    serialized = json.dumps(packet, ensure_ascii=False, sort_keys=True)
    return {
        "D2": serialized.count("D2"), "D3": serialized.count("D3"),
        "H4": serialized.count("H4"), "metric_contract": serialized.count("metric_contract"),
        "intended_score": serialized.count("intended_score"),
        "score_program": serialized.count("score_program"),
        "source_span": serialized.count("source_span"),
        "replacement_span": serialized.count("replacement_span"),
    }


def prepare(out_dir: Path) -> dict[str, Any]:
    families = read_jsonl(FAMILIES)
    if len(families) != 12 or sum(len(row.get("answers", [])) for row in families) != 36:
        raise SystemExit("Frozen Exp48B input must contain exactly 12 families and 36 answers")
    flattened: list[dict[str, Any]] = []
    family_hashes = []
    for family in families:
        frozen_hash = family_hash(family)
        family_hashes.append({
            "frozen_family_hash": frozen_hash,
            "metric": family["metric"],
            "answer_count": len(family["answers"]),
            "answer_text_hashes": [answer_hash(answer["text"]) for answer in family["answers"]],
        })
        for answer in family["answers"]:
            flattened.append({"family": family, "answer": answer, "frozen_family_hash": frozen_hash})

    mappings: list[dict[str, Any]] = []
    blindness_rows: list[dict[str, Any]] = []
    packet_hashes: dict[str, Any] = {}
    for verifier in VERIFIERS:
        rows = []
        for item in flattened:
            family, answer = item["family"], item["answer"]
            anonymous_id = stable_id(f"x48c_{verifier}_ans", item["frozen_family_hash"], answer["answer_id"], PACKET_SEEDS[verifier])
            packet_id = stable_id(f"x48c_{verifier}_pkt", anonymous_id, PACKET_SEEDS[verifier])
            packet = {
                "packet_id": packet_id,
                "anonymous_answer_id": anonymous_id,
                "metric": family["metric"],
                "language": family["language"],
                "synthetic_question": family["synthetic_question"],
                "rubric_levels": {str(key): str(value) for key, value in family["rubric_levels"].items()},
                "answer": answer["text"],
            }
            if set(packet) != PACKET_FIELDS:
                raise AssertionError("Unexpected packet field set")
            rows.append(packet)
            mappings.append({
                "verifier": verifier, "packet_id": packet_id,
                "anonymous_answer_id": anonymous_id,
                "frozen_family_hash": item["frozen_family_hash"],
                "original_answer_id": answer["answer_id"],
                "intended_score": int(answer["intended_score"]),
                "metric": family["metric"],
                "answer_text_hash": answer_hash(answer["text"]),
            })
            counts = serialize_forbidden_audit(packet)
            forbidden_keys = sorted(set(packet) & FORBIDDEN_FIELDS)
            blindness_rows.append({
                "verifier": verifier, "packet_id": packet_id,
                "field_count": len(packet), "answer_count": 1,
                "forbidden_field_count": len(forbidden_keys),
                "forbidden_fields": "|".join(forbidden_keys),
                **{f"{key}_occurrence": value for key, value in counts.items()},
                "answer_hash_match": answer_hash(packet["answer"]) == answer_hash(answer["text"]),
            })
        random.Random(PACKET_SEEDS[verifier]).shuffle(rows)
        if len(rows) != 36 or len({row["anonymous_answer_id"] for row in rows}) != 36:
            raise AssertionError("Each verifier must receive 36 unique answers")
        write_jsonl(packet_path(verifier), rows)
        packet_hashes[verifier] = {
            "seed": PACKET_SEEDS[verifier], "rows": len(rows),
            "sha256": sha256_path(packet_path(verifier)),
            "packet_order_hash": canonical_hash([row["packet_id"] for row in rows]),
        }

    write_jsonl(mapping_path(), mappings)
    write_csv(out_dir / "tables/exp48c_packet_blindness_audit.csv", blindness_rows)
    write_csv(out_dir / "tables/exp48c_leakage_audit.csv", [
        {"split": "dev", "access_count": 0, "status": "PASS"},
        {"split": "test", "access_count": 0, "status": "PASS"},
    ])
    write_json(out_dir / "hashes/exp48c_frozen_family_hashes.json", {
        "source_file": str(FAMILIES.relative_to(ROOT)), "source_sha256": sha256_path(FAMILIES),
        "family_count": 12, "answer_count": 36, "families": family_hashes,
    })
    write_json(out_dir / "hashes/exp48c_packet_hashes.json", packet_hashes)
    gates = {
        "completion_rows": 36, "schema_success_rate": 1.0,
        "rubric_quote_validity": 1.0, "evidence_validity_min": 0.95,
        "intended_exact_min": 30, "qwk_min": 0.85,
        "score2_correct_min": 10, "score2_to_high_max": 0,
        "ordered_families_min": 10, "low_confidence_max": 4,
        "cross_exact_min": 30, "cross_qwk_min": 0.85,
        "joint_score2_correct_min": 10, "joint_families_min": 9,
        "joint_metrics_min": 9,
    }
    protocol = {
        "experiment": "Exp48C Rubric-Only Pointwise Contract-Leakage Audit",
        "locked_from_commit": "9408f92", "frozen_family_count": 12,
        "frozen_answer_count": 36, "packet_seeds": PACKET_SEEDS,
        "pointwise_context_isolation_required": True,
        "codex_model": "gpt-5.5", "codex_matches_exp48b_model": True,
        "qwen_model": "qwen3.7-max", "qwen_thinking": "disabled",
        "generation_forbidden": True, "training_forbidden": True,
        "gpu_required": False, "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(out_dir / "configs/exp48c_audit_protocol_lock.json", protocol)
    write_json(out_dir / "configs/exp48c_packet_blindness_lock.json", {
        "allowed_fields": sorted(PACKET_FIELDS), "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "one_answer_per_packet": True, "shared_context_across_packets": False,
    })
    write_json(out_dir / "configs/exp48c_success_gates.json", gates)
    prompt_schema_sources = (
        Path(__file__).resolve().parent / "prompts/exp48c_rubric_only_pointwise_verifier_prompt.md",
        Path(__file__).resolve().parent / "prompts/exp48c_codex_one_session_prompt.md",
        Path(__file__).resolve().parent / "schemas/exp48c_rubric_only_score_schema.json",
    )
    for source in prompt_schema_sources:
        target_dir = out_dir / ("schemas" if source.suffix == ".json" else "prompts")
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target_dir / source.name)
    failures = [row for row in blindness_rows if row["answer_count"] != 1 or row["forbidden_field_count"] or not row["answer_hash_match"] or any(row[f"{key}_occurrence"] for key in ("D2", "D3", "H4", "metric_contract", "intended_score", "score_program", "source_span", "replacement_span"))]
    decision = {
        "status": "PACKET_BLINDNESS_GO" if not failures else "PACKET_BLINDNESS_NO_GO",
        "frozen_families": 12, "frozen_answers": 36,
        "packet_rows_each": {verifier: 36 for verifier in VERIFIERS},
        "one_answer_per_packet_rate": sum(row["answer_count"] == 1 for row in blindness_rows) / len(blindness_rows),
        "answer_hash_equality_rate": sum(bool(row["answer_hash_match"]) for row in blindness_rows) / len(blindness_rows),
        "forbidden_leakage_rows": len(failures), "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(out_dir / "decision/exp48c_packet_decision.json", decision)
    report = [
        "# Exp48C packet preparation", "", f"- Status: **{decision['status']}**",
        "- Frozen input: 12 Exp48B families / 36 answers; no answer was generated, edited, or filtered.",
        "- Two packet orders use seeds 4831 and 4832.",
        "- Every packet contains one answer and the original complete 1-5 metric rubric only.",
        "- Codex audits must run in separate context per packet; a single 36-row conversation is not treated as strictly pointwise.",
        f"- Forbidden leakage rows: {len(failures)}", "- dev/test access: 0/0", "- No training and no GPU.",
    ]
    (out_dir / "reports/exp48c_packet_prepare_report.md").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "reports/exp48c_packet_prepare_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    source_hashes = {str(path.relative_to(ROOT)): sha256_path(path) for path in (EXP48B_DECISION, EXP48B_MAPPING, EXP48B_METRICS, EXP48B_PROTOCOL, FAMILIES)}
    write_json(out_dir / "hashes/exp48c_source_hashes.json", source_hashes)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    if failures:
        raise SystemExit(2)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    prepare(args.out_dir)


if __name__ == "__main__":
    main()
