"""Build unique train-only answer-blind rubric compilation units and frozen locks."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.exp41_rubric_bridge.common import (  # noqa: E402
    FORBIDDEN_UNIT_FIELDS, PROCESSED_PATH, PROMPT_PATH, ROOT, SCHEMA_PATH, TRAIN_PATH,
    canonical_metric, ensure_output_dirs, non_label_metadata, normalize_text, raw_rubric_text,
    read_jsonl, rubric_unit_id, rubric_unit_key, sha256_file, stable_hash, write_csv, write_json, write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_PATH)
    parser.add_argument("--processed-jsonl", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.out_dir)
    rows = read_jsonl(args.train_jsonl)
    if len(rows) != 2654:
        raise ValueError(f"Exp41A requires exactly 2654 paper-like train rows, found {len(rows)}")
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[rubric_unit_key(row)].append(row)
    units = []
    for key in sorted(grouped):
        members = grouped[key]
        exemplar = members[0]
        unit = {
            "rubric_unit_id": rubric_unit_id(exemplar),
            "question_key": str(exemplar["question_key"]),
            "question": str(exemplar.get("question") or ""),
            "metric": canonical_metric(exemplar),
            "metric_group": str(exemplar.get("metric_group") or "unknown"),
            "raw_rubric": raw_rubric_text(exemplar),
            "language": str(exemplar.get("language") or "unknown"),
            "metadata": non_label_metadata(exemplar),
            "answer_row_count": len(members),
        }
        if any(field in unit for field in FORBIDDEN_UNIT_FIELDS):
            raise RuntimeError("Rubric unit contains an answer/label field")
        units.append(unit)
    if len(units) != 1044 or sum(int(unit["answer_row_count"]) for unit in units) != 2654:
        raise RuntimeError(f"Frozen unit resolution changed: units={len(units)} rows={sum(unit['answer_row_count'] for unit in units)}")
    if len({unit["rubric_unit_id"] for unit in units}) != len(units):
        raise RuntimeError("Rubric unit IDs are not unique")
    serialized = json.dumps(units, ensure_ascii=False, sort_keys=True)
    leaked = sorted(field for field in FORBIDDEN_UNIT_FIELDS if f'"{field}"' in serialized)
    if leaked:
        raise RuntimeError(f"Answer/label leakage in rubric units: {leaked}")

    unit_path = args.out_dir / "private/rubric_units/exp41a_rubric_units.jsonl"
    write_jsonl(unit_path, units)
    distribution = Counter((unit["language"], unit["metric_group"], unit["answer_row_count"]) for unit in units)
    write_csv(
        args.out_dir / "tables/exp41a_rubric_unit_distribution.csv",
        [
            {"language": key[0], "metric_group": key[1], "answer_rows_per_unit": key[2], "rubric_units": count,
             "shared_answer_rows": count * int(key[2])}
            for key, count in sorted(distribution.items())
        ],
    )
    prompt_copy = args.out_dir / "prompts/exp41a_answer_blind_rubric_compiler.md"
    schema_copy = args.out_dir / "schemas/exp41a_compiled_rubric_schema.json"
    shutil.copy2(PROMPT_PATH, prompt_copy)
    shutil.copy2(SCHEMA_PATH, schema_copy)
    protocol = {
        "experiment": "Exp41A RUBRIC-Bridge", "provider": "qwen", "model": "qwen3.7-max",
        "temperature": 0, "thinking": "disabled", "response_format": "json_object",
        "unit_key": "question_key + canonical_metric + normalized_raw_rubric + language",
        "rubric_unit_count": len(units), "answer_row_count": 2654,
        "teacher_role": "answer_blind_rubric_compiler_not_judge", "target_scope": "rubric_only",
        "prompt_sha256": sha256_file(PROMPT_PATH), "schema_sha256": sha256_file(SCHEMA_PATH),
        "rubric_unit_sha256": sha256_file(unit_path), "forbidden_input_fields": sorted(FORBIDDEN_UNIT_FIELDS),
        "qualification_gate": {
            "all_units_completed": True, "schema_success_min": 0.99, "target_scope_success": 1.0,
            "quote_validity": 1.0, "answer_label_leakage": 0, "duplicate_criterion_rate_max": 0.01,
            "units_coverage_ge_0p60_rate_min": 0.95, "compiled_length_le_256_rate": 1.0,
        },
        "dev_access_count": 0, "test_access_count": 0,
    }
    protocol_path = args.out_dir / "configs/exp41a_compiler_protocol_lock.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError(f"Frozen compiler protocol changed: {protocol_path}")
    write_json(protocol_path, protocol)
    training_lock = {
        "experiment": "Exp41A train-only GroupCV", "variants": [
            "v0_original_hard", "v0h_human_soft", "v1_raw_rubric", "v2_deterministic_checklist",
            "v3_rubric_bridge", "v4_shuffled_compiled_rubric", "v5_human_soft_logit_adjustment",
        ],
        "formal_variants": ["v0h_human_soft", "v1_raw_rubric", "v2_deterministic_checklist", "v3_rubric_bridge", "v4_shuffled_compiled_rubric", "v5_human_soft_logit_adjustment"],
        "rows_per_variant": 2654, "folds": 5, "groups": "question_key",
        "stratify": "rounded_label_x_language", "random_state": 42,
        "model": "Qwen3-Reranker-0.6B", "epochs": 10, "learning_rate": 2e-5,
        "optimizer": "AdamW", "weight_decay": 0.01, "warmup_ratio": 0.05,
        "micro_batch": 4, "gradient_accumulation": 32, "effective_batch": 128,
        "max_length": 2048, "question_budget": 256, "metric_budget": 64, "rubric_budget": 256,
        "loss": "standard_hard_or_soft_cross_entropy", "logit_adjustment_tau_v5": 1.0,
        "sample_weighting": False, "class_weighting": False, "auxiliary_head": False,
        "custom_ordinal_or_risk_loss": False, "fixed_final_epoch": 10,
        "heldout_checkpoint_selection": False, "label_free_inference_preprocessing": True,
        "dev_access_count": 0, "test_access_count": 0,
    }
    training_path = args.out_dir / "configs/exp41a_training_matrix_lock.json"
    if training_path.exists() and json.loads(training_path.read_text(encoding="utf-8")) != training_lock:
        raise RuntimeError(f"Frozen training matrix changed: {training_path}")
    write_json(training_path, training_lock)
    write_json(args.out_dir / "configs/exp41a_groupcv_config.json", {
        "folds": 5, "seed": 42, "group": "question_key", "evaluation_rows": "original_human_only",
        "label_free_inference_preprocessing": True, "dev_access_count": 0, "test_access_count": 0,
    })
    write_json(args.out_dir / "hashes/exp41a_rubric_unit_hashes.json", {
        "train_path": str(args.train_jsonl), "train_rows": 2654, "train_sha256": sha256_file(args.train_jsonl),
        "processed_path_audit_only": str(args.processed_jsonl), "processed_sha256": sha256_file(args.processed_jsonl),
        "rubric_unit_rows": len(units), "rubric_unit_sha256": sha256_file(unit_path),
        "unit_identity_sha256": stable_hash([(unit["rubric_unit_id"], normalize_text(unit["raw_rubric"])) for unit in units]),
        "dev_access_count": 0, "test_access_count": 0,
    })
    write_csv(args.out_dir / "tables/exp41a_compiler_completion.csv", [{
        "stage": "units_prepared", "rubric_units": len(units), "expected_outputs": len(units),
        "completed_outputs": 0, "completion_rate": 0.0, "dev_access_count": 0, "test_access_count": 0,
    }])
    print(json.dumps({"status": "PREPARED", "rubric_units": len(units), "answer_rows": 2654, "dev_access_count": 0, "test_access_count": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
