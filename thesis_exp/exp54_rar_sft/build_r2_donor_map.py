"""Build the R2 matched semantic-mismatch donor map.

The formal map requires the frozen Qwen tokenizer. A character-length mode is
provided only for train-only feasibility diagnostics; it can never emit a
formal READY status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.audit_rar0_alignment import (
    file_sha256,
    read_jsonl,
    reject_eval_path,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_REFERENCES = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/data/"
    "label_consistent_reference_sets.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "thesis_exp/outputs/exp54_rar_sft/rar_v2"
DEFAULT_TOKENIZER_LOCK_SPEC = (
    REPO_ROOT
    / "thesis_exp/exp54_rar_sft/configs/qwen_tokenizer_lock_spec.json"
)
DEFAULT_ORACLE_REPORT = (
    DEFAULT_OUTPUT / "audit/r2_solver_oracle_report.json"
)
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def stable_tie_cost(recipient_id: str, donor_id: str, modulo: int = 1000) -> int:
    digest = hashlib.sha256(f"{recipient_id}|{donor_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def flatten_references(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_reference_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        record_id = str(row.get("record_id") or "")
        if not record_id:
            raise ValueError(f"row {row_index}: missing record_id")
        normalized_qa_key = str(row.get("normalized_qa_key") or "")
        if not normalized_qa_key:
            raise ValueError(f"{record_id}: missing normalized_qa_key")
        for reference_index, reference in enumerate(row.get("references") or []):
            reference_id = str(reference.get("reference_id") or "")
            if not reference_id:
                raise ValueError(f"{record_id}/{reference_index}: missing reference_id")
            if reference_id in seen_reference_ids:
                raise ValueError(f"duplicate reference_id: {reference_id}")
            seen_reference_ids.add(reference_id)
            reason = str(reference.get("reason") or "")
            if not reason:
                raise ValueError(f"{reference_id}: empty reason")
            items.append(
                {
                    "record_id": record_id,
                    "normalized_qa_key": normalized_qa_key,
                    "reference_id": reference_id,
                    "reference_index": reference_index,
                    "label_5": int(row["label_5"]),
                    "metric_id": str(row["metric_id"]),
                    "language": str(row["language"]),
                    "reason": reason,
                    "reason_sha256": str(reference["clean_reason_sha256"]),
                }
            )
    return items


def group_key(item: dict[str, Any]) -> tuple[int, str, str]:
    return int(item["label_5"]), str(item["metric_id"]), str(item["language"])


def stratum_id(item: dict[str, Any]) -> str:
    label, metric, language = group_key(item)
    return f"label={label}|metric={metric}|language={language}"


def _active_cycle_ids(
    ordered: list[dict[str, Any]],
    assignment: dict[int, int],
) -> dict[int, str]:
    """Return deterministic IDs for all nontrivial cycles in a full assignment."""
    active_indices = {index for index, donor in assignment.items() if index != donor}
    cycle_by_index: dict[int, str] = {}
    while active_indices:
        start = min(active_indices, key=lambda index: ordered[index]["reference_id"])
        cycle: list[int] = []
        current = start
        while current not in cycle:
            if current not in active_indices:
                raise AssertionError("active assignment does not close over its active subset")
            cycle.append(current)
            current = assignment[current]
        if current != start:
            raise AssertionError("active assignment entered a previously visited cycle")
        reference_ids = [ordered[index]["reference_id"] for index in cycle]
        canonical_start = min(range(len(reference_ids)), key=reference_ids.__getitem__)
        canonical_ids = (
            reference_ids[canonical_start:] + reference_ids[:canonical_start]
        )
        cycle_digest = hashlib.sha256(
            "\n".join(canonical_ids).encode("utf-8")
        ).hexdigest()[:16]
        cycle_id = f"cycle-{cycle_digest}"
        for index in cycle:
            cycle_by_index[index] = cycle_id
            active_indices.remove(index)
    return cycle_by_index


def match_stratum(
    items: list[dict[str, Any]],
    length_fn: Callable[[str], int],
) -> list[dict[str, Any]]:
    """Return a minimum-length-cost partial derangement.

    A diagonal assignment means the reference is deactivated. All non-diagonal
    assignments form a permutation over the active subset. Same-sample edges
    are forbidden, including edges between different references from one row.
    """
    if not items:
        return []
    ordered = sorted(items, key=lambda item: item["reference_id"])
    n = len(ordered)
    lengths = [int(length_fn(item["reason"])) for item in ordered]
    if any(length < 1 for length in lengths):
        raise ValueError("R2 matching requires positive rationale lengths")

    tie_modulo = 1000
    max_diff = max(lengths) - min(lengths)
    length_scale = n * tie_modulo + 1
    max_real_edge = max_diff * length_scale + tie_modulo - 1
    deactivate_penalty = n * max_real_edge + n * tie_modulo + 1
    forbidden_cost = deactivate_penalty * (n + 1)

    costs = np.full((n, n), forbidden_cost, dtype=np.int64)
    for recipient_index, recipient in enumerate(ordered):
        for donor_index, donor in enumerate(ordered):
            tie = stable_tie_cost(recipient["reference_id"], donor["reference_id"], tie_modulo)
            if recipient_index == donor_index:
                costs[recipient_index, donor_index] = deactivate_penalty + tie
            elif (
                recipient["record_id"] != donor["record_id"]
                and recipient["normalized_qa_key"] != donor["normalized_qa_key"]
            ):
                length_cost = abs(lengths[recipient_index] - lengths[donor_index])
                costs[recipient_index, donor_index] = length_cost * length_scale + tie

    row_indices, col_indices = linear_sum_assignment(costs)
    assignment = dict(zip(row_indices.tolist(), col_indices.tolist()))
    cycle_by_index = _active_cycle_ids(ordered, assignment)
    output: list[dict[str, Any]] = []
    for recipient_index, recipient in enumerate(ordered):
        donor_index = assignment[recipient_index]
        donor = ordered[donor_index]
        active = donor_index != recipient_index
        if active and recipient["record_id"] == donor["record_id"]:
            raise AssertionError("same-sample R2 assignment escaped the forbidden edge")
        if active and recipient["normalized_qa_key"] == donor["normalized_qa_key"]:
            raise AssertionError("same-content R2 assignment escaped the forbidden edge")
        if costs[recipient_index, donor_index] >= forbidden_cost:
            raise AssertionError("forbidden R2 assignment selected")
        output.append(
            {
                "stratum_id": stratum_id(recipient),
                "recipient_record_id": recipient["record_id"],
                "recipient_sample_id": recipient["record_id"],
                "recipient_reference_id": recipient["reference_id"],
                "recipient_reference_index": recipient["reference_index"],
                "recipient_normalized_qa_key": recipient["normalized_qa_key"],
                "donor_record_id": donor["record_id"] if active else "",
                "donor_sample_id": donor["record_id"] if active else "",
                "donor_reference_id": donor["reference_id"] if active else "",
                "donor_reference_index": donor["reference_index"] if active else "",
                "donor_normalized_qa_key": donor["normalized_qa_key"] if active else "",
                "label_5": recipient["label_5"],
                "metric_id": recipient["metric_id"],
                "language": recipient["language"],
                "recipient_length": lengths[recipient_index],
                "donor_length": lengths[donor_index] if active else "",
                "absolute_length_difference": (
                    abs(lengths[recipient_index] - lengths[donor_index]) if active else ""
                ),
                "active": active,
                "deactivation_reason": "" if active else "no_full_derangement_at_reference_position",
                "inactive_reason": "" if active else "no_legal_strict_permutation",
                "cycle_id": cycle_by_index.get(recipient_index, ""),
                "solver_name": "scipy.optimize.linear_sum_assignment",
                "solver_version": scipy.__version__,
            }
        )

    active_rows = [row for row in output if row["active"]]
    active_recipient_ids = {row["recipient_reference_id"] for row in active_rows}
    active_donor_ids = {row["donor_reference_id"] for row in active_rows}
    if active_recipient_ids != active_donor_ids:
        raise AssertionError("active R2 mapping is not a permutation over the active subset")
    if len(active_donor_ids) != len(active_rows):
        raise AssertionError("R2 donor reference reused within stratum")
    if any(not row["cycle_id"] for row in active_rows):
        raise AssertionError("active R2 mapping is missing a permutation cycle ID")
    return output


def build_donor_map(
    reference_rows: list[dict[str, Any]],
    length_fn: Callable[[str], int],
) -> list[dict[str, Any]]:
    items = flatten_references(reference_rows)
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[group_key(item)].append(item)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        output.extend(match_stratum(grouped[key], length_fn))
    return sorted(output, key=lambda row: row["recipient_reference_id"])


def validate_map(
    reference_rows: list[dict[str, Any]],
    donor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    items = flatten_references(reference_rows)
    by_reference = {item["reference_id"]: item for item in items}
    map_by_recipient = {row["recipient_reference_id"]: row for row in donor_rows}
    active = [row for row in donor_rows if row["active"]]
    inactive = [row for row in donor_rows if not row["active"]]
    checks = {
        "one_map_row_per_reference": len(map_by_recipient) == len(items) == len(donor_rows),
        "all_references_covered": set(map_by_recipient) == set(by_reference),
        "no_fixed_points": all(
            row["recipient_reference_id"] != row["donor_reference_id"] for row in active
        ),
        "no_same_sample_donors": all(
            row["recipient_record_id"] != row["donor_record_id"] for row in active
        ),
        "no_same_content_donors": all(
            row["recipient_normalized_qa_key"] != row["donor_normalized_qa_key"]
            for row in active
        ),
        "score_metric_language_match": all(
            (
                row["label_5"],
                row["metric_id"],
                row["language"],
            )
            == (
                by_reference[row["donor_reference_id"]]["label_5"],
                by_reference[row["donor_reference_id"]]["metric_id"],
                by_reference[row["donor_reference_id"]]["language"],
            )
            for row in active
        ),
        "active_donors_unique": len({row["donor_reference_id"] for row in active})
        == len(active),
        "active_subset_permutation": {row["recipient_reference_id"] for row in active}
        == {row["donor_reference_id"] for row in active},
        "inactive_rows_have_no_donor": all(
            not row["donor_reference_id"] and not row["donor_record_id"] for row in inactive
        ),
        "active_rows_have_cycle_ids": all(row["cycle_id"] for row in active),
    }
    length_differences = [
        int(row["absolute_length_difference"]) for row in active
    ]
    strata_counts = Counter(
        (row["label_5"], row["metric_id"], row["language"]) for row in donor_rows
    )
    inactive_strata_counts = Counter(
        (row["label_5"], row["metric_id"], row["language"]) for row in inactive
    )
    active_references_by_label = Counter(int(row["label_5"]) for row in active)
    inactive_references_by_label = Counter(int(row["label_5"]) for row in inactive)
    source_rows_by_label: dict[int, set[str]] = defaultdict(set)
    active_rows_by_label: dict[int, set[str]] = defaultdict(set)
    for item in items:
        source_rows_by_label[int(item["label_5"])].add(str(item["record_id"]))
    for row in active:
        active_rows_by_label[int(row["label_5"])].add(str(row["recipient_record_id"]))
    row_coverage_by_label = []
    for label in sorted(source_rows_by_label):
        source_row_count = len(source_rows_by_label[label])
        active_row_count = len(active_rows_by_label[label])
        row_coverage_by_label.append(
            {
                "label_5": label,
                "source_reason_rows": source_row_count,
                "active_reason_rows": active_row_count,
                "fully_deactivated_rows": source_row_count - active_row_count,
                "active_row_coverage": (
                    active_row_count / source_row_count if source_row_count else 0.0
                ),
                "active_references": active_references_by_label[label],
                "inactive_references": inactive_references_by_label[label],
            }
        )
    return {
        "checks": checks,
        "counts": {
            "source_references": len(items),
            "active_references": len(active),
            "inactive_references": len(inactive),
            "active_coverage": len(active) / len(items) if items else 0.0,
            "strata": len(strata_counts),
            "strata_with_deactivation": len(inactive_strata_counts),
            "max_stratum_references": max(strata_counts.values(), default=0),
            "max_absolute_length_difference": max(length_differences, default=0),
            "mean_absolute_length_difference": (
                sum(length_differences) / len(length_differences)
                if length_differences
                else 0.0
            ),
        },
        "inactive_by_stratum": [
            {
                "label_5": key[0],
                "metric_id": key[1],
                "language": key[2],
                "inactive_references": count,
                "total_references": strata_counts[key],
            }
            for key, count in sorted(inactive_strata_counts.items())
        ],
        "row_coverage_by_label": row_coverage_by_label,
    }


def character_length(text: str) -> int:
    return len(text)


def tokenizer_length_function(
    tokenizer_path: Path,
    lock_spec: dict[str, Any],
) -> tuple[Callable[[str], int], dict[str, Any]]:
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Formal tokenizer mode requires transformers in the frozen training environment"
        ) from error
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(tokenizer_path)
    model_id = str(lock_spec.get("model_id") or "")
    upstream_revision = str(lock_spec.get("upstream_revision") or "")
    if not model_id:
        raise ValueError("tokenizer lock spec is missing model_id")
    if not IMMUTABLE_REVISION.fullmatch(upstream_revision):
        raise ValueError("tokenizer upstream_revision must be an immutable 40-character SHA")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), trust_remote_code=True, local_files_only=True
    )

    def length_fn(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    tokenizer_files = sorted(path for path in tokenizer_path.rglob("*") if path.is_file())
    hash_payload = [
        {
            "path": str(path.relative_to(tokenizer_path)),
            "sha256": file_sha256(path),
        }
        for path in tokenizer_files
        if path.name
        in {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "config.json",
        }
    ]
    if not hash_payload:
        raise ValueError(f"No tokenizer/config files found under {tokenizer_path}")
    actual_hashes = {
        row["path"]: row["sha256"] for row in hash_payload
    }
    expected_hashes = {
        str(path): str(digest)
        for path, digest in (lock_spec.get("expected_files") or {}).items()
    }
    mismatches = {
        path: {
            "expected": expected,
            "actual": actual_hashes.get(path),
        }
        for path, expected in expected_hashes.items()
        if actual_hashes.get(path) != expected
    }
    if mismatches:
        raise ValueError(
            "tokenizer snapshot does not match the frozen upstream file hashes: "
            + json.dumps(mismatches, sort_keys=True)
        )

    unicode_probes = {}
    for probe_id, text in {
        "chinese": "答案缺少关键推导。",
        "english": "The answer omits a required derivation.",
        "mixed": "步骤 2 uses the wrong formula.",
        "fullwidth_nfkc_pair": "ＡＢＣ，得分不是理由。",
        "emoji_special": "证据✅，结论⚠️。",
    }.items():
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"tokenizer produced no tokens for Unicode probe {probe_id}")
        unicode_probes[probe_id] = {
            "token_count": len(token_ids),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }

    lock = {
        "status": "QWEN_TOKENIZER_REVISION_LOCKED",
        "model_id": model_id,
        "upstream_revision": upstream_revision,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_files": hash_payload,
        "expected_file_hashes": expected_hashes,
        "expected_file_hashes_match": True,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "unicode_probes": unicode_probes,
        "runtime": {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
        },
        "tokenizer_lock_sha256": hashlib.sha256(
            json.dumps(
                {
                    "model_id": model_id,
                    "upstream_revision": upstream_revision,
                    "tokenizer_files": hash_payload,
                    "unicode_probes": unicode_probes,
                    "transformers": transformers.__version__,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    return length_fn, lock


def _write_diagnostic_csv(path: Path, donor_rows: list[dict[str, Any]]) -> None:
    write_csv(path, donor_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--length-mode",
        choices=("character-diagnostic", "tokenizer"),
        default="character-diagnostic",
    )
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument(
        "--tokenizer-lock-spec",
        type=Path,
        default=DEFAULT_TOKENIZER_LOCK_SPEC,
    )
    parser.add_argument(
        "--oracle-report",
        type=Path,
        default=DEFAULT_ORACLE_REPORT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_eval_path(args.references)
    if not args.references.exists():
        raise FileNotFoundError(args.references)
    reference_rows = read_jsonl(args.references, protect_split=True)

    if args.length_mode == "tokenizer":
        if args.tokenizer_path is None:
            raise ValueError("--tokenizer-path is required for formal tokenizer mode")
        if not args.tokenizer_lock_spec.exists():
            raise FileNotFoundError(args.tokenizer_lock_spec)
        if not args.oracle_report.exists():
            raise FileNotFoundError(args.oracle_report)
        lock_spec = json.loads(args.tokenizer_lock_spec.read_text(encoding="utf-8"))
        oracle_report = json.loads(args.oracle_report.read_text(encoding="utf-8"))
        if oracle_report.get("status") != "R2_MATCHER_ORACLE_PASS":
            raise ValueError("formal tokenizer mode requires R2_MATCHER_ORACLE_PASS")
        matcher_source_hash = file_sha256(Path(__file__))
        if oracle_report.get("matcher_source_sha256") != matcher_source_hash:
            raise ValueError("oracle report does not cover the current matcher source")
        length_fn, tokenizer_lock = tokenizer_length_function(
            args.tokenizer_path,
            lock_spec,
        )
        oracle_lock = {
            "path": str(args.oracle_report),
            "sha256": file_sha256(args.oracle_report),
            "matcher_source_sha256": matcher_source_hash,
            "status": oracle_report["status"],
        }
        status_name = "R2_TOKENIZED_DONOR_MAP_READY"
    else:
        length_fn = character_length
        tokenizer_lock = {
            "mode": "character_length_only",
            "formal_training_allowed": False,
        }
        oracle_lock = {
            "status": "NOT_REQUIRED_FOR_CHARACTER_DIAGNOSTIC",
        }
        status_name = "R2_DONOR_DIAGNOSTIC_ONLY"

    donor_rows = build_donor_map(reference_rows, length_fn)
    for row in donor_rows:
        row["tokenizer_revision"] = tokenizer_lock.get("upstream_revision", "")
        row["tokenizer_lock_sha256"] = tokenizer_lock.get("tokenizer_lock_sha256", "")
    validation = validate_map(reference_rows, donor_rows)
    all_checks_pass = all(validation["checks"].values())
    status = status_name if all_checks_pass else "R2_DONOR_MAP_NO_GO"
    report = {
        "status": status,
        "length_mode": args.length_mode,
        "formal_training_allowed": False,
        "r2_r3_manifest_build_allowed": status == "R2_TOKENIZED_DONOR_MAP_READY",
        "dev_accessed": False,
        "test_accessed": False,
        "api_used": False,
        "model_used": args.length_mode == "tokenizer",
        "training_used": False,
        "reference_source_sha256": file_sha256(args.references),
        "tokenizer_lock": tokenizer_lock,
        "oracle_lock": oracle_lock,
        **validation,
    }

    out_dir: Path = args.out_dir
    if args.length_mode == "tokenizer":
        map_path = out_dir / "data/shuffled_rationale_donor_map.jsonl"
        report_path = out_dir / "audit/r2_donor_match_report.json"
        diagnostic_path = out_dir / "audit/donor_match_diagnostics.csv"
        write_jsonl(map_path, donor_rows)
        write_json(report_path, report)
        _write_diagnostic_csv(diagnostic_path, donor_rows)
        write_json(
            out_dir / "protocol/r2_donor_map_lock.json",
            {
                "status": status,
                "reference_source_sha256": file_sha256(args.references),
                "donor_map_sha256": file_sha256(map_path),
                "diagnostics_sha256": file_sha256(diagnostic_path),
                "tokenizer_lock": tokenizer_lock,
                "oracle_lock": oracle_lock,
                "formal_training_allowed": False,
                "next_required_gate": "R2_R3_ACTIVE_MASK_IDENTICAL",
            },
        )
    else:
        write_jsonl(
            out_dir / "audit/r2_character_length_donor_map_diagnostic.jsonl",
            donor_rows,
        )
        write_json(
            out_dir / "audit/r2_character_length_feasibility_report.json",
            report,
        )

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not all_checks_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
