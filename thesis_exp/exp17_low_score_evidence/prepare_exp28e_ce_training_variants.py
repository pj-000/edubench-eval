"""Build ordinary-CE Exp28 training variants from train-only teacher audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.exp02.build_exp02_dataset import convert_row


DEFAULT_SPLIT_DIR = Path("thesis_exp/data/splits/paper_like_triple_seed42")
DEFAULT_TEACHER_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28c_teacher_protocol_evaluation_seed42"
)
DEFAULT_QUALIFICATION_DECISION = (
    DEFAULT_TEACHER_DIR / "decision" / "exp28c_sealed_qualification_protocol_decision.json"
)
DEFAULT_ADJUDICATION = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_model_adjudication_seed42/"
    "private/exp28e_model_adjudication.jsonl"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42"
)
VARIANTS = (
    "b0_original_human",
    "b1_primary_teacher_all",
    "b2_selective_dual_teacher",
    "b3_filter_unresolved",
    "b4_random_transition_control",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sid(row: dict[str, Any]) -> str:
    value = row.get("record_id") or row.get("sample_id") or row.get("id")
    if not value:
        raise ValueError("Missing sample ID")
    return str(value)


def stable_key(*values: Any) -> str:
    payload = "|".join(str(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_valid_outputs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    outputs = {}
    for row in read_jsonl(path):
        annotation = row.get("annotation")
        if not isinstance(annotation, dict) or row.get("schema_errors"):
            continue
        outputs[str(row["sample_id"])] = annotation
    return outputs


def load_protocol(path: Path, override: str | None) -> str:
    if override:
        return override
    if not path.exists():
        raise FileNotFoundError(path)
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("status") != "READY_FOR_FULL_TRAIN_ANNOTATION":
        raise ValueError("Sealed qualification does not authorize training-data construction")
    protocol = decision.get("selected_protocol")
    if not protocol:
        raise ValueError("Qualification decision has no selected protocol")
    return str(protocol)


def load_adjudication(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in read_jsonl(path) if row.get("sample_id")}


def accepted_teacher_change(
    original: int,
    primary: dict[str, Any],
    secondary: dict[str, Any] | None,
    adjudication: dict[str, Any] | None,
    confidence_threshold: float,
) -> tuple[bool, str]:
    primary_score = int(primary["score"])
    if primary_score == original:
        return False, "primary_confirms_original"
    if secondary is None:
        return False, "secondary_missing"
    secondary_score = int(secondary["score"])
    if primary_score != secondary_score:
        return False, "teacher_score_disagreement"
    if min(float(primary.get("confidence", 0.0)), float(secondary.get("confidence", 0.0))) < confidence_threshold:
        return False, "teacher_confidence_below_threshold"
    risky_transition = (
        abs(primary_score - original) >= 2
        or (original <= 2 and primary_score >= 4)
        or (original >= 4 and primary_score <= 2)
    )
    if risky_transition:
        if adjudication is None:
            return False, "risky_transition_requires_model_adjudication"
        confidence = str(adjudication.get("confidence") or "").lower()
        final_score = adjudication.get("final_score")
        if confidence not in {"high", "medium"} or int(final_score or -1) != primary_score:
            return False, "risky_transition_not_confirmed_by_model_adjudication"
        return True, "dual_teacher_plus_model_adjudication"
    return True, "dual_teacher_consensus_adjacent_change"


def converted(source: dict[str, Any], split: str, index: int, target: int, provenance: str) -> dict[str, Any]:
    row = convert_row(source, split, index)
    row["label"] = target - 1
    row["label_5"] = target
    row["original_label_5"] = int(source["label_5"])
    row["target_provenance"] = provenance
    row["training_loss"] = "ordinary_cross_entropy"
    return row


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(args.qualification_decision, args.protocol)
    train_source = read_jsonl(args.split_dir / "train.jsonl")
    dev_source = read_jsonl(args.split_dir / "dev.jsonl")
    if len(train_source) != 2654 or len(dev_source) != 664:
        raise ValueError("Exp28E requires the locked 2654/664 paper train/dev split")
    source_by_id = {sid(row): row for row in train_source}
    primary_path = args.teacher_dir / "private" / "qwen" / protocol / "all_train.jsonl"
    secondary_path = args.teacher_dir / "private" / "deepseek" / protocol / "secondary_route.jsonl"
    primary = load_valid_outputs(primary_path)
    secondary = load_valid_outputs(secondary_path)
    if set(primary) != set(source_by_id):
        raise ValueError(f"Primary teacher coverage mismatch: {len(primary)}/2654")
    if not set(secondary) <= set(source_by_id):
        raise ValueError("Secondary outputs contain rows outside paper train")
    adjudications = load_adjudication(args.adjudication)
    if not set(adjudications) <= set(source_by_id):
        raise ValueError("Adjudication contains rows outside paper train")

    decisions = []
    accepted_changes: dict[str, int] = {}
    unresolved_conflicts: set[str] = set()
    for sample_id, source in source_by_id.items():
        original = int(source["label_5"])
        primary_score = int(primary[sample_id]["score"])
        accepted, reason = accepted_teacher_change(
            original,
            primary[sample_id],
            secondary.get(sample_id),
            adjudications.get(sample_id),
            args.confidence_threshold,
        )
        if accepted:
            accepted_changes[sample_id] = primary_score
        elif primary_score != original:
            unresolved_conflicts.add(sample_id)
        decisions.append(
            {
                "sample_id": sample_id,
                "original_label": original,
                "primary_teacher_score": primary_score,
                "secondary_teacher_score": secondary.get(sample_id, {}).get("score"),
                "model_adjudication_score": adjudications.get(sample_id, {}).get("final_score"),
                "accepted_change": accepted,
                "decision_reason": reason,
                "final_selective_label": accepted_changes.get(sample_id, original),
                "reference_status": "teacher_and_model_review_silver_not_human_gold",
            }
        )

    # Negative control: apply the same accepted transition multiset to random rows
    # with the same original label and language, excluding genuinely accepted rows.
    available_by_stratum: dict[tuple[int, str], list[str]] = {}
    for sample_id, source in source_by_id.items():
        if sample_id in accepted_changes:
            continue
        key = (int(source["label_5"]), str(source.get("language") or "unknown"))
        available_by_stratum.setdefault(key, []).append(sample_id)
    for key in available_by_stratum:
        available_by_stratum[key].sort(key=lambda value: stable_key("random-control", value))
    random_changes: dict[str, int] = {}
    for changed_id, target in sorted(accepted_changes.items(), key=lambda item: stable_key("transition", item[0])):
        source = source_by_id[changed_id]
        key = (int(source["label_5"]), str(source.get("language") or "unknown"))
        candidates = available_by_stratum.get(key, [])
        while candidates and candidates[0] in random_changes:
            candidates.pop(0)
        if not candidates:
            fallback = [
                sample_id
                for sample_id, row in source_by_id.items()
                if int(row["label_5"]) == key[0]
                and sample_id not in accepted_changes
                and sample_id not in random_changes
            ]
            fallback.sort(key=lambda value: stable_key("fallback-control", value))
            candidates = fallback
        if not candidates:
            raise RuntimeError(f"No random-control candidate for transition {key[0]}->{target}")
        random_changes[candidates.pop(0)] = target

    variants: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    for index, source in enumerate(train_source):
        sample_id = sid(source)
        original = int(source["label_5"])
        primary_score = int(primary[sample_id]["score"])
        variants["b0_original_human"].append(converted(source, "train", index, original, "original_benchmark_human_label"))
        variants["b1_primary_teacher_all"].append(converted(source, "train", index, primary_score, "qwen_primary_teacher"))
        selective = accepted_changes.get(sample_id, original)
        variants["b2_selective_dual_teacher"].append(
            converted(
                source,
                "train",
                index,
                selective,
                "accepted_dual_teacher_model_silver" if sample_id in accepted_changes else "original_retained",
            )
        )
        if sample_id not in unresolved_conflicts:
            variants["b3_filter_unresolved"].append(
                converted(source, "train", index, selective, "selective_target_unresolved_removed")
            )
        random_target = random_changes.get(sample_id, original)
        variants["b4_random_transition_control"].append(
            converted(
                source,
                "train",
                index,
                random_target,
                "random_transition_control" if sample_id in random_changes else "original_retained",
            )
        )

    dev = [converted(row, "dev", index, int(row["label_5"]), "original_benchmark_dev_label") for index, row in enumerate(dev_source)]
    output_root = args.out_dir / "private" / "datasets"
    for variant, rows in variants.items():
        write_jsonl(output_root / variant / "train.jsonl", rows)
        write_jsonl(output_root / variant / "dev.jsonl", dev)

    summary_rows = []
    for variant, rows in variants.items():
        counts = Counter(int(row["label_5"]) for row in rows)
        summary_rows.append(
            {
                "variant": variant,
                "train_rows": len(rows),
                **{f"label_{score}": counts[score] for score in range(1, 6)},
                "changed_from_original": sum(int(row["label_5"]) != int(row["original_label_5"]) for row in rows),
                "training_loss": "ordinary_cross_entropy",
                "model_input": "question+answer+evaluation_dimension",
            }
        )
    write_csv(
        args.out_dir / "tables" / "exp28e_training_variant_summary.csv",
        summary_rows,
        ["variant", "train_rows", "label_1", "label_2", "label_3", "label_4", "label_5", "changed_from_original", "training_loss", "model_input"],
    )
    transition_counts = Counter(
        (int(source_by_id[sample_id]["label_5"]), target) for sample_id, target in accepted_changes.items()
    )
    random_transition_counts = Counter(
        (int(source_by_id[sample_id]["label_5"]), target) for sample_id, target in random_changes.items()
    )
    transition_rows = []
    for transition in sorted(set(transition_counts) | set(random_transition_counts)):
        transition_rows.append(
            {
                "original_label": transition[0],
                "target_label": transition[1],
                "selective_count": transition_counts[transition],
                "random_control_count": random_transition_counts[transition],
                "matched": transition_counts[transition] == random_transition_counts[transition],
            }
        )
    write_csv(
        args.out_dir / "tables" / "exp28e_transition_control_audit.csv",
        transition_rows,
        ["original_label", "target_label", "selective_count", "random_control_count", "matched"],
    )
    write_csv(
        args.out_dir / "tables" / "exp28e_fusion_decisions_light.csv",
        decisions,
        [
            "sample_id", "original_label", "primary_teacher_score", "secondary_teacher_score",
            "model_adjudication_score", "accepted_change", "decision_reason", "final_selective_label",
            "reference_status",
        ],
    )
    decision = {
        "status": "READY_FOR_DEV_ONLY_CE_TRAINING",
        "protocol": protocol,
        "paper_train_rows": len(train_source),
        "paper_dev_rows": len(dev_source),
        "accepted_changes": len(accepted_changes),
        "unresolved_conflicts_filtered_in_b3": len(unresolved_conflicts),
        "model_adjudication_rows": len(adjudications),
        "variants": {name: len(rows) for name, rows in variants.items()},
        "loss": "ordinary_cross_entropy",
        "student_input": "question+answer+evaluation_dimension",
        "test_read": False,
        "test_file_written": False,
    }
    write_json(args.out_dir / "decision" / "exp28e_training_variant_decision.json", decision)
    report = f"""# Exp28E Ordinary-CE Training Variants

- protocol: `{protocol}`
- paper train/dev: {len(train_source)}/{len(dev_source)}
- accepted selective changes: {len(accepted_changes)}
- unresolved conflicts filtered in B3: {len(unresolved_conflicts)}
- student input: question + answer + evaluation dimension
- loss: ordinary cross-entropy
- test read: no

B0 preserves the original benchmark labels. B1 replaces every target with the primary teacher
score. B2 changes labels only when the locked dual-teacher/adjudication rule accepts the change.
B3 removes unresolved teacher conflicts. B4 applies the same transition multiset to randomly
matched rows and is a negative control for selective targeting.
"""
    report_path = args.out_dir / "reports" / "exp28e_training_variant_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--qualification-decision", type=Path, default=DEFAULT_QUALIFICATION_DECISION)
    parser.add_argument("--protocol", choices=["p0_holistic_zero_shot", "p1_rubric_first", "p2_rubric_verify_then_score"], default=None)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(prepare(parse_args()), ensure_ascii=False, sort_keys=True))
