from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "thesis_exp/outputs/exp61_soft_sts15_external_confirmation"
EXPECTED_COMMIT = "ca754a21e58437c4b843d10161d2838f39230e7f"
EXPECTED_DATA_SHA256 = "8707cdb85e2da9818bc8c08cd90b9b2c17c483e08f7bb7ff5c6dc69b555dde71"
EXPECTED_ROWS = 7890
EXPECTED_STRATA = 20
EXPECTED_ARCHIVE_SHA256 = "c1c175b30e570a3995a79ca0f6169223682ba1f1e0788822554df9c79c7fdaf1"
OFFICIAL_SCRIPT_HASHES = {
    "sts2015-en-post/perl/find_pairs.pl": "9ea7fb786919b059579ffacdb7a2e6ce0caef4a8b4f42d8001a874d726359948",
    "sts2015-en-post/perl/get_gs.pl": "905605938c98dc705965fae47fb7021d8df033dddb45a6a21e4953c1cdd6b46f",
    "sts2015-en-post/perl/read_csv.pl": "ef3446b42b5142a293583aeb8d663f0e888813b245ca53c6638d397ed75fe5d2",
    "sts2015-en-post/perl/select_subset.pl": "7db6fa1975dae51c14f76ff8c9f4d69a7228c25e6ab1fb030e543587d5d22579",
}
OFFICIAL_CLEAN_PATH = "sts2015-en-post/data/clean/text.clean"
ALLOWED_LABELS = frozenset(range(6))
SPLITS = ("train", "dev", "test")
EXPECTED_SPLIT_ROWS = {"train": 4734, "dev": 1578, "test": 1578}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalized_sentence(text: str) -> str:
    return " ".join(text.lower().split())


def unique_mode(scores: tuple[int, ...]) -> int:
    counts = Counter(scores)
    maximum = max(counts.values())
    modes = sorted(label for label, count in counts.items() if count == maximum)
    return modes[0] if len(modes) == 1 else -1


def empirical_distribution(scores: tuple[int, ...]) -> list[float]:
    counts = Counter(scores)
    return [counts[label] / len(scores) for label in range(6)]


def quantized_mean_target(scores: tuple[int, ...]) -> int:
    return math.floor(sum(scores) / len(scores) + 0.5)


def require_all(gates: dict[str, bool], context: str) -> None:
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise RuntimeError(f"{context} failed closed: {', '.join(failed)}")


@dataclass(frozen=True)
class UnionFind:
    parent: dict[str, str]

    @classmethod
    def from_values(cls, values: set[str]) -> "UnionFind":
        return cls(parent={value: value for value in sorted(values)})

    def find(self, value: str) -> str:
        cursor = value
        while self.parent[cursor] != cursor:
            self.parent[cursor] = self.parent[self.parent[cursor]]
            cursor = self.parent[cursor]
        return cursor

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def parse_dataset_lines(lines: Iterable[str], expected_rows: int | None = None) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row_id, line in enumerate(lines):
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 6:
            raise RuntimeError(f"row {row_id} has {len(fields)} fields instead of 6")
        try:
            scores = tuple(int(value) for value in fields[3].split())
        except ValueError as exc:
            raise RuntimeError(f"row {row_id} has a non-integer published score") from exc
        if len(scores) != 5:
            raise RuntimeError(f"row {row_id} has {len(scores)} published scores instead of 5")
        if any(score not in ALLOWED_LABELS for score in scores):
            raise RuntimeError(f"row {row_id} has a published score outside [0, 5]")
        records.append(
            {
                "row_id": row_id,
                "gs_score": float(fields[0]),
                "n_annotators": int(fields[1]),
                "origin": fields[2],
                "scores": scores,
                "sentence1": fields[4],
                "sentence2": fields[5],
                "sentence1_normalized": normalized_sentence(fields[4]),
                "sentence2_normalized": normalized_sentence(fields[5]),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(f"dataset has {len(frame)} rows instead of {expected_rows}")
    if frame.empty or frame["row_id"].duplicated().any():
        raise RuntimeError("dataset row IDs are empty or duplicated")
    return frame


def load_dataset(path: Path, expected_rows: int = EXPECTED_ROWS) -> pd.DataFrame:
    return parse_dataset_lines(path.read_text(encoding="utf-8").splitlines(), expected_rows)


def verify_official_archive(path: Path, public_frame: pd.DataFrame) -> dict[str, Any]:
    archive_hash = sha256_bytes(path.read_bytes())
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(f"official archive hash mismatch: {archive_hash}")
    with zipfile.ZipFile(path) as archive:
        script_hashes = {
            name: sha256_bytes(archive.read(name)) for name in OFFICIAL_SCRIPT_HASHES
        }
        if script_hashes != OFFICIAL_SCRIPT_HASHES:
            raise RuntimeError("official SemEval Perl-script hash mismatch")
        find_pairs = archive.read("sts2015-en-post/perl/find_pairs.pl").decode("utf-8")
        behavior_markers = {
            "gold_mean_uses_all_collected_scores_before_truncation": (
                "my $score = $pair_score{$pair} / $pair_count{$pair};" in find_pairs
            ),
            "score_list_truncates_after_five_values": (
                "if (@ss > 5)" in find_pairs and "splice(@ss, 5);" in find_pairs
            ),
        }
        require_all(behavior_markers, "official source behavior")
        official_clean = parse_dataset_lines(
            archive.read(OFFICIAL_CLEAN_PATH).decode("utf-8").splitlines(),
            expected_rows=8331,
        )

    def pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
        return set(zip(frame["sentence1_normalized"], frame["sentence2_normalized"]))

    official_pairs, public_pairs = pairs(official_clean), pairs(public_frame)
    if not public_pairs.issubset(official_pairs):
        raise RuntimeError("public 7,890-row data is not a pair subset of official clean data")
    return {
        "archive_sha256": archive_hash,
        "script_sha256": script_hashes,
        "behavior_gates": behavior_markers,
        "official_clean_rows": int(len(official_clean)),
        "public_rows": int(len(public_frame)),
        "official_rows_not_in_public_subset": int(len(official_pairs - public_pairs)),
        "interpretation": (
            "The verified official script computes gs_score from all collected scores, "
            "then retains the first five encountered scores in the published score list."
        ),
    }


def add_agreement_strata(frame: pd.DataFrame, source_repo: Path) -> tuple[pd.DataFrame, list[float]]:
    sys.path.insert(0, str(source_repo))
    try:
        from preprocess.krippendorff import alpha_per_item
    finally:
        sys.path.pop(0)
    reliability = np.asarray(frame["scores"].tolist(), dtype=int).transpose()
    agreement, _ = alpha_per_item(reliability_data=reliability, level_of_measurement="ordinal")
    thresholds = np.percentile(agreement, [33.33, 66.67])
    output = frame.copy()
    output["agreement"] = agreement
    output["tercile"] = [
        1 if value <= thresholds[0] else 2 if value <= thresholds[1] else 3
        for value in agreement
    ]
    output["mode"] = output["scores"].map(unique_mode)
    output["stratum"] = output["mode"].astype(str) + "_" + output["tercile"].astype(str)
    if output["stratum"].nunique() != EXPECTED_STRATA:
        raise RuntimeError("unexpected number of observed agreement/mode strata")
    return output, [float(value) for value in thresholds]


def add_components(frame: pd.DataFrame) -> pd.DataFrame:
    sentences = set(frame["sentence1_normalized"]) | set(frame["sentence2_normalized"])
    union_find = UnionFind.from_values(sentences)
    for left, right in zip(frame["sentence1_normalized"], frame["sentence2_normalized"]):
        union_find.union(left, right)
    output = frame.copy()
    output["component"] = [union_find.find(value) for value in output["sentence1_normalized"]]
    return output


def upstream_item_split(frame: pd.DataFrame) -> pd.Series:
    train_dev, test = train_test_split(
        frame, test_size=0.20, stratify=frame["stratum"], random_state=42
    )
    train, dev = train_test_split(
        train_dev, test_size=0.25, stratify=train_dev["stratum"], random_state=42
    )
    assignment = pd.Series(index=frame.index, dtype="object")
    assignment.loc[train.index] = "train"
    assignment.loc[dev.index] = "dev"
    assignment.loc[test.index] = "test"
    return assignment


def cross_split_summary(frame: pd.DataFrame, assignment: pd.Series) -> dict[str, Any]:
    audit = frame.copy()
    audit["split"] = assignment
    pair_groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    sentence_groups: defaultdict[str, list[int]] = defaultdict(list)
    component_groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in audit.iterrows():
        pair = tuple(sorted((row["sentence1_normalized"], row["sentence2_normalized"])))
        pair_groups[pair].append(index)
        for sentence in {row["sentence1_normalized"], row["sentence2_normalized"]}:
            sentence_groups[sentence].append(index)
        component_groups[row["component"]].append(index)

    def summarize(groups: dict[Any, list[int]]) -> dict[str, int]:
        crossing = [
            indices
            for indices in groups.values()
            if len(set(audit.loc[indices, "split"])) > 1
        ]
        return {
            "cross_split_groups": len(crossing),
            "rows_in_cross_split_groups": len({index for group in crossing for index in group}),
        }

    return {
        "canonical_pair": summarize(pair_groups),
        "sentence_any_position": summarize(sentence_groups),
        "sentence_component": summarize(component_groups),
    }


def split_summary(frame: pd.DataFrame, assignment: pd.Series) -> dict[str, Any]:
    audit = frame.copy()
    audit["split"] = assignment
    overall_strata = audit["stratum"].value_counts(normalize=True)
    overall_origins = audit["origin"].value_counts(normalize=True)
    result: dict[str, Any] = {}
    for split_name in SPLITS:
        subset = audit[audit["split"] == split_name]
        strata = subset["stratum"].value_counts(normalize=True).reindex(overall_strata.index, fill_value=0)
        origins = subset["origin"].value_counts(normalize=True).reindex(overall_origins.index, fill_value=0)
        component_sizes = subset.groupby("component").size().sort_values(ascending=False)
        largest_component = str(component_sizes.index[0])
        result[split_name] = {
            "rows": int(len(subset)),
            "components": int(subset["component"].nunique()),
            "strata_present": int(subset["stratum"].nunique()),
            "maximum_absolute_stratum_proportion_deviation": float((strata - overall_strata).abs().max()),
            "maximum_absolute_origin_proportion_deviation": float((origins - overall_origins).abs().max()),
            "origins": {
                origin: {"rows": int(count), "proportion": float(count / len(subset))}
                for origin, count in sorted(Counter(subset["origin"]).items())
            },
            "largest_component_rows": int(component_sizes.iloc[0]),
            "largest_component_share": float(component_sizes.iloc[0] / len(subset)),
            "largest_component_sha256": sha256_bytes(largest_component.encode("utf-8")),
        }
    return result


def assignment_gates(frame: pd.DataFrame, assignment: pd.Series) -> dict[str, bool]:
    leakage = cross_split_summary(frame, assignment)
    summary = split_summary(frame, assignment)
    return {
        "all_rows_assigned": bool(assignment.notna().all() and len(assignment) == len(frame)),
        "exact_split_names": set(assignment) == set(SPLITS),
        "expected_split_sizes": all(
            abs(summary[name]["rows"] - EXPECTED_SPLIT_ROWS[name]) <= 2 for name in SPLITS
        ),
        "all_strata_in_every_split": all(
            summary[name]["strata_present"] == EXPECTED_STRATA for name in SPLITS
        ),
        "zero_pair_overlap": leakage["canonical_pair"]["cross_split_groups"] == 0,
        "zero_sentence_overlap": leakage["sentence_any_position"]["cross_split_groups"] == 0,
        "zero_component_overlap": leakage["sentence_component"]["cross_split_groups"] == 0,
        "dev_test_largest_component_share_at_most_0p05": max(
            summary[name]["largest_component_share"] for name in ("dev", "test")
        ) <= 0.05,
    }


def candidate_assignment(
    frame: pd.DataFrame,
    train_dev_positions: np.ndarray,
    test_positions: np.ndarray,
    inner_train_relative: np.ndarray,
    dev_relative: np.ndarray,
) -> pd.Series:
    train_dev = frame.iloc[train_dev_positions]
    assignment = pd.Series("test", index=frame.index, dtype="object")
    assignment.loc[train_dev.index[inner_train_relative]] = "train"
    assignment.loc[train_dev.index[dev_relative]] = "dev"
    assignment.loc[frame.index[test_positions]] = "test"
    return assignment


def select_component_disjoint_split(frame: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=61)
    candidates: list[tuple[tuple[float, float, float, int, int], pd.Series, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for outer_index, (train_dev_positions, test_positions) in enumerate(
        outer.split(frame, frame["stratum"], groups=frame["component"])
    ):
        train_dev = frame.iloc[train_dev_positions]
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=61)
        for inner_index, (train_relative, dev_relative) in enumerate(
            inner.split(train_dev, train_dev["stratum"], groups=train_dev["component"])
        ):
            assignment = candidate_assignment(
                frame,
                train_dev_positions,
                test_positions,
                train_relative,
                dev_relative,
            )
            summary = split_summary(frame, assignment)
            gates = assignment_gates(frame, assignment)
            record = {
                "outer_fold": outer_index,
                "inner_fold": inner_index,
                "gates": gates,
                "summary": summary,
            }
            if not all(gates.values()):
                rejected.append(record)
                continue
            ranking = (
                max(summary[name]["largest_component_share"] for name in ("dev", "test")),
                max(
                    summary[name]["maximum_absolute_origin_proportion_deviation"]
                    for name in ("dev", "test")
                ),
                max(summary[name]["maximum_absolute_stratum_proportion_deviation"] for name in SPLITS),
                outer_index,
                inner_index,
            )
            record["ranking_tuple"] = list(ranking)
            candidates.append((ranking, assignment, record))
    if not candidates:
        raise RuntimeError("no component-disjoint split candidate passed all frozen filters")
    candidates.sort(key=lambda item: item[0])
    _, selected, selected_record = candidates[0]
    require_all(assignment_gates(frame, selected), "selected split")
    return selected, {
        "generator": "enumerated two-stage StratifiedGroupKFold",
        "random_state": 61,
        "sklearn_version": sklearn.__version__,
        "candidate_count": 20,
        "eligible_candidate_count": len(candidates),
        "rejected_candidate_count": len(rejected),
        "selection_rule": (
            "filter integrity, full strata, size tolerance <=2, dev/test max component share <=5%; "
            "then lexicographically minimize max eval component share, max eval origin deviation, "
            "max stratum deviation, outer fold, inner fold"
        ),
        "selected": selected_record,
    }


def target_semantics(frame: pd.DataFrame, assignment: pd.Series) -> dict[str, Any]:
    audit = frame.copy()
    audit["split"] = assignment

    def summarize(subset: pd.DataFrame) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for scores in subset["scores"]:
            mean = sum(scores) / 5
            hard = quantized_mean_target(scores)
            mode = unique_mode(scores)
            distribution = empirical_distribution(scores)
            rows.append(
                {
                    "mean": mean,
                    "hard": hard,
                    "mode": mode,
                    "median": int(np.median(scores)),
                    "hard_in_support": hard in scores,
                    "score_span": max(scores) - min(scores),
                    "residual_nonzero": any(
                        abs(distribution[label] - (1.0 if label == hard else 0.0)) > 1e-12
                        for label in range(6)
                    ),
                }
            )
        unique_mode_rows = [row for row in rows if row["mode"] != -1]
        return {
            "rows": len(rows),
            "mode_tie_rows": sum(row["mode"] == -1 for row in rows),
            "unique_mode_rows": len(unique_mode_rows),
            "quantized_mean_differs_from_unique_mode_rows": sum(
                row["hard"] != row["mode"] for row in unique_mode_rows
            ),
            "quantized_mean_not_in_published_support_rows": sum(
                not row["hard_in_support"] for row in rows
            ),
            "quantized_mean_differs_from_median_rows": sum(
                row["hard"] != row["median"] for row in rows
            ),
            "residual_nonzero_rows": sum(row["residual_nonzero"] for row in rows),
            "hard_label_distribution": {
                str(key): value for key, value in sorted(Counter(row["hard"] for row in rows).items())
            },
            "human_mean_distribution": {
                f"{key:.1f}": value
                for key, value in sorted(Counter(row["mean"] for row in rows).items())
            },
            "score_span_distribution": {
                str(key): value
                for key, value in sorted(Counter(row["score_span"] for row in rows).items())
            },
        }

    return {
        "terminology": {
            "paper_name": "Quantized-Mean Main-Target Only",
            "legacy_code_identifier": "consensus_only",
            "forbidden_unqualified_names": ["majority label", "annotator consensus"],
            "residual_name": "residual relative to the quantized-mean main target",
        },
        "all": summarize(audit),
        "by_split": {name: summarize(audit[audit["split"] == name]) for name in SPLITS},
    }


def manifest_record(row: pd.Series) -> dict[str, Any]:
    pair_key = "\n".join((row["sentence1_normalized"], row["sentence2_normalized"]))
    scores = tuple(row["scores"])
    target_hash = sha256_bytes(stable_json(list(scores)).encode("utf-8"))
    return {
        "row_id": int(row["row_id"]),
        "pair_sha256": sha256_bytes(pair_key.encode("utf-8")),
        "target_sha256": target_hash,
        "component_sha256": sha256_bytes(row["component"].encode("utf-8")),
        "split": row["frozen_split"],
        "origin": row["origin"],
        "human_mean": sum(scores) / 5,
        "hard_label": quantized_mean_target(scores),
        "mode": int(row["mode"]),
        "agreement_tercile": int(row["tercile"]),
    }


def serialize_manifest(frame: pd.DataFrame) -> bytes:
    records = [manifest_record(row) for _, row in frame.sort_values("row_id").iterrows()]
    return "".join(stable_json(record) + "\n" for record in records).encode("utf-8")


def validate_manifest(payload: bytes, frame: pd.DataFrame) -> dict[str, bool]:
    records = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    row_ids = [int(record["row_id"]) for record in records]
    return {
        "row_count": len(records) == EXPECTED_ROWS,
        "row_ids_unique": len(row_ids) == len(set(row_ids)),
        "row_ids_complete": sorted(row_ids) == list(range(EXPECTED_ROWS)),
        "split_values_valid": {record["split"] for record in records} == set(SPLITS),
        "target_hash_present": all(len(record["target_sha256"]) == 64 for record in records),
        "human_mean_present": all(0.0 <= float(record["human_mean"]) <= 5.0 for record in records),
        "hard_label_present": all(int(record["hard_label"]) in ALLOWED_LABELS for record in records),
        "deterministic_replay": payload == serialize_manifest(frame),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    source_commit = subprocess.run(
        ["git", "-C", str(args.source_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != EXPECTED_COMMIT:
        raise RuntimeError(f"upstream commit mismatch: {source_commit}")
    dataset_path = args.source_repo / "data/text.clean"
    dataset_hash = sha256_bytes(dataset_path.read_bytes())
    if dataset_hash != EXPECTED_DATA_SHA256:
        raise RuntimeError(f"dataset hash mismatch: {dataset_hash}")

    frame = load_dataset(dataset_path)
    official_source = verify_official_archive(args.official_archive, frame)
    frame, agreement_thresholds = add_agreement_strata(frame, args.source_repo)
    frame = add_components(frame)
    upstream_assignment = upstream_item_split(frame)
    frozen_assignment, selection = select_component_disjoint_split(frame)
    replay_assignment, replay_selection = select_component_disjoint_split(frame)
    if not frozen_assignment.equals(replay_assignment) or selection != replay_selection:
        raise RuntimeError("component split is not deterministic under immediate replay")
    frame["frozen_split"] = frozen_assignment

    means = frame["scores"].map(lambda scores: sum(scores) / 5)
    gs_mismatches = np.abs(frame["gs_score"] - means) > 1e-12
    distributions = {tuple(empirical_distribution(scores)) for scores in frame["scores"]}
    unanimous = frame["scores"].map(lambda scores: len(set(scores)) == 1)
    largest_components = frame.groupby("component").size().sort_values(ascending=False)

    manifest_payload = serialize_manifest(frame)
    manifest_gates = validate_manifest(manifest_payload, frame)
    require_all(manifest_gates, "split manifest")
    manifest_path = args.output_root / "data/split_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_payload)

    frozen_gates = assignment_gates(frame, frozen_assignment)
    require_all(frozen_gates, "frozen component split")
    data_gates = {
        "upstream_commit_matches": source_commit == EXPECTED_COMMIT,
        "dataset_hash_matches": dataset_hash == EXPECTED_DATA_SHA256,
        "rows_exactly_7890": len(frame) == EXPECTED_ROWS,
        "row_ids_unique": not frame["row_id"].duplicated().any(),
        "five_scores_per_row": all(len(scores) == 5 for scores in frame["scores"]),
        "scores_are_integers_0_to_5": all(
            all(isinstance(score, int) and score in ALLOWED_LABELS for score in scores)
            for scores in frame["scores"]
        ),
        "twenty_observed_strata": frame["stratum"].nunique() == EXPECTED_STRATA,
        "official_source_closed": all(official_source["behavior_gates"].values()),
        "split_replay_identical": frozen_assignment.equals(replay_assignment),
        "manifest_replay_identical": manifest_gates["deterministic_replay"],
    }
    require_all(data_gates, "Stage 0 data")

    audit = {
        "experiment": "Exp61-SoftSTS15-External-Confirmation",
        "stage": 0,
        "upstream_commit": source_commit,
        "dataset_sha256": dataset_hash,
        "official_source_closure": official_source,
        "rows": int(len(frame)),
        "origins": dict(sorted(Counter(frame["origin"]).items())),
        "published_score_list_lengths": {"5": int(len(frame))},
        "n_annotators_field": {
            str(key): value for key, value in sorted(Counter(frame["n_annotators"]).items())
        },
        "gs_score_vs_published_five_mean": {
            "matches": int((~gs_mismatches).sum()),
            "mismatches": int(gs_mismatches.sum()),
            "maximum_absolute_difference": float(np.abs(frame["gs_score"] - means).max()),
            "protocol_resolution": "exclude gs_score; derive all targets from the same five published scores",
        },
        "target_patterns": {
            "unique_empirical_distributions": len(distributions),
            "unanimous_rows": int(unanimous.sum()),
            "non_unanimous_rows": int((~unanimous).sum()),
        },
        "target_semantics": target_semantics(frame, frozen_assignment),
        "agreement_tercile_thresholds": agreement_thresholds,
        "strata": int(frame["stratum"].nunique()),
        "sentences": int(len(set(frame["sentence1_normalized"]) | set(frame["sentence2_normalized"]))),
        "sentence_components": int(frame["component"].nunique()),
        "largest_component_rows": [int(value) for value in largest_components.head(20)],
        "upstream_item_split": {
            "summary": split_summary(frame, upstream_assignment),
            "leakage": cross_split_summary(frame, upstream_assignment),
            "accepted_for_exp61": False,
        },
        "frozen_component_split": {
            "selection": selection,
            "summary": split_summary(frame, frozen_assignment),
            "leakage": cross_split_summary(frame, frozen_assignment),
            "gates": frozen_gates,
            "accepted_for_exp61": True,
        },
        "data_gates": data_gates,
        "manifest_gates": manifest_gates,
        "split_manifest_sha256": sha256_bytes(manifest_payload),
        "model_training_performed": False,
        "gpu_used": False,
    }
    decision = {
        "status": "EXP61_REVISED_DATA_STAGE0_PASS_TOKENIZER_AUDIT_PENDING",
        "all_data_gates_pass": True,
        "formal_training_authorized": False,
        "next_required_artifact": "audit/tokenizer_length_audit.json",
    }
    write_json(args.output_root / "audit/soft_sts15_stage0_audit.json", audit)
    write_json(args.output_root / "decision/stage0_data_decision.json", decision)
    print(json.dumps({"audit": audit, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
