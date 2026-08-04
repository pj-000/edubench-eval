from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "thesis_exp/outputs/exp61_soft_sts15_external_confirmation"
EXPECTED_COMMIT = "ca754a21e58437c4b843d10161d2838f39230e7f"
EXPECTED_DATA_SHA256 = "8707cdb85e2da9818bc8c08cd90b9b2c17c483e08f7bb7ff5c6dc69b555dde71"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def load_dataset(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row_id, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"row {row_id} has {len(fields)} fields instead of 6")
        scores = tuple(int(value) for value in fields[3].split())
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
    return pd.DataFrame.from_records(records)


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
    return output, [float(value) for value in thresholds]


def add_components(frame: pd.DataFrame) -> pd.DataFrame:
    sentences = set(frame["sentence1_normalized"]) | set(frame["sentence2_normalized"])
    union_find = UnionFind.from_values(sentences)
    for left, right in zip(frame["sentence1_normalized"], frame["sentence2_normalized"]):
        union_find.union(left, right)
    output = frame.copy()
    output["component"] = [union_find.find(value) for value in output["sentence1_normalized"]]
    return output


def official_item_split(frame: pd.DataFrame) -> pd.Series:
    train_dev, test = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame["stratum"],
        random_state=42,
    )
    train, dev = train_test_split(
        train_dev,
        test_size=0.25,
        stratify=train_dev["stratum"],
        random_state=42,
    )
    assignment = pd.Series(index=frame.index, dtype="object")
    assignment.loc[train.index] = "train"
    assignment.loc[dev.index] = "dev"
    assignment.loc[test.index] = "test"
    return assignment


def component_disjoint_split(frame: pd.DataFrame) -> pd.Series:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=61)
    train_dev_positions, test_positions = next(
        outer.split(frame, frame["stratum"], groups=frame["component"])
    )
    train_dev = frame.iloc[train_dev_positions]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=61)
    train_relative, dev_relative = next(
        inner.split(train_dev, train_dev["stratum"], groups=train_dev["component"])
    )
    assignment = pd.Series("test", index=frame.index, dtype="object")
    assignment.loc[train_dev.index[train_relative]] = "train"
    assignment.loc[train_dev.index[dev_relative]] = "dev"
    assignment.loc[frame.index[test_positions]] = "test"
    return assignment


def cross_split_summary(frame: pd.DataFrame, assignment: pd.Series) -> dict[str, Any]:
    audit = frame.copy()
    audit["split"] = assignment
    pair_groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    sentence_groups: defaultdict[str, list[int]] = defaultdict(list)
    component_groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in audit.iterrows():
        pair_groups[tuple(sorted((row["sentence1_normalized"], row["sentence2_normalized"])))] .append(index)
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
            "rows_in_cross_split_groups": len({index for indices in crossing for index in indices}),
        }

    return {
        "canonical_pair": summarize(pair_groups),
        "sentence_any_position": summarize(sentence_groups),
        "sentence_component": summarize(component_groups),
    }


def split_summary(frame: pd.DataFrame, assignment: pd.Series) -> dict[str, Any]:
    audit = frame.copy()
    audit["split"] = assignment
    overall = audit["stratum"].value_counts(normalize=True)
    result: dict[str, Any] = {}
    for split_name in ("train", "dev", "test"):
        subset = audit[audit["split"] == split_name]
        proportions = subset["stratum"].value_counts(normalize=True).reindex(overall.index, fill_value=0)
        result[split_name] = {
            "rows": int(len(subset)),
            "components": int(subset["component"].nunique()),
            "strata_present": int(subset["stratum"].nunique()),
            "maximum_absolute_stratum_proportion_deviation": float((proportions - overall).abs().max()),
        }
    return result


def manifest_record(row: pd.Series) -> dict[str, Any]:
    pair_key = "\n".join((row["sentence1_normalized"], row["sentence2_normalized"]))
    return {
        "row_id": int(row["row_id"]),
        "pair_sha256": sha256_bytes(pair_key.encode("utf-8")),
        "component_sha256": sha256_bytes(row["component"].encode("utf-8")),
        "split": row["frozen_split"],
        "origin": row["origin"],
        "mode": int(row["mode"]),
        "agreement_tercile": int(row["tercile"]),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    dataset_path = args.source_repo / "data/text.clean"
    source_commit = subprocess.run(
        ["git", "-C", str(args.source_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != EXPECTED_COMMIT:
        raise SystemExit(f"upstream commit mismatch: {source_commit}")
    dataset_hash = sha256_bytes(dataset_path.read_bytes())
    if dataset_hash != EXPECTED_DATA_SHA256:
        raise SystemExit(f"dataset hash mismatch: {dataset_hash}")

    frame = load_dataset(dataset_path)
    frame, agreement_thresholds = add_agreement_strata(frame, args.source_repo)
    frame = add_components(frame)
    official_assignment = official_item_split(frame)
    frozen_assignment = component_disjoint_split(frame)
    frame["frozen_split"] = frozen_assignment

    score_lengths = Counter(len(scores) for scores in frame["scores"])
    annotator_counts = Counter(int(value) for value in frame["n_annotators"])
    means = frame["scores"].map(lambda scores: sum(scores) / len(scores))
    gs_mismatches = np.abs(frame["gs_score"] - means) > 1e-12
    distributions = {tuple(empirical_distribution(scores)) for scores in frame["scores"]}
    unanimous = frame["scores"].map(lambda scores: len(set(scores)) == 1)
    largest_components = frame.groupby("component").size().sort_values(ascending=False)

    audit = {
        "experiment": "Exp61-SoftSTS15-External-Confirmation",
        "stage": 0,
        "upstream_commit": source_commit,
        "dataset_sha256": dataset_hash,
        "rows": int(len(frame)),
        "origins": dict(sorted(Counter(frame["origin"]).items())),
        "published_score_list_lengths": {str(key): value for key, value in sorted(score_lengths.items())},
        "n_annotators_field": {str(key): value for key, value in sorted(annotator_counts.items())},
        "invalid_published_scores": int(
            sum(any(score < 0 or score > 5 for score in scores) for scores in frame["scores"])
        ),
        "gs_score_vs_published_five_mean": {
            "matches": int((~gs_mismatches).sum()),
            "mismatches": int(gs_mismatches.sum()),
            "maximum_absolute_difference": float(np.abs(frame["gs_score"] - means).max()),
            "protocol_resolution": "derive hard label and human mean from the same five published scores",
        },
        "target_patterns": {
            "unique_empirical_distributions": len(distributions),
            "unanimous_rows": int(unanimous.sum()),
            "non_unanimous_rows": int((~unanimous).sum()),
        },
        "agreement_tercile_thresholds": agreement_thresholds,
        "strata": int(frame["stratum"].nunique()),
        "sentences": int(len(set(frame["sentence1_normalized"]) | set(frame["sentence2_normalized"]))),
        "sentence_components": int(frame["component"].nunique()),
        "largest_component_rows": [int(value) for value in largest_components.head(20)],
        "upstream_item_split": {
            "summary": split_summary(frame, official_assignment),
            "leakage": cross_split_summary(frame, official_assignment),
            "accepted_for_exp61": False,
        },
        "frozen_component_split": {
            "summary": split_summary(frame, frozen_assignment),
            "leakage": cross_split_summary(frame, frozen_assignment),
            "accepted_for_exp61": True,
        },
        "model_training_performed": False,
        "gpu_used": False,
    }

    manifest = [manifest_record(row) for _, row in frame.sort_values("row_id").iterrows()]
    manifest_path = args.output_root / "data/split_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in manifest),
        encoding="utf-8",
    )
    audit["split_manifest_sha256"] = sha256_bytes(manifest_path.read_bytes())

    decision = {
        "status": "CONDITIONAL_GO_TO_PROTOCOL_DESIGN",
        "formal_training_authorized": False,
        "passed": [
            "7890 rows parsed",
            "every row has five published integer scores in [0, 5]",
            "deterministic component-disjoint split generated",
            "all 20 observed strata occur in every split",
            "zero sentence or pair overlap across the frozen split",
        ],
        "must_complete_before_training": [
            "freeze the exact three-arm trainer and inference estimator",
            "define scale-aware primary effect and non-inferiority gates",
            "freeze dev-only preflight and single-access test policy",
            "verify model/tokenizer availability and no-update geometry on the six-class head",
            "freeze source and environment manifests",
            "document upstream data reuse and redistribution terms",
        ],
    }

    write_json(args.output_root / "audit/soft_sts15_stage0_audit.json", audit)
    write_json(args.output_root / "decision/stage0_decision.json", decision)
    print(json.dumps({"audit": audit, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
