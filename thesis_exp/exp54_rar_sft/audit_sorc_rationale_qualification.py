"""Independently audit the train-only SORC rationale qualification package."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import (
    load_train_rows,
    read_jsonl,
    sha256_file,
)


SELECTOR_SEED = "exp54-sorc-rationale-qualification-v1|20260729"
SCHEMA_VERSION = "exp54-sorc-rationale-qualification-v1"
QUOTAS = {"L12": 24, "L3": 32, "L45": 64}
PAIR_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/preference_pairs"
)
PAIR_PATH = PAIR_ROOT / "private/rationale_pairs_r3_r2.jsonl"
PAIR_LOCK_PATH = PAIR_ROOT / "candidate_lock.json"
ROOT = PAIR_ROOT / "rationale_qualification"
PRIVATE = ROOT / "private"
PUBLIC_REPORT = ROOT / "candidate_report.json"
PUBLIC_LOCK = ROOT / "candidate_lock.json"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(*values: Any) -> str:
    return hashlib.sha256(canonical(list(values))).hexdigest()


def band(label: int) -> str:
    if label <= 2:
        return "L12"
    if label == 3:
        return "L3"
    return "L45"


def rank(row: dict[str, Any]) -> tuple[str, str]:
    return (
        digest(SELECTOR_SEED, row["pair_hash"], row["record_id"]),
        str(row["record_id"]),
    )


def independently_select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for target_band, quota in QUOTAS.items():
        remaining = [
            row for row in rows if band(int(row["gold_label"])) == target_band
        ]
        chosen: list[dict[str, Any]] = []
        while len(chosen) < quota:
            if not remaining:
                raise ValueError("independent selector cannot meet quota")
            current = selected + chosen
            metric = Counter(str(row["metric_id"]) for row in current)
            language = Counter(str(row["language"]) for row in current)
            joint = Counter(
                (str(row["metric_id"]), str(row["language"]))
                for row in current
            )
            item = min(
                remaining,
                key=lambda row: (
                    joint[(str(row["metric_id"]), str(row["language"]))],
                    metric[str(row["metric_id"])],
                    language[str(row["language"])],
                    *rank(row),
                ),
            )
            chosen.append(item)
            remaining.remove(item)
        selected.extend(chosen)
    return sorted(selected, key=rank)


def assert_no_evaluation_path(path: Path) -> None:
    normalized = "/" + path.resolve().as_posix().lower().strip("/") + "/"
    if "/dev." in normalized or "/test." in normalized:
        raise ValueError("qualification auditor forbids dev/test")


def expected_task(
    *,
    source: dict[str, Any],
    pair: dict[str, Any],
    pair_id: str,
    orientation: int,
    visible: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    swap_first = int(digest(SELECTOR_SEED, pair_id)[:2], 16) % 2
    r3_is_a = bool(swap_first ^ orientation)
    presentation_id = digest(
        SCHEMA_VERSION,
        pair_id,
        orientation,
    )
    r3 = str(pair["chosen"]["rationale"])
    r2 = str(pair["rejected"]["rationale"])
    candidate_a: dict[str, Any] = {"rationale": r3 if r3_is_a else r2}
    candidate_b: dict[str, Any] = {"rationale": r2 if r3_is_a else r3}
    if visible:
        candidate_a["score"] = int(pair["gold_label"])
        candidate_b["score"] = int(pair["gold_label"])
    task = {
        "presentation_id": presentation_id,
        "question": source["question"],
        "answer": source["answer"],
        "metric_id": str(pair["metric_id"]),
        "metric": str(
            source.get("metric_canonical")
            or source.get("metric_raw")
            or source["metric_id"]
        ),
        "rubric": source["rubric"],
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
    }
    key = {
        "presentation_id": presentation_id,
        "pair_id": pair_id,
        "orientation_index": orientation,
        "a_source": "R3_ALIGNED" if r3_is_a else "R2_SHUFFLED",
        "b_source": "R2_SHUFFLED" if r3_is_a else "R3_ALIGNED",
    }
    return task, key


def audit() -> dict[str, Any]:
    for path in (
        PAIR_PATH,
        PAIR_LOCK_PATH,
        PUBLIC_REPORT,
        PUBLIC_LOCK,
        PRIVATE / "sample_manifest.jsonl",
        PRIVATE / "answer_key.jsonl",
        PRIVATE / "score_blind_tasks.jsonl",
        PRIVATE / "score_visible_tasks.jsonl",
    ):
        assert_no_evaluation_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)

    pair_lock = json.loads(PAIR_LOCK_PATH.read_text(encoding="utf-8"))
    expected_pair_sha = str(
        pair_lock["private_output_hashes"]["rationale_pairs_r3_r2"]
    )
    if sha256_file(PAIR_PATH) != expected_pair_sha:
        raise ValueError("source rationale pair hash differs")
    pairs = read_jsonl(PAIR_PATH)
    selected = independently_select(pairs)
    train = load_train_rows()
    train_by_id = {str(row["record_id"]): row for row in train}
    samples = read_jsonl(PRIVATE / "sample_manifest.jsonl")
    keys = read_jsonl(PRIVATE / "answer_key.jsonl")
    blind = read_jsonl(PRIVATE / "score_blind_tasks.jsonl")
    visible = read_jsonl(PRIVATE / "score_visible_tasks.jsonl")

    expected_samples: list[dict[str, Any]] = []
    expected_keys: list[dict[str, Any]] = []
    expected_blind: list[dict[str, Any]] = []
    expected_visible: list[dict[str, Any]] = []
    for pair in selected:
        record_id = str(pair["record_id"])
        source = train_by_id[record_id]
        if (
            str(source["metric_id"]) != str(pair["metric_id"])
            or str(source["language"]) != str(pair["language"])
            or int(source["label_5"]) != int(pair["gold_label"])
        ):
            raise ValueError("selected pair metadata differs from train")
        pair_id = digest(
            SCHEMA_VERSION,
            pair["pair_hash"],
            record_id,
        )
        expected_samples.append(
            {
                "pair_id": pair_id,
                "record_id": record_id,
                "source_pair_hash": str(pair["pair_hash"]),
                "gold_label": int(pair["gold_label"]),
                "metric_id": str(pair["metric_id"]),
                "language": str(pair["language"]),
                "question": source["question"],
                "answer": source["answer"],
                "metric": str(
                    source.get("metric_canonical")
                    or source.get("metric_raw")
                    or source["metric_id"]
                ),
                "rubric": source["rubric"],
                "r3_aligned_rationale": str(pair["chosen"]["rationale"]),
                "r2_shuffled_rationale": str(pair["rejected"]["rationale"]),
            }
        )
        for orientation in (0, 1):
            blind_task, key = expected_task(
                source=source,
                pair=pair,
                pair_id=pair_id,
                orientation=orientation,
                visible=False,
            )
            visible_task, visible_key = expected_task(
                source=source,
                pair=pair,
                pair_id=pair_id,
                orientation=orientation,
                visible=True,
            )
            if key != visible_key:
                raise AssertionError("auditor constructed inconsistent keys")
            expected_keys.append(key)
            expected_blind.append(blind_task)
            expected_visible.append(visible_task)

    if samples != expected_samples:
        raise ValueError("sample manifest differs from independent derivation")
    if keys != expected_keys:
        raise ValueError("answer key differs from independent derivation")
    if blind != expected_blind:
        raise ValueError("score-blind tasks differ from derivation")
    if visible != expected_visible:
        raise ValueError("score-visible tasks differ from derivation")

    report = json.loads(PUBLIC_REPORT.read_text(encoding="utf-8"))
    lock = json.loads(PUBLIC_LOCK.read_text(encoding="utf-8"))
    private_paths = {
        "sample_manifest": PRIVATE / "sample_manifest.jsonl",
        "answer_key": PRIVATE / "answer_key.jsonl",
        "score_blind_tasks": PRIVATE / "score_blind_tasks.jsonl",
        "score_visible_tasks": PRIVATE / "score_visible_tasks.jsonl",
    }
    for name, path in private_paths.items():
        item = report["private_output_hashes"][name]
        if (
            int(item["rows"]) != len(read_jsonl(path))
            or int(item["bytes"]) != path.stat().st_size
            or str(item["sha256"]) != sha256_file(path)
        ):
            raise ValueError(f"private artifact lock differs: {name}")
        if lock["private_output_hashes"][name] != item:
            raise ValueError(f"public lock differs: {name}")
    if str(lock["candidate_report_sha256"]) != sha256_file(PUBLIC_REPORT):
        raise ValueError("candidate report hash differs")
    for value in (report, lock):
        if (
            value.get("preference_training_allowed") is not False
            or value.get("dev_accessed") is not False
            or value.get("test_accessed") is not False
        ):
            raise ValueError("candidate boundary is not fail-closed")

    return {
        "schema_version": f"{SCHEMA_VERSION}-audit-report",
        "status": "SORC_RATIONALE_QUALIFICATION_CANDIDATE_AUDIT_PASS",
        "aggregate": {
            "selected_pairs": len(samples),
            "score_blind_presentations": len(blind),
            "score_visible_presentations": len(visible),
            "metric_count": len({row["metric_id"] for row in samples}),
            "language_counts": dict(
                sorted(Counter(row["language"] for row in samples).items())
            ),
            "label_band_counts": dict(
                sorted(
                    Counter(
                        band(int(row["gold_label"])) for row in samples
                    ).items()
                )
            ),
        },
        "source_hashes": {
            "auditor_source": sha256_file(Path(__file__)),
            "rationale_pairs": expected_pair_sha,
            "candidate_report": sha256_file(PUBLIC_REPORT),
            "candidate_lock": sha256_file(PUBLIC_LOCK),
        },
        "selection_independently_rederived": True,
        "task_bytes_independently_rederived": True,
        "rationale_blind_qualification_completed": False,
        "p3_preference_training_allowed": False,
        "preference_training_allowed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Optional public aggregate audit-report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit()
    if args.write_report is not None:
        payload = (
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(payload, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
