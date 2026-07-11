"""Materialize the paper test only after the frozen Exp28 lock authorizes it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.exp02.build_exp02_dataset import convert_row
from thesis_exp.src.edujudge.utils.io import read_jsonl, write_jsonl


DEFAULT_LOCK = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28f_paper_dev_statistical_lock/"
    "configs/exp28f_final_test_lock.json"
)
DEFAULT_TEST = Path("thesis_exp/data/splits/paper_like_triple_seed42/test.jsonl")
DEFAULT_DATA_ROOT = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28e_paper_ce_training_variants_seed42/private/datasets"
)
DEFAULT_OUT_DIR = Path(
    "thesis_exp/exp17_low_score_evidence/outputs/exp28g_one_shot_final_test"
)
VARIANTS = ("b0_original_human", "b2_selective_dual_teacher")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if not args.lock.exists():
        raise FileNotFoundError(args.lock)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("status") != "READY_FOR_ONE_SHOT_FINAL_TEST" or not lock.get("test_open_authorized"):
        raise ValueError("Final lock does not authorize test access")
    if tuple(lock.get("locked_variants") or []) != VARIANTS:
        raise ValueError("Unexpected locked test variants")
    state_path = args.out_dir / "decision" / "exp28g_test_access_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "TEST_MATERIALIZED_ONCE":
            for variant in VARIANTS:
                path = args.data_root / variant / "test.jsonl"
                if not path.exists() or sha256(path) != state["test_dataset_sha256"]:
                    raise ValueError("Previously materialized test dataset changed")
            state["resume_without_source_test_read"] = True
            print(json.dumps(state, ensure_ascii=False, sort_keys=True))
            return state
        if state.get("status") == "TEST_ACCESS_STARTED":
            paths = [args.data_root / variant / "test.jsonl" for variant in VARIANTS]
            if all(path.exists() for path in paths):
                hashes = {sha256(path) for path in paths}
                counts = {count_jsonl(path) for path in paths}
                if len(hashes) == 1 and counts == {2218}:
                    state.update(
                        {
                            "status": "TEST_MATERIALIZED_ONCE",
                            "test_rows": 2218,
                            "test_dataset_sha256": next(iter(hashes)),
                            "resume_without_source_test_read": True,
                        }
                    )
                    write_json(state_path, state)
                    print(json.dumps(state, ensure_ascii=False, sort_keys=True))
                    return state
            raise RuntimeError(
                "A prior one-shot test access started but did not leave two complete identical datasets; "
                "the source test will not be read again."
            )
        raise ValueError("Existing test access state is not resumable")

    write_json(
        state_path,
        {
            "status": "TEST_ACCESS_STARTED",
            "test_access_count": 1,
            "locked_variants": list(VARIANTS),
            "labels_used_for_selection": False,
            "resume_without_source_test_read": False,
        },
    )
    source_rows = read_jsonl(args.test)
    if len(source_rows) != 2218:
        raise ValueError(f"Expected 2218 paper test rows, found {len(source_rows)}")
    converted = [
        convert_row(row, "test", index)
        for index, row in enumerate(source_rows)
    ]
    hashes = set()
    for variant in VARIANTS:
        path = args.data_root / variant / "test.jsonl"
        write_jsonl(path, converted)
        hashes.add(sha256(path))
    if len(hashes) != 1:
        raise ValueError("B0/B2 test datasets are not identical")
    state = {
        "status": "TEST_MATERIALIZED_ONCE",
        "test_access_count": 1,
        "test_rows": len(converted),
        "test_dataset_sha256": next(iter(hashes)),
        "locked_variants": list(VARIANTS),
        "labels_used_for_selection": False,
        "resume_without_source_test_read": False,
    }
    write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, sort_keys=True))
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
