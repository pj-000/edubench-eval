"""Verify that the historical Exp51 source closure is restored exactly."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp60_geometry_matched_shuffle import OUTPUT_ROOT, REPO_ROOT


HISTORICAL_COMMIT = "fa72bd4"
SOURCE_ROOTS = (
    "thesis_exp/exp49_cphce",
    "thesis_exp/exp50_cahs",
    "thesis_exp/exp51_hmsa",
    "thesis_exp/configs/exp49_cphce",
    "thesis_exp/configs/exp50_cahs",
    "thesis_exp/configs/exp51_hmsa",
)
OUTPUT_PATH = OUTPUT_ROOT / "audit" / "exp51_source_closure.json"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def run() -> dict[str, Any]:
    files = _git(
        "ls-tree", "-r", "--name-only", HISTORICAL_COMMIT, *SOURCE_ROOTS
    ).splitlines()
    rows: list[dict[str, Any]] = []
    for relative in files:
        path = REPO_ROOT / relative
        expected_blob = _git("rev-parse", f"{HISTORICAL_COMMIT}:{relative}")
        actual_blob = _git("hash-object", relative) if path.is_file() else None
        rows.append(
            {
                "path": relative,
                "expected_blob": expected_blob,
                "actual_blob": actual_blob,
                "exact_match": actual_blob == expected_blob,
            }
        )
    manifest = "\n".join(
        f"{row['path']}\t{row['actual_blob']}" for row in rows
    ).encode("utf-8")
    checks = {
        "all_historical_exp49_to_exp51_files_present": len(rows) == 43
        and all(row["actual_blob"] is not None for row in rows),
        "all_historical_exp49_to_exp51_blobs_exact": all(
            row["exact_match"] for row in rows
        ),
        "no_test_access": True,
    }
    report = {
        "status": "EXP49_TO_EXP51_SOURCE_CLOSURE_RESTORED"
        if all(checks.values())
        else "EXP49_TO_EXP51_SOURCE_CLOSURE_INCOMPLETE",
        "historical_commit": HISTORICAL_COMMIT,
        "source_roots": list(SOURCE_ROOTS),
        "files": rows,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "checks": checks,
        "test_access_count": 0,
    }
    write_json(OUTPUT_PATH, report)
    return report


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
