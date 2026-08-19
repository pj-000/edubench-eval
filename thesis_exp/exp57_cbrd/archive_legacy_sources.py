"""Materialize an exact, read-only legacy HMSA source snapshot under Exp57.

The current branch intentionally does not contain ``exp49_cphce``,
``exp50_cahs``, or ``exp51_hmsa`` source trees.  This script obtains the
locked text blobs from immutable Git commits and writes them only below the
new CBRD experiment directory.  It never touches legacy experiment paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd import LEGACY_HMSA_SOURCE_COMMIT, LEGACY_RESULT_COMMIT, REPO_ROOT


SOURCE_LOCK_RELATIVE = "thesis_exp/configs/exp51_hmsa/source_lock.json"
FORMAL_LOCK_RELATIVE = "thesis_exp/configs/exp51_hmsa/formal_lock.json"
ARCHIVE_ROOT = REPO_ROOT / "thesis_exp" / "exp57_cbrd" / "legacy_source"
LOCK_ARCHIVE_ROOT = ARCHIVE_ROOT / "locks"


def git_blob(revision: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def lock_payload(revision: str, relative: str, archive_name: str) -> bytes:
    """Read a lock from Git, or from the verified offline archive fallback."""

    try:
        return git_blob(revision, relative)
    except subprocess.CalledProcessError:
        fallback = LOCK_ARCHIVE_ROOT / archive_name
        if not fallback.is_file():
            raise RuntimeError(
                f"Neither Git object {revision}:{relative} nor offline lock {fallback} is available"
            )
        return fallback.read_bytes()


def source_payload(revision: str, relative: str, archive_root: Path) -> bytes:
    """Read a source blob from Git, falling back to its archived destination."""

    try:
        return git_blob(revision, relative)
    except subprocess.CalledProcessError:
        fallback = archive_root / relative
        if not fallback.is_file():
            raise RuntimeError(
                f"Neither Git object {revision}:{relative} nor archived source {fallback} is available"
            )
        return fallback.read_bytes()


def write_blob(root: Path, relative: str, payload: bytes) -> dict[str, Any]:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise RuntimeError(f"Refusing to overwrite a mismatched archived source: {destination}")
    destination.write_bytes(payload)
    return {
        "path": str(destination.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def materialize(*, check_only: bool = False) -> dict[str, Any]:
    source_lock_bytes = lock_payload(
        LEGACY_RESULT_COMMIT,
        SOURCE_LOCK_RELATIVE,
        "source_lock.json",
    )
    formal_lock_bytes = lock_payload(
        LEGACY_RESULT_COMMIT,
        FORMAL_LOCK_RELATIVE,
        "formal_lock.json",
    )
    source_lock = json.loads(source_lock_bytes)
    formal_lock = json.loads(formal_lock_bytes)
    if not check_only:
        write_blob(LOCK_ARCHIVE_ROOT, "source_lock.json", source_lock_bytes)
        write_blob(LOCK_ARCHIVE_ROOT, "formal_lock.json", formal_lock_bytes)
    targets: list[tuple[str, str, str, str]] = []
    for relative, expected in sorted(source_lock["files"].items()):
        targets.append(("hmsa_source", LEGACY_HMSA_SOURCE_COMMIT, relative, expected))
    for relative, expected in sorted(formal_lock["files"].items()):
        # Data splits are already hash-locked in their canonical location and
        # are deliberately not duplicated into a source archive.
        if relative.endswith(".jsonl"):
            continue
        targets.append(("formal_snapshot", LEGACY_RESULT_COMMIT, relative, expected))
    files: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for scope, revision, relative, expected in targets:
        key = (scope, relative)
        if key in seen:
            continue
        seen.add(key)
        archive_root = ARCHIVE_ROOT / scope
        payload = source_payload(revision, relative, archive_root)
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Legacy source mismatch for {scope}:{relative}: {actual} != {expected}"
            )
        if check_only:
            destination = archive_root / relative
            files.append({
                "path": str(destination.relative_to(REPO_ROOT)),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "present_and_matching": destination.is_file() and destination.read_bytes() == payload,
            })
        else:
            files.append(write_blob(archive_root, relative, payload))
    report = {
        "status": "PASS" if all(item.get("present_and_matching", True) for item in files) else "FAIL",
        "legacy_hmsa_source_commit": LEGACY_HMSA_SOURCE_COMMIT,
        "legacy_result_commit": LEGACY_RESULT_COMMIT,
        "archive_root": str(ARCHIVE_ROOT.relative_to(REPO_ROOT)),
        "files": files,
        "test_access_count": 0,
    }
    report_path = REPO_ROOT / "thesis_exp" / "outputs" / "exp57_cbrd" / "audit" / "legacy_source_archive.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(materialize(check_only=args.check_only), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
