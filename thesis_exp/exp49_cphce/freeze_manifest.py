"""Freeze the formal Exp49 dev decision before any new test inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from thesis_exp.exp49_cphce import CONFIG_ROOT, FORMAL_SEEDS, OUTPUT_ROOT, REPO_ROOT, VARIANTS, checkpoint_dir, split_path


MANIFEST_PATH = OUTPUT_ROOT / "freeze" / "frozen_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for path in paths:
        files.extend(path.rglob("*") if path.is_dir() else [path])
    for path in sorted(file for file in files if file.is_file()):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def freeze() -> dict[str, Any]:
    decision_path = OUTPUT_ROOT / "decision" / "formal_decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "FORMAL_PASS":
        raise PermissionError(f"Cannot freeze without FORMAL_PASS: {decision.get('status')}")
    checkpoints: dict[str, Any] = {}
    for variant in VARIANTS:
        for seed in FORMAL_SEEDS:
            root = checkpoint_dir(variant, seed)
            if not root.exists():
                raise FileNotFoundError(root)
            metadata_path = root / "exp49_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            checkpoints[f"{variant}/seed_{seed}"] = {
                "path": str(root),
                "sha256": sha256_tree([root]),
                "selected_epoch": metadata["epoch"],
            }
    sources = [REPO_ROOT / "thesis_exp" / "exp49_cphce", CONFIG_ROOT]
    manifest = {
        "status": "FORMAL_PASS_TEST_NOT_RUN",
        "git_commit": git_head(),
        "exp49_source_tree_sha256": sha256_tree(sources),
        "split_hashes": {split: sha256_file(split_path(split)) for split in ("train", "dev", "test")},
        "metric_contract_sha256": sha256_file(REPO_ROOT / "thesis_exp" / "exp49_cphce" / "metric_contract.py"),
        "config_tree_sha256": sha256_tree([CONFIG_ROOT]),
        "formal_decision_sha256": sha256_file(decision_path),
        "formal_dev_status": decision["status"],
        "checkpoints": checkpoints,
        "exp49_test_access_count": 0,
        "legacy_test_previously_accessed": True,
        "legacy_test_note": "Earlier project experiments used the same test split; this counter is Exp49-specific, not global.",
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def begin_test() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if int(manifest.get("exp49_test_access_count", -1)) != 0:
        raise PermissionError("Exp49 test has already been accessed")
    manifest["exp49_test_access_count"] = 1
    manifest["status"] = "TEST_IN_PROGRESS"
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def mark_tested() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if int(manifest.get("exp49_test_access_count", -1)) != 1 or manifest.get("status") != "TEST_IN_PROGRESS":
        raise PermissionError("Exp49 test is not in the one-shot in-progress state")
    manifest["status"] = "FORMAL_PASS_AND_TESTED"
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin-test", action="store_true")
    parser.add_argument("--mark-tested", action="store_true")
    args = parser.parse_args()
    if args.begin_test and args.mark_tested:
        raise ValueError("Choose only one transition")
    value = begin_test() if args.begin_test else (mark_tested() if args.mark_tested else freeze())
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
