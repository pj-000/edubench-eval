"""Validate and optionally sync only public lightweight Exp43 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

from thesis_exp.exp43_rubimor.common import ROOT, write_json

PUBLIC_LAYOUT = {
    "configs": {".json"},
    "tables": {".csv"},
    "reports": {".md"},
    "decision": {".json"},
    "hashes": {".json"},
    "state": {".json"},
}
MAX_BYTES = 10 * 1024 * 1024
PROHIBITED_SUFFIXES = {
    ".bin", ".ckpt", ".jsonl", ".log", ".npy", ".npz", ".pt",
    ".pth", ".safetensors",
}
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{16,}"),
)
PRIVATE_CSV_COLUMNS = {"sample_id", "question_key", "question", "answer"}


def public_files(out_dir: Path) -> list[Path]:
    files: list[Path] = []
    for directory in PUBLIC_LAYOUT:
        root = out_dir / directory
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(files)


def staged_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return [repo_root / name.decode("utf-8") for name in result.stdout.split(b"\0") if name]


def validate_file(path: Path, allowed_suffixes: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    if path.is_symlink():
        return [f"symlink is not allowed: {path}"]
    size = path.stat().st_size
    if size > MAX_BYTES:
        errors.append(f"file exceeds 10 MB: {path} ({size} bytes)")
    suffix = path.suffix.lower()
    if allowed_suffixes is not None and suffix not in allowed_suffixes:
        errors.append(f"extension is not allowed in public directory: {path}")
    if suffix in PROHIBITED_SUFFIXES:
        errors.append(f"prohibited artifact staged/public: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return errors + [f"non-text artifact staged/public: {path}"]
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            errors.append(f"possible API key or secret detected: {path}")
            break
    try:
        if suffix == ".json":
            json.loads(text)
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None) or []
                private_columns = PRIVATE_CSV_COLUMNS.intersection(header)
                if private_columns:
                    errors.append(
                        f"private columns in public CSV {path}: {sorted(private_columns)}"
                    )
                for _ in reader:
                    pass
    except (csv.Error, json.JSONDecodeError) as exc:
        errors.append(f"invalid {suffix[1:].upper()} file {path}: {exc}")
    return errors


def validate_public(out_dir: Path) -> tuple[list[Path], list[str]]:
    files = public_files(out_dir)
    errors: list[str] = []
    for path in files:
        relative = path.relative_to(out_dir)
        allowed = PUBLIC_LAYOUT.get(relative.parts[0])
        errors.extend(validate_file(path, allowed))
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(out_dir)
        if relative.parts[0] in PUBLIC_LAYOUT or relative.parts[0] in {"private", "logs_private"}:
            continue
        errors.append(f"file is outside the public/private layout: {path}")
    return files, errors


def staged_allowed(path: Path, repo_root: Path, out_dir: Path) -> bool:
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return False
    if path.is_relative_to(out_dir):
        output_relative = path.relative_to(out_dir)
        return bool(output_relative.parts) and output_relative.parts[0] in PUBLIC_LAYOUT
    if len(relative.parts) == 3 and relative.parts[:2] == ("thesis_exp", "exp43_rubimor"):
        # Exp43 Python modules live directly below thesis_exp/exp43_rubimor.
        return path.suffix == ".py"
    return (
        len(relative.parts) == 3
        and relative.parts[:2] == ("thesis_exp", "scripts")
        and relative.name.startswith("run_exp43")
        and path.suffix == ".sh"
    )


def sync_public(files: list[Path], out_dir: Path, repo_root: Path, destination: Path) -> int:
    copied = 0
    for source in files:
        relative = source.relative_to(repo_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--local-sync-root", type=Path)
    parser.add_argument("--check-staged", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    files, errors = validate_public(out_dir)
    staged: list[Path] = []
    if args.check_staged:
        staged = staged_files(repo_root)
        for path in staged:
            if not path.exists():
                continue
            if not staged_allowed(path, repo_root, out_dir):
                errors.append(f"staged file is outside the Exp43 allowlist: {path}")
            errors.extend(validate_file(path))
    if errors:
        raise SystemExit("Exp43 lightweight validation failed:\n- " + "\n- ".join(errors))

    copied = 0
    if args.local_sync_root:
        copied = sync_public(files, out_dir, repo_root, args.local_sync_root.expanduser().resolve())
    result = {
        "status": "PASS",
        "public_file_count": len(files),
        "public_bytes": sum(path.stat().st_size for path in files),
        "staged_file_count_checked": len(staged),
        "local_sync_file_count": copied,
        "max_file_bytes": MAX_BYTES,
        "prohibited_artifacts_found": 0,
        "secrets_found": 0,
    }
    if not args.verify_only:
        write_json(out_dir / "decision/exp43_lightweight_validation.json", result)
        report = [
            "# Exp43 Lightweight Artifact Validation",
            "",
            "- Status: **PASS**",
            f"- Public files: {len(files)}",
            f"- Public bytes: {result['public_bytes']}",
            f"- Staged files checked: {len(staged)}",
            f"- Files copied to LOCAL_SYNC_ROOT: {copied}",
            "- Maximum file size: 10 MB",
            "- Prohibited artifacts: 0",
            "- API keys or secrets: 0",
            "",
            "Only configs, aggregate CSV tables, Markdown reports, decision/state JSON, and hashes are public.",
        ]
        (out_dir / "reports/exp43_lightweight_validation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
