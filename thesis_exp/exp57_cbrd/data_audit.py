"""Train/dev-only source, target-support, and shuffled-residual audits."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp57_cbrd import (
    DATA_ROOT,
    LEGACY_HMSA_SOURCE_COMMIT,
    LEGACY_RESULT_COMMIT,
    OUTPUT_ROOT,
    SHUFFLE_SEED,
)
from thesis_exp.exp57_cbrd.method import describe_target, soft_target_from_scores, target_state, thirds


EXPECTED_SHARED_HASHES = {
    "thesis_exp/data/splits/paper_like_triple_seed42/train.jsonl": "0a1733b9209984c5c4291d205d1ac6057bed341717903b9de075d07de44a878e",
    "thesis_exp/data/splits/paper_like_triple_seed42/dev.jsonl": "a18d6a27b9a524d4592a359658ae70c9348fe88e43c962971ba95f62d2b6cdf0",
    "thesis_exp/src/edujudge/exp02/train_ce_baseline.py": "fccf33d68bdd7ea889032c071422baff6bfd1cf8a4e3e54927638bcf258c9b68",
    "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py": "82eaefd6c711f632a23d61e7591d5c447076e9e351134887b726d7db75a7d241",
    "thesis_exp/src/edujudge/utils/io.py": "e19250fbf9b7a2013b61078d29100137ee6cefab76cd8dda01ab6888156f1364",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(split: str) -> list[dict[str, Any]]:
    if split not in {"train", "dev"}:
        raise PermissionError("CBRD Stage 0 is strictly train/dev-only; test is forbidden")
    return read_jsonl(DATA_ROOT / f"{split}.jsonl")


def row_scores(row: dict[str, Any]) -> tuple[int, int, int]:
    values = tuple(int(round(float(row[f"human_{index}_5"]))) for index in (1, 2, 3))
    if any(value not in (1, 2, 3, 4, 5) for value in values):
        raise ValueError(f"Out-of-range human score for {row.get('record_id')}: {values}")
    return values  # type: ignore[return-value]


def row_target(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return soft_target_from_scores(row_scores(row))


def model_rows(split: str) -> list[dict[str, Any]]:
    """Expose frozen raw rows in the exact Exp02 input schema, without Exp49 imports."""

    from thesis_exp.src.edujudge.exp02.build_exp02_dataset import make_prompt

    converted: list[dict[str, Any]] = []
    for row in load_rows(split):
        scores = row_scores(row)
        converted.append(
            {
                "id": row["record_id"],
                "record_id": row["record_id"],
                "split": split,
                "text": make_prompt(row),
                "label": int(row["label_5"]) - 1,
                "label_5": int(row["label_5"]),
                "human_mean_5": float(row["human_mean_5"]),
                "human_1_5": scores[0],
                "human_2_5": scores[1],
                "human_3_5": scores[2],
                "soft_target_5": list(row_target(row)),
                "triple_key": row.get("triple_key"),
                "question_key": row.get("question_key"),
                "answer_key": row.get("answer_key"),
                "metric_canonical": row.get("metric_canonical"),
                "scenario_canonical": row.get("scenario_canonical"),
                "subject_canonical": row.get("subject_canonical"),
                "language": row.get("language"),
                "generator_model": row.get("generator_model"),
            }
        )
    return converted


def stable_row_id(row: dict[str, Any]) -> str:
    identifier = row.get("record_id") or row.get("id")
    if not identifier:
        raise ValueError("Every row needs a stable record identifier")
    return str(identifier)


def donor_sort_key(row_id: str, label: int, shuffle_seed: int) -> str:
    return hashlib.sha256(f"{shuffle_seed}\t{label}\t{row_id}".encode("utf-8")).hexdigest()


def mapping_sha256(mapping: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(mapping, key=lambda value: str(value["recipient_record_id"])):
        digest.update(
            (
                f"{row['hard_label']}\t{row['recipient_record_id']}\t"
                f"{row['donor_record_id']}\t{row['shuffled_target_thirds']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def audit_split(split: str) -> dict[str, Any]:
    rows = load_rows(split)
    by_label: Counter[int] = Counter()
    by_state: Counter[str] = Counter()
    targets: Counter[tuple[int, int, int, int, int]] = Counter()
    residuals: Counter[tuple[int, int, int, int, int]] = Counter()
    deltas: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    for row in rows:
        label = int(row["label_5"])
        target = row_target(row)
        try:
            description = describe_target(label, target)
            stored_mean = float(row["human_mean_5"])
            if abs(description["human_mean"] - stored_mean) > 1e-9:
                raise ValueError(f"stored human mean {stored_mean} is inconsistent")
        except ValueError as error:
            invalid.append({"record_id": stable_row_id(row), "reason": str(error)})
            continue
        by_label[label] += 1
        by_state[description["state"]] += 1
        targets[tuple(description["target_thirds"])] += 1
        residuals[tuple(description["residual_thirds"])] += 1
        deltas[f"{description['mean_minus_hard_label']:+.6f}"] += 1
    return {
        "split": split,
        "rows": len(rows),
        "hard_label_counts": {str(key): value for key, value in sorted(by_label.items())},
        "relation_state_counts": dict(sorted(by_state.items())),
        "distinct_target_vectors": len(targets),
        "target_thirds_counts": {str(list(key)): value for key, value in sorted(targets.items())},
        "distinct_residual_vectors": len(residuals),
        "residual_thirds_counts": {str(list(key)): value for key, value in sorted(residuals.items())},
        "mean_minus_hard_label_counts": dict(sorted(deltas.items())),
        "invalid_rows": invalid,
        "checks": {
            "no_invalid_rows": not invalid,
            "only_zero_down_up_relation_states": set(by_state).issubset({"zero", "down", "up"}),
            "three_rater_target_grid": all(sum(key) == 3 for key in targets),
            "no_test_access": True,
        },
    }


def shuffled_residual_audit(rows: list[dict[str, Any]], *, shuffle_seed: int = SHUFFLE_SEED) -> dict[str, Any]:
    """Reproduce Exp55's fixed permutation without importing unavailable Exp51 code.

    The old shuffled-soft control moved both the auxiliary-head target and the
    backbone signal.  This audit only establishes the deterministic mapping;
    the later CBRD shuffled-residual arm will keep the auxiliary-head target
    fixed and permute this residual route alone.
    """

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label_5"])].append(row)
    mapping: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    changed = 0
    for label in sorted(by_label):
        recipients = sorted(by_label[label], key=stable_row_id)
        donors = sorted(
            by_label[label],
            key=lambda row: donor_sort_key(stable_row_id(row), label, shuffle_seed),
        )
        original_multiset = Counter(thirds(row_target(row)) for row in recipients)
        donor_multiset = Counter(thirds(row_target(row)) for row in donors)
        if original_multiset != donor_multiset:
            raise AssertionError(f"Target multiset was not preserved for label {label}")
        for recipient, donor in zip(recipients, donors):
            original = row_target(recipient)
            shuffled = row_target(donor)
            original_state = target_state(label, original)
            shuffled_state = target_state(label, shuffled)
            original_key = thirds(original)
            shuffled_key = thirds(shuffled)
            effectively_changed = original_key != shuffled_key
            changed += int(effectively_changed)
            transitions[f"{original_state}->{shuffled_state}"] += 1
            mapping.append(
                {
                    "hard_label": label,
                    "recipient_record_id": stable_row_id(recipient),
                    "donor_record_id": stable_row_id(donor),
                    "self_assignment": stable_row_id(recipient) == stable_row_id(donor),
                    "original_target_thirds": list(original_key),
                    "shuffled_target_thirds": list(shuffled_key),
                    "original_state": original_state,
                    "shuffled_state": shuffled_state,
                    "effectively_changed": effectively_changed,
                }
            )
    expected_mapping_sha = "b4e96c49607700be99816582c1b85a8085b8c5abb260ddafbad4e9ee0dc25ad4"
    mapping_hash = mapping_sha256(mapping)
    return {
        "shuffle_seed": shuffle_seed,
        "rows": len(rows),
        "mapping_sha256": mapping_hash,
        "expected_exp55_mapping_sha256": expected_mapping_sha,
        "mapping_matches_exp55": mapping_hash == expected_mapping_sha,
        "effective_target_changes": changed,
        "effective_change_rate": changed / len(rows),
        "state_transition_counts": dict(sorted(transitions.items())),
        "checks": {
            "within_hard_label_multisets_preserved": True,
            "effective_change_rate_matches_exp55": abs(changed / len(rows) - 0.5614167294649586) < 1e-12,
            "mapping_matches_exp55": mapping_hash == expected_mapping_sha,
            "no_test_access": True,
        },
        "mapping": mapping,
    }


def _git_blob(revision: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=repo_root(),
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _legacy_blob(revision: str, relative: str, *, scope: str) -> bytes:
    """Resolve a historical blob from Git or the Exp57 offline snapshot."""

    try:
        return _git_blob(revision, relative)
    except subprocess.CalledProcessError:
        archive_root = repo_root() / "thesis_exp" / "exp57_cbrd" / "legacy_source"
        if scope == "lock":
            fallback = archive_root / "locks" / Path(relative).name
        elif relative.endswith(".jsonl"):
            fallback = repo_root() / relative
        else:
            fallback = archive_root / scope / relative
        if not fallback.is_file():
            raise RuntimeError(
                f"Neither Git object {revision}:{relative} nor archived fallback {fallback} is available"
            )
        return fallback.read_bytes()


def source_closure_audit() -> dict[str, Any]:
    """Verify all legacy blobs and the live shared dependencies by SHA-256."""

    source_lock_relative = "thesis_exp/configs/exp51_hmsa/source_lock.json"
    formal_lock_relative = "thesis_exp/configs/exp51_hmsa/formal_lock.json"
    source_lock_bytes = _legacy_blob(
        LEGACY_RESULT_COMMIT,
        source_lock_relative,
        scope="lock",
    )
    formal_lock_bytes = _legacy_blob(
        LEGACY_RESULT_COMMIT,
        formal_lock_relative,
        scope="lock",
    )
    source_lock = json.loads(source_lock_bytes)
    formal_lock = json.loads(formal_lock_bytes)
    historical: list[dict[str, Any]] = []
    for relative, expected in sorted(source_lock["files"].items()):
        actual = sha256_bytes(
            _legacy_blob(LEGACY_HMSA_SOURCE_COMMIT, relative, scope="hmsa_source")
        )
        historical.append({"scope": "hmsa_source", "path": relative, "expected": expected, "actual": actual, "match": actual == expected})
    for relative, expected in sorted(formal_lock["files"].items()):
        actual = sha256_bytes(
            _legacy_blob(LEGACY_RESULT_COMMIT, relative, scope="formal_snapshot")
        )
        historical.append({"scope": "formal_snapshot", "path": relative, "expected": expected, "actual": actual, "match": actual == expected})
    live: list[dict[str, Any]] = []
    root = repo_root()
    for relative, expected in sorted(EXPECTED_SHARED_HASHES.items()):
        file_path = root / relative
        actual = sha256_file(file_path) if file_path.is_file() else None
        live.append({"path": relative, "expected": expected, "actual": actual, "match": actual == expected})
    return {
        "legacy_result_commit": LEGACY_RESULT_COMMIT,
        "legacy_hmsa_source_commit": LEGACY_HMSA_SOURCE_COMMIT,
        "source_lock_sha256": sha256_bytes(source_lock_bytes),
        "formal_lock_sha256": sha256_bytes(formal_lock_bytes),
        "historical_blob_checks": historical,
        "live_shared_dependency_checks": live,
        "checks": {
            "all_historical_blobs_match": all(item["match"] for item in historical),
            "all_live_shared_dependencies_match": all(item["match"] for item in live),
            "no_test_access": True,
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    train = audit_split("train")
    dev = audit_split("dev")
    shuffle = shuffled_residual_audit(load_rows("train"))
    shuffle_without_mapping = {key: value for key, value in shuffle.items() if key != "mapping"}
    source_closure = source_closure_audit()
    report = {
        "status": "PASS" if all((
            source_closure["checks"]["all_historical_blobs_match"],
            source_closure["checks"]["all_live_shared_dependencies_match"],
            train["checks"]["no_invalid_rows"],
            dev["checks"]["no_invalid_rows"],
            shuffle["checks"]["mapping_matches_exp55"],
        )) else "FAIL",
        "source_closure": source_closure,
        "train": train,
        "dev": dev,
        "shuffled_residual": shuffle_without_mapping,
        "test_access_count": 0,
    }
    write_json(OUTPUT_ROOT / "audit" / "stage0_data_and_source_audit.json", report)
    write_json(OUTPUT_ROOT / "audit" / "stage0_shuffle_mapping.json", {"mapping": shuffle["mapping"]})
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
