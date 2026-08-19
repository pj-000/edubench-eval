"""Post-hoc Exp49 mechanism audit that cannot alter the locked Exp50 target."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp49_cphce import REPO_ROOT, split_path
from thesis_exp.exp50_cahs import EXP49_OUTPUT_ROOT, LOCKED_BASELINE_COMMIT, OUTPUT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rater_pattern(row: dict[str, Any]) -> str:
    return "[" + ",".join(str(int(row[f"human_{index}"])) for index in (1, 2, 3)) + "]"


def sorted_pattern(row: dict[str, Any]) -> str:
    values = sorted(int(row[f"human_{index}"]) for index in (1, 2, 3))
    return "[" + ",".join(map(str, values)) + "]"


def margin_4(row: dict[str, Any]) -> float:
    return float(row["logit_4"]) - max(float(row["logit_3"]), float(row["logit_5"]))


def main() -> None:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", LOCKED_BASELINE_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Locked Exp49 commit 9ad6190 is not an ancestor of the current implementation")
    dev = {str(row["record_id"]): row for row in read_jsonl(split_path("dev"))}
    b_path = EXP49_OUTPUT_ROOT / "runs" / "b0_hard_ce" / "seed_42" / "predictions" / "predictions_dev.jsonl"
    m_path = EXP49_OUTPUT_ROOT / "runs" / "m1_human_soft" / "seed_42" / "predictions" / "predictions_dev.jsonl"
    baseline = {str(row["record_id"]): row for row in read_jsonl(b_path)}
    treatment = {str(row["record_id"]): row for row in read_jsonl(m_path)}
    if set(dev) != set(baseline) or set(dev) != set(treatment):
        raise ValueError("Exp49 dev prediction IDs do not align with the fixed dev split")

    flips: dict[str, list[dict[str, Any]]] = {"fixes": [], "breaks": []}
    label4_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    distance_counts: Counter[int] = Counter()
    for ident in sorted(dev):
        source, b, m = dev[ident], baseline[ident], treatment[ident]
        gold = int(source["label_5"])
        b_pred, m_pred = int(b["pred_label_5"]), int(m["pred_label_5"])
        scores = [int(source[f"human_{index}"]) for index in (1, 2, 3)]
        counts = Counter(scores)
        if max(counts.values()) == 2:
            majority = max(counts, key=counts.get)
            minority = next(value for value, count in counts.items() if count == 1)
            distance_counts[abs(majority - minority)] += 1
        record = {
            "record_id": ident,
            "gold_label_5": gold,
            "human_pattern": sorted_pattern(source),
            "b0_pred_label_5": b_pred,
            "m1_pred_label_5": m_pred,
            "b0_expected": float(b["pred_score_expected"]),
            "m1_expected": float(m["pred_score_expected"]),
            "b0_margin_4_vs_3_5": margin_4(b),
            "m1_margin_4_vs_3_5": margin_4(m),
            **{f"b0_prob_{label}": float(b[f"prob_{label}"]) for label in range(1, 6)},
            **{f"m1_prob_{label}": float(m[f"prob_{label}"]) for label in range(1, 6)},
            **{f"b0_logit_{label}": float(b[f"logit_{label}"]) for label in range(1, 6)},
            **{f"m1_logit_{label}": float(m[f"logit_{label}"]) for label in range(1, 6)},
        }
        if b_pred != gold and m_pred == gold:
            flips["fixes"].append(record)
        elif b_pred == gold and m_pred != gold:
            flips["breaks"].append(record)
        if gold == 4:
            label4_groups[sorted_pattern(source)].append(record)

    audit_dir = OUTPUT_ROOT / "audit"
    write_csv(audit_dir / "fixes_45.csv", flips["fixes"])
    write_csv(audit_dir / "breaks_48.csv", flips["breaks"])
    label4_rows: list[dict[str, Any]] = []
    for pattern in ("[3,4,4]", "[4,4,4]", "[4,4,5]"):
        rows = label4_groups[pattern]
        label4_rows.append(
            {
                "human_pattern": pattern,
                "n": len(rows),
                "b0_correct_m1_wrong": sum(r["b0_pred_label_5"] == 4 and r["m1_pred_label_5"] != 4 for r in rows),
                "b0_wrong_m1_correct": sum(r["b0_pred_label_5"] != 4 and r["m1_pred_label_5"] == 4 for r in rows),
                "b0_4_to_m1_3": sum(r["b0_pred_label_5"] == 4 and r["m1_pred_label_5"] == 3 for r in rows),
                "b0_4_to_m1_5": sum(r["b0_pred_label_5"] == 4 and r["m1_pred_label_5"] == 5 for r in rows),
                "mean_b0_margin_4_vs_3_5": float(np.mean([r["b0_margin_4_vs_3_5"] for r in rows])),
                "mean_m1_margin_4_vs_3_5": float(np.mean([r["m1_margin_4_vs_3_5"] for r in rows])),
                "mean_expected_shift_m1_minus_b0": float(np.mean([r["m1_expected"] - r["b0_expected"] for r in rows])),
            }
        )
    write_csv(audit_dir / "label4_transition_by_human_pattern.csv", label4_rows)

    b_history = json.loads((EXP49_OUTPUT_ROOT / "runs" / "b0_hard_ce" / "seed_42" / "dev_metrics_history.json").read_text())
    m_history = json.loads((EXP49_OUTPUT_ROOT / "runs" / "m1_human_soft" / "seed_42" / "dev_metrics_history.json").read_text())
    trajectory: list[dict[str, Any]] = []
    keys = ("Exact_rounded", "MAE_human_mean", "Kendall_human_mean", "Recall_3_rounded", "Recall_4_rounded", "Recall_5_rounded", "Bias_human_mean", "L2H_count", "ECE_rounded")
    for b, m in zip(b_history, m_history):
        row: dict[str, Any] = {"epoch": b["epoch"]}
        for key in keys:
            row[f"b0_{key}"] = b[key]
            row[f"m1_{key}"] = m[key]
        trajectory.append(row)
    write_csv(audit_dir / "exp49_same_epoch_trajectory.csv", trajectory)

    repro_dir = audit_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)
    patch_path = repro_dir / "exp49_de00041_to_9ad6190.patch"
    patch_path.write_bytes(subprocess.check_output(["git", "format-patch", "--stdout", f"de00041..{LOCKED_BASELINE_COMMIT}"], cwd=REPO_ROOT))
    hash_paths = [
        REPO_ROOT / "thesis_exp" / "exp49_cphce" / "train.py",
        REPO_ROOT / "thesis_exp" / "exp49_cphce" / "metric_contract.py",
        REPO_ROOT / "thesis_exp" / "configs" / "exp49_cphce" / "baseline_hard_ce.yaml",
        REPO_ROOT / "thesis_exp" / "configs" / "exp49_cphce" / "cphce_human_soft.yaml",
        EXP49_OUTPUT_ROOT / "decision" / "seed42_decision.json",
        EXP49_OUTPUT_ROOT / "seed42_analysis.md",
        split_path("train"),
        split_path("dev"),
        b_path,
        m_path,
        patch_path,
    ]
    manifest = {
        "locked_commit": LOCKED_BASELINE_COMMIT,
        "test_split_hashed_or_read": False,
        "files": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in hash_paths},
        "adjacent_disagreement_distance_counts_dev": dict(sorted(distance_counts.items())),
        "fixes": len(flips["fixes"]),
        "breaks": len(flips["breaks"]),
        "label4_groups": label4_rows,
    }
    (repro_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
