"""Evidence-bounded Exp00-Exp48 registry and metric implementation audit."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from thesis_exp.exp49_cphce import OUTPUT_ROOT, REPO_ROOT, split_path
from thesis_exp.exp49_cphce.metric_contract import compute_metrics


FIELDS = (
    "experiment_id", "hypothesis", "baseline", "changed_variable", "actual_changes", "split", "backbone",
    "input_fields", "input_template", "target", "output", "loss", "sampler", "inference", "checkpoint_rule",
    "seed", "metric_implementation", "reported_result", "prediction_path", "recomputed_result", "evidence_level",
    "validity", "conclusion",
)


KNOWN: dict[str, dict[str, str]] = {
    "Exp00": {"hypothesis": "Build the processed dataset and fixed split", "validity": "VALID_INFRASTRUCTURE", "evidence_level": "A"},
    "Exp01": {"hypothesis": "Reproduce evaluator-versus-human audit metrics", "validity": "VALID_METRIC_AUDIT", "evidence_level": "A", "metric_implementation": "early audit/human_mean"},
    "Exp02": {"hypothesis": "Reproduce the 0.6B hard-CE judge", "baseline": "paper EduBenchEvaluator", "changed_variable": "local reproduction", "split": "paper_like_triple_seed42", "backbone": "Qwen3-Reranker-0.6B", "input_fields": "question|answer|metric_canonical", "input_template": "qa_metric_baseline", "target": "label_5", "output": "5-class logits", "loss": "unweighted CE", "sampler": "uniform shuffled", "inference": "argmax", "checkpoint_rule": "highest dev Exact; earlier epoch on tie", "seed": "42", "metric_implementation": "human_mean primary", "reported_result": "test Exact=.7299 MAE=.4238 Bias=+.1410 Kendall=.5693", "validity": "VALID_POSITIVE_SINGLE_SEED", "evidence_level": "A"},
    "Exp03": {"hypothesis": "Ablate question/metric/rubric/metadata inputs", "baseline": "Exp02", "changed_variable": "input template", "target": "label_5", "output": "5-class logits", "loss": "CE", "inference": "argmax", "checkpoint_rule": "dev Exact", "seed": "42", "reported_result": "A4 Exact=.7412 MAE=.4031", "validity": "VALID_INPUT_ABLATION_SINGLE_SEED", "evidence_level": "A"},
    "Exp04": {"hypothesis": "Compare classification, regression and ordinal objectives", "baseline": "Exp03 A4", "changed_variable": "objective/output/decode/checkpoint", "input_fields": "A4", "validity": "CONFOUNDED_SINGLE_SEED", "evidence_level": "A", "reported_result": "O3 Exact=.7381 MAE=.3777 Kendall=.6238"},
    "Exp05": {"hypothesis": "Improve low-score behavior with weighting/asymmetric penalties", "baseline": "Exp04 O3", "validity": "VALID_NEGATIVE_FOR_TESTED_STRENGTHS", "evidence_level": "A"},
    "Exp27": {"hypothesis": "Teacher/crossfit/soft-target family", "input_fields": "A4", "metric_implementation": "exp27p rounded-label MAE/Bias/Kendall", "checkpoint_rule": "rounded MAE/L2H/QWK composite", "validity": "METRIC_DRIFT_AND_CONFOUNDED", "evidence_level": "A/B"},
    "Exp35": {"hypothesis": "Qualify label methods without formal student training", "validity": "VALID_DIAGNOSTIC", "evidence_level": "B"},
    "Exp36": {"hypothesis": "Compare human-soft, teacher, LA and SAFER supervision", "baseline": "v0_original_hard", "input_fields": "question|answer|metric_canonical", "input_template": "Exp02/A2", "target": "hard vs human empirical distribution vs teacher variants", "output": "5-class logits plus inactive/active failure head", "checkpoint_rule": "highest dev Exact", "seed": "42", "metric_implementation": "rounded-label MAE/Bias/Kendall", "reported_result": "human-soft Exact .7048->.7244; rounded MAE .3434->.3117", "validity": "POSITIVE_SIGNAL_METRIC_DRIFT_SINGLE_SEED", "evidence_level": "A"},
    "Exp42": {"hypothesis": "RubiDist hard/soft rubric factorial", "validity": "VALID_NEGATIVE_ONLY_UNDER_GROUPCV_PROTOCOL", "evidence_level": "B"},
    "Exp43": {"hypothesis": "RubiMOR ordinal/multi-observer metric heads", "validity": "INCOMPLETE_PAPER_CONTRACT_EVIDENCE", "evidence_level": "B/C"},
    "Exp45": {"hypothesis": "Prototype/DOPR class representations", "validity": "NEGATIVE_OR_INCOMPLETE", "evidence_level": "B/C"},
    "Exp47": {"hypothesis": "Label identifiability diagnosis", "validity": "VALID_DIAGNOSTIC", "evidence_level": "B"},
    "Exp48A": {"hypothesis": "Synthetic low-score candidate generation", "validity": "QUALIFICATION_ONLY", "evidence_level": "B"},
    "Exp48B": {"hypothesis": "Blind qualification of synthetic candidates", "validity": "QUALIFICATION_ONLY", "evidence_level": "B"},
    "Exp48C": {"hypothesis": "Contract-driven pointwise synthetic label audit", "validity": "VALID_NEGATIVE_FOR_TESTED_GENERATION_CONTRACT", "evidence_level": "A/B"},
}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def experiment_ids() -> list[str]:
    values = [f"Exp{index:02d}" for index in range(49)]
    values.extend(("Exp10r", "Exp39b", "Exp48A", "Exp48B", "Exp48C"))
    return values


def _experiment_token(value: str) -> tuple[int, str] | None:
    match = re.search(r"(?<![a-z0-9])exp0*(\d+)([a-z]?)(?=[^a-z0-9]|$)", value.lower())
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def _matching_paths(experiment_id: str) -> list[Path]:
    expected = _experiment_token(experiment_id)
    if expected is None:
        raise ValueError(f"invalid experiment id: {experiment_id}")
    expected_number, expected_suffix = expected
    thesis_root = REPO_ROOT / "thesis_exp"
    search_roots = (
        thesis_root,
        thesis_root / "configs",
        thesis_root / "outputs",
        thesis_root / "artifacts",
        thesis_root / "src" / "edujudge",
        thesis_root / "scripts",
        thesis_root / "exp17_low_score_evidence" / "outputs",
    )
    found: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.iterdir():
            parsed = _experiment_token(path.name)
            if parsed == expected:
                found.append(path)
            elif (
                expected_number == 48
                and expected_suffix in {"a", "b", "c"}
                and parsed == (48, "")
            ):
                # Exp48A/B/C share the family-level implementation directory.
                found.append(path)
    return sorted(set(found))


def build_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_id in experiment_ids():
        paths = _matching_paths(experiment_id)
        known = KNOWN.get(experiment_id, {})
        evidence = known.get("evidence_level") or ("C" if paths else "D")
        validity = known.get("validity") or ("INCOMPLETE_EVIDENCE" if paths else "MISSING_CANONICAL_ARTIFACT")
        row = {field: "" for field in FIELDS}
        row.update(
            {
                "experiment_id": experiment_id,
                "actual_changes": " | ".join(str(path.relative_to(REPO_ROOT)) for path in paths),
                "prediction_path": "PENDING_INVENTORY",
                "recomputed_result": "PENDING_INVENTORY",
                "evidence_level": evidence,
                "validity": validity,
                "conclusion": known.get("conclusion", "No stronger conclusion than the recorded evidence level."),
            }
        )
        row.update(known)
        rows.append(row)
    return rows


def prediction_files() -> list[Path]:
    candidates: list[Path] = []
    for pattern in ("*prediction*.jsonl", "predictions*.jsonl"):
        candidates.extend((REPO_ROOT / "thesis_exp").rglob(pattern))
    return sorted({path for path in candidates if path.is_file() and "exp49_cphce" not in str(path)})


def _reference_by_id() -> dict[str, dict[str, Any]]:
    reference: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev", "test"):
        for line in split_path(split).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            reference[str(row["record_id"])] = row
    return reference


def _normalize_prediction_rows(rows: list[dict[str, Any]], reference: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        ident = str(row.get("record_id") or row.get("id") or row.get("sample_id") or "")
        ref = reference.get(ident, {})
        pred = row.get("pred_label_5", row.get("predicted_label", row.get("prediction")))
        gold = row.get("label_5", row.get("gold_label_5", ref.get("label_5")))
        mean = row.get("human_mean_5", ref.get("human_mean_5"))
        if pred is None or gold is None or mean is None:
            continue
        item = {**row, "pred_label_5": int(round(float(pred))), "label_5": int(round(float(gold))), "human_mean_5": float(mean)}
        output.append(item)
    return output


def inventory_and_recompute() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = _reference_by_id()
    inventory: list[dict[str, Any]] = []
    recomputed: list[dict[str, Any]] = []
    for path in prediction_files():
        relative = str(path.relative_to(REPO_ROOT))
        size = path.stat().st_size
        entry: dict[str, Any] = {"prediction_path": relative, "size_bytes": size, "status": "NOT_RECOMPUTABLE", "n_rows": ""}
        if size > 50 * 1024 * 1024:
            entry["reason"] = "file exceeds 50MB audit ceiling"
            inventory.append(entry)
            continue
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            entry["n_rows"] = len(rows)
            normalized = _normalize_prediction_rows(rows, reference)
            if normalized and len(normalized) == len(rows):
                metrics = compute_metrics(normalized)
                entry["status"] = "RECOMPUTED"
                recomputed.append({"prediction_path": relative, **metrics})
            else:
                entry["reason"] = f"normalized {len(normalized)}/{len(rows)} rows"
        except Exception as exc:
            entry["reason"] = f"{type(exc).__name__}: {exc}"
        inventory.append(entry)
    return inventory, recomputed


def metric_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in (REPO_ROOT / "thesis_exp").rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if not any(token in text for token in ("human_mean_5", "MAE_argmax", "Kendall_tau", "Bin_Agreement", "compute_audit_metrics")):
            continue
        target = "human_mean" if re.search(r"pred[^\n]*-\s*[^\n]*human_mean_5|kendall[^\n]*human_mean", text, re.I) else "unknown_or_rounded"
        if "gold_label_5" in text and "MAE_argmax" in text:
            target = "rounded_label"
        rows.append(
            {
                "implementation_path": str(path.relative_to(REPO_ROOT)),
                "primary_target_classification": target,
                "has_repo_3way_bin": "values <= 2" in text or "gold <= 2" in text or "score_bin" in text,
                "has_human_mean": "human_mean_5" in text,
            }
        )
    return rows


def bin_agreement_audit() -> dict[str, Any]:
    path = REPO_ROOT / "thesis_exp" / "outputs" / "exp01_audit" / "predictions_aligned.jsonl"
    paper = {
        "EduBenchEvaluator": 0.897,
        "DeepSeekV3": 0.867,
        "DeepSeekR1": 0.854,
        "QwQPlus": 0.860,
        "GPT4o": 0.868,
    }
    if not path.exists():
        return {"status": "UNRESOLVED", "reason": "missing aligned evaluator predictions"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    values: dict[str, Any] = {}
    max_error = 0.0
    for evaluator, expected in paper.items():
        correct: list[bool] = []
        for row in rows:
            pred = row.get(f"pred_label_{evaluator}")
            if pred is None:
                continue
            gold = int(row["label_5"])
            pred = int(pred)
            bucket = lambda value: 0 if value <= 2 else (1 if value == 3 else 2)
            correct.append(bucket(gold) == bucket(pred))
        observed = sum(correct) / len(correct)
        error = abs(observed - expected)
        max_error = max(max_error, error)
        values[evaluator] = {"paper": expected, "local": observed, "absolute_error": error, "n": len(correct)}
    status = "REPRODUCED_SPLIT_TOLERANT" if max_error <= 0.003 else "UNRESOLVED"
    return {
        "status": status,
        "definition": "accuracy of {1,2}=low, {3}=mid, {4,5}=high bins",
        "max_absolute_error": max_error,
        "split_note": "Local reconstructed paper-like split differs slightly from the PDF row-level split.",
        "evaluators": values,
    }


def main() -> None:
    audit_dir = OUTPUT_ROOT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    registry = build_registry()
    inventory, recomputed = inventory_and_recompute()
    metric_rows = metric_registry()
    bin_result = bin_agreement_audit()
    prediction_by_exp: dict[str, list[str]] = {}
    for item in inventory:
        match = re.search(r"exp\d+[a-z]?", item["prediction_path"], re.I)
        if match:
            prediction_by_exp.setdefault(match.group(0).lower(), []).append(item["prediction_path"])
    for row in registry:
        key = row["experiment_id"].lower()
        paths = prediction_by_exp.get(key, [])
        row["prediction_path"] = " | ".join(paths) if paths else "NOT_FOUND"
        row["recomputed_result"] = "SEE_RECOMPUTED_METRICS" if any(item.get("prediction_path") in paths for item in recomputed) else "NOT_RECOMPUTABLE"
    _write_csv(audit_dir / "experiment_registry.csv", registry, FIELDS)
    _write_csv(audit_dir / "prediction_inventory.csv", inventory)
    _write_csv(audit_dir / "recomputed_metrics.csv", recomputed)
    _write_csv(audit_dir / "metric_implementation_registry.csv", metric_rows)
    unresolved = [row for row in registry if row["validity"] in {"INCOMPLETE_EVIDENCE", "MISSING_CANONICAL_ARTIFACT"}]
    _write_csv(audit_dir / "unresolved_experiments.csv", unresolved, FIELDS)
    (audit_dir / "bin_agreement_audit.json").write_text(json.dumps(bin_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (audit_dir / "bin_agreement_audit.md").write_text(
        "# Bin Agreement audit\n\n"
        f"Status: **{bin_result['status']}**\n\n"
        f"Definition: `{bin_result.get('definition', 'unknown')}`\n\n"
        f"Maximum absolute error against all five PDF values: {bin_result.get('max_absolute_error', 'n/a')}\n",
        encoding="utf-8",
    )
    (audit_dir / "history_audit_report.md").write_text(
        "# Exp00-Exp48 history audit\n\n"
        f"- Registry rows: {len(registry)}\n"
        f"- Prediction files inventoried: {len(inventory)}\n"
        f"- Prediction files fully recomputed: {len(recomputed)}\n"
        f"- Unresolved/missing experiment entries: {len(unresolved)}\n"
        f"- Paper Bin Agreement: {bin_result['status']}\n\n"
        "Only row-level predictions were recomputed. Missing artifacts remain explicitly unresolved.\n",
        encoding="utf-8",
    )
    print(json.dumps({"registry": len(registry), "inventory": len(inventory), "recomputed": len(recomputed), "bin": bin_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
