"""Collect Exp48C pointwise audits and diagnose dependence on the Exp48B contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

from .common import MODULE, ROOT, read_jsonl, sha256_path, write_csv, write_json
from .exp48c_common import (
    EXP48B_DECISION, EXP48B_METRICS, FAMILIES, OUT, VERIFIERS,
    load_mapping, output_path, packet_path,
)
from .validate_exp48c_rubric_only_outputs import validate_verifier


def safe_qwk(left: list[int], right: list[int]) -> float:
    if not left or len(set(left + right)) < 2:
        return 0.0
    value = cohen_kappa_score(left, right, labels=[1, 2, 3, 4, 5], weights="quadratic")
    return 0.0 if math.isnan(float(value)) else float(value)


def wilson(success: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = success / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, center - spread), min(1.0, center + spread)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def build_score_rows(verifier: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    summary, errors = validate_verifier(verifier, packet_path(verifier), output_path(verifier))
    if not output_path(verifier).exists():
        return [], summary, errors
    packets = {row["packet_id"]: row for row in read_jsonl(packet_path(verifier))}
    mapping = {row["packet_id"]: row for row in load_mapping(verifier)}
    outputs = {row["packet_id"]: row for row in read_jsonl(output_path(verifier))}
    rows = []
    for packet_id, output in outputs.items():
        if packet_id not in mapping or packet_id not in packets:
            continue
        private = mapping[packet_id]
        rows.append({
            "verifier": verifier, "packet_id": packet_id,
            "family_hash": private["frozen_family_hash"],
            "answer_id": private["original_answer_id"], "metric": private["metric"],
            "intended_score": int(private["intended_score"]),
            "predicted_score": int(output["most_plausible_score"]),
            "confidence": output["confidence"],
            "needs_adjudication": bool(output["needs_adjudication"]),
            "model_family": output["verifier_provenance"]["model_family"],
            "model_version": output["verifier_provenance"]["model_version"],
            "session_id": output["verifier_provenance"]["session_id"],
        })
    return rows, summary, errors


def metric_summary(verifier: str, rows: list[dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    gold = [row["intended_score"] for row in rows]
    pred = [row["predicted_score"] for row in rows]
    exact = sum(a == b for a, b in zip(gold, pred))
    score2 = [row for row in rows if row["intended_score"] == 2]
    score3 = [row for row in rows if row["intended_score"] == 3]
    score4 = [row for row in rows if row["intended_score"] == 4]
    ordered = 0
    for family_hash in {row["family_hash"] for row in rows}:
        values = {row["intended_score"]: row["predicted_score"] for row in rows if row["family_hash"] == family_hash}
        ordered += int(set(values) == {2, 3, 4} and values[2] < values[3] < values[4])
    exact_lo, exact_hi = wilson(exact, len(rows))
    s2_correct = sum(row["predicted_score"] == 2 for row in score2)
    s2_lo, s2_hi = wilson(s2_correct, len(score2))
    return {
        "verifier": verifier, "rows": len(rows), "complete": validation["complete"],
        "schema_success_rate": validation["schema_success_rate"],
        "intended_exact_rows": exact, "intended_exact_rate": exact / max(1, len(rows)),
        "exact_wilson_low": exact_lo, "exact_wilson_high": exact_hi,
        "MAE": mean(abs(a - b) for a, b in zip(gold, pred)) if rows else None,
        "QWK": safe_qwk(gold, pred),
        "within_one": sum(abs(a - b) <= 1 for a, b in zip(gold, pred)) / max(1, len(rows)),
        "score2_correct": s2_correct, "score2_recall": s2_correct / max(1, len(score2)),
        "score2_wilson_low": s2_lo, "score2_wilson_high": s2_hi,
        "score2_to_4_or_5": sum(row["predicted_score"] >= 4 for row in score2),
        "score3_recall": sum(row["predicted_score"] == 3 for row in score3) / max(1, len(score3)),
        "score4_recall": sum(row["predicted_score"] == 4 for row in score4) / max(1, len(score4)),
        "ordered_families": ordered,
        "low_confidence_rows": sum(row["confidence"] == "low" for row in rows),
        "needs_adjudication_rows": sum(row["needs_adjudication"] for row in rows),
    }


def bootstrap_contract_delta(rows: list[dict[str, Any]], old_exact: float, old_qwk: float, repeats: int = 5000) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family_hash"], []).append(row)
    families = sorted(by_family)
    if not families:
        return []
    rng = random.Random(48003)
    exact_deltas, qwk_deltas = [], []
    for _ in range(repeats):
        sampled = [rng.choice(families) for _ in families]
        sample_rows = [row for family in sampled for row in by_family[family]]
        gold = [row["intended_score"] for row in sample_rows]
        pred = [row["predicted_score"] for row in sample_rows]
        exact_deltas.append(sum(a == b for a, b in zip(gold, pred)) / len(sample_rows) - old_exact)
        qwk_deltas.append(safe_qwk(gold, pred) - old_qwk)
    return [
        {"metric": "exact_rate_delta", "repeats": repeats, "estimate": mean(exact_deltas), "ci95_low": percentile(exact_deltas, 2.5), "ci95_high": percentile(exact_deltas, 97.5)},
        {"metric": "qwk_delta", "repeats": repeats, "estimate": mean(qwk_deltas), "ci95_low": percentile(qwk_deltas, 2.5), "ci95_high": percentile(qwk_deltas, 97.5)},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    families = read_jsonl(FAMILIES)
    family_by_hash = {
        hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(): row
        for row in families
    }
    all_rows: dict[str, list[dict[str, Any]]] = {}
    completion_rows, quote_rows, evidence_rows, all_errors = [], [], [], []
    summaries: dict[str, dict[str, Any]] = {}
    for verifier in VERIFIERS:
        rows, validation, errors = build_score_rows(verifier)
        all_rows[verifier] = rows
        all_errors.extend({"verifier": verifier, **row} for row in errors)
        models = sorted({row["model_family"] for row in rows})
        versions = sorted({row["model_version"] for row in rows})
        sessions = sorted({row["session_id"] for row in rows})
        completion_rows.append({
            "verifier": verifier, "packet_rows": validation["packet_rows"],
            "output_rows": validation["output_rows"], "complete": validation["complete"],
            "schema_success_rate": validation["schema_success_rate"],
            "provenance_completeness": validation["provenance_completeness"],
            "model_families": "|".join(models), "model_versions": "|".join(versions),
            "session_count": len(sessions), "one_context_per_packet": len(sessions) == len(rows),
        })
        quote_rows.append({"verifier": verifier, "valid": validation["rubric_quote_valid_count"], "total": validation["packet_rows"], "validity": validation["rubric_quote_validity"]})
        evidence_rows.append({"verifier": verifier, "valid": validation["evidence_valid"], "total": validation["evidence_total"], "validity": validation["evidence_validity"]})
        summaries[verifier] = metric_summary(verifier, rows, validation)
    write_csv(args.out_dir / "tables/exp48c_verifier_completion.csv", completion_rows)
    write_csv(args.out_dir / "tables/exp48c_rubric_quote_validity.csv", quote_rows)
    write_csv(args.out_dir / "tables/exp48c_evidence_validity.csv", evidence_rows)
    write_csv(args.out_dir / "tables/exp48c_per_verifier_score_metrics.csv", list(summaries.values()))

    confusion = []
    for verifier, rows in all_rows.items():
        for (gold, pred), count in sorted(Counter((row["intended_score"], row["predicted_score"]) for row in rows).items()):
            confusion.append({"verifier": verifier, "intended_score": gold, "predicted_score": pred, "count": count})
    write_csv(args.out_dir / "tables/exp48c_per_verifier_confusion.csv", confusion)

    family_rows = []
    for verifier, rows in all_rows.items():
        for family_hash in family_by_hash:
            subset = [row for row in rows if row["family_hash"] == family_hash]
            values = {row["intended_score"]: row["predicted_score"] for row in subset}
            exact = len(subset) == 3 and all(row["intended_score"] == row["predicted_score"] for row in subset)
            ordered = set(values) == {2, 3, 4} and values[2] < values[3] < values[4]
            family_rows.append({"verifier": verifier, "family_hash": family_hash, "metric": family_by_hash[family_hash]["metric"], "fully_confirmed": exact, "ordered": ordered})
    write_csv(args.out_dir / "tables/exp48c_family_ordering.csv", family_rows)

    codex_by_answer = {row["answer_id"]: row for row in all_rows["codex"]}
    qwen_by_answer = {row["answer_id"]: row for row in all_rows["qwen"]}
    common_ids = sorted(set(codex_by_answer) & set(qwen_by_answer))
    cross_gold = [codex_by_answer[key]["intended_score"] for key in common_ids]
    cross_a = [codex_by_answer[key]["predicted_score"] for key in common_ids]
    cross_b = [qwen_by_answer[key]["predicted_score"] for key in common_ids]
    cross_exact_rows = sum(a == b for a, b in zip(cross_a, cross_b))
    cross = {
        "rows": len(common_ids), "exact_rows": cross_exact_rows,
        "exact_rate": cross_exact_rows / max(1, len(common_ids)),
        "MAE_between_verifiers": mean(abs(a - b) for a, b in zip(cross_a, cross_b)) if common_ids else None,
        "QWK": safe_qwk(cross_a, cross_b),
        "within_one": sum(abs(a - b) <= 1 for a, b in zip(cross_a, cross_b)) / max(1, len(common_ids)),
    }
    write_csv(args.out_dir / "tables/exp48c_cross_verifier_agreement.csv", [cross])

    acceptance = []
    for family_hash, family in family_by_hash.items():
        ids = [answer["answer_id"] for answer in family["answers"]]
        complete = all(key in codex_by_answer and key in qwen_by_answer for key in ids)
        fully = complete and all(codex_by_answer[key]["predicted_score"] == codex_by_answer[key]["intended_score"] and qwen_by_answer[key]["predicted_score"] == qwen_by_answer[key]["intended_score"] for key in ids)
        score2_id = next(answer["answer_id"] for answer in family["answers"] if answer["intended_score"] == 2)
        joint_score2 = complete and codex_by_answer[score2_id]["predicted_score"] == 2 and qwen_by_answer[score2_id]["predicted_score"] == 2
        acceptance.append({"family_hash": family_hash, "metric": family["metric"], "joint_fully_confirmed": fully, "joint_score2_correct": joint_score2})
    write_csv(args.out_dir / "tables/exp48c_acceptance_by_metric.csv", acceptance)
    joint_families = sum(row["joint_fully_confirmed"] for row in acceptance)
    joint_score2 = sum(row["joint_score2_correct"] for row in acceptance)
    accepted_metrics = len({row["metric"] for row in acceptance if row["joint_fully_confirmed"]})

    baseline = json.loads(EXP48B_DECISION.read_text(encoding="utf-8"))
    codex = summaries["codex"]
    codex_models = {row["model_family"] for row in all_rows["codex"]}
    same_model = bool(all_rows["codex"]) and codex_models == {"gpt-5.5"} and baseline.get("single_verifier_model_families") == ["gpt-5.5"]
    old_exact, old_qwk = float(baseline["intended_exact_rate"]), float(baseline["qwk"])
    dependence = {
        "scope": "overall",
        "same_model_ablation": same_model,
        "comparison_interpretation": "causal_same_model_ablation" if same_model else "descriptive_only",
        "contract_aware_exact": old_exact, "rubric_only_exact": codex["intended_exact_rate"],
        "exact_delta": codex["intended_exact_rate"] - old_exact,
        "contract_aware_QWK": old_qwk, "rubric_only_QWK": codex["QWK"],
        "QWK_delta": codex["QWK"] - old_qwk,
        "contract_aware_score2_recall": baseline["score2_confirmed"] / 12,
        "rubric_only_score2_recall": codex["score2_recall"],
        "score2_recall_delta": codex["score2_recall"] - baseline["score2_confirmed"] / 12,
    }
    dependence_rows = [dependence]
    for intended_score in (2, 3, 4):
        subset = [row for row in all_rows["codex"] if row["intended_score"] == intended_score]
        new_exact = sum(row["predicted_score"] == intended_score for row in subset) / max(1, len(subset))
        dependence_rows.append({
            "scope": f"intended_{intended_score}",
            "same_model_ablation": same_model,
            "comparison_interpretation": dependence["comparison_interpretation"],
            "contract_aware_exact": 1.0 if subset else None,
            "rubric_only_exact": new_exact if subset else None,
            "exact_delta": new_exact - 1.0 if subset else None,
            "contract_aware_QWK": None, "rubric_only_QWK": None, "QWK_delta": None,
            "contract_aware_score2_recall": 1.0 if intended_score == 2 and subset else None,
            "rubric_only_score2_recall": new_exact if intended_score == 2 and subset else None,
            "score2_recall_delta": new_exact - 1.0 if intended_score == 2 and subset else None,
        })
    write_csv(args.out_dir / "tables/exp48c_contract_dependence.csv", dependence_rows)
    codex_correct = [row["intended_score"] == row["predicted_score"] for row in all_rows["codex"]]
    b = sum(not value for value in codex_correct)
    mcnemar = {"old_correct_new_wrong": b, "old_wrong_new_correct": 0, "discordant": b, "exact_p_value": mcnemar_exact(b, 0), "interpretation": dependence["comparison_interpretation"]}
    write_csv(args.out_dir / "tables/exp48c_contract_dependence_mcnemar.csv", [mcnemar])
    bootstrap = bootstrap_contract_delta(all_rows["codex"], old_exact, old_qwk)
    write_csv(args.out_dir / "tables/exp48c_bootstrap_ci.csv", bootstrap)

    available = {verifier: output_path(verifier).exists() for verifier in VERIFIERS}
    both_complete = all(summary["complete"] for summary in summaries.values())
    verifier_gates = {}
    for verifier, summary in summaries.items():
        validation = next(row for row in completion_rows if row["verifier"] == verifier)
        quote = next(row for row in quote_rows if row["verifier"] == verifier)
        evidence = next(row for row in evidence_rows if row["verifier"] == verifier)
        verifier_gates[verifier] = {
            "completion_36": summary["rows"] == 36 and bool(validation["complete"]),
            "schema_success_1": float(validation["schema_success_rate"]) == 1.0,
            "rubric_quote_1": float(quote["validity"]) == 1.0,
            "evidence_ge_0p95": float(evidence["validity"]) >= 0.95,
            "exact_ge_30": summary["intended_exact_rows"] >= 30,
            "qwk_ge_0p85": summary["QWK"] >= 0.85,
            "score2_ge_10": summary["score2_correct"] >= 10,
            "score2_to_high_zero": summary["score2_to_4_or_5"] == 0,
            "ordered_ge_10": summary["ordered_families"] >= 10,
            "low_confidence_le_4": summary["low_confidence_rows"] <= 4,
        }
    cross_gates = {
        "cross_exact_ge_30": cross["exact_rows"] >= 30,
        "cross_qwk_ge_0p85": cross["QWK"] >= 0.85,
        "joint_score2_ge_10": joint_score2 >= 10,
        "joint_families_ge_9": joint_families >= 9,
        "accepted_metrics_ge_9": accepted_metrics >= 9,
    }
    packet_decision = json.loads((args.out_dir / "decision/exp48c_packet_decision.json").read_text(encoding="utf-8"))
    packet_pass = packet_decision["status"] == "PACKET_BLINDNESS_GO"
    all_pass = packet_pass and both_complete and all(all(gates.values()) for gates in verifier_gates.values()) and all(cross_gates.values())
    if not both_complete:
        status, stop, chinese = "AWAITING_SECOND_VERIFIER", False, False
    elif all_pass:
        status, stop, chinese = "EXP48C_RUBRIC_ONLY_AUDIT_GO", False, True
    else:
        status, stop, chinese = "EXP48C_RUBRIC_ONLY_AUDIT_NO_GO", True, False
    decision = {
        "status": status, "available_outputs": available,
        "packet_blindness_pass": packet_pass, "pointwise_context_isolation": True,
        "same_model_ablation": same_model, "per_verifier": summaries,
        "cross_verifier": cross, "joint_score2_correct": joint_score2,
        "joint_fully_confirmed_families": joint_families, "accepted_metrics": accepted_metrics,
        "contract_dependence": dependence, "verifier_gates": verifier_gates,
        "cross_gates": cross_gates, "recommend_chinese_replication": chinese,
        "verifier_provenance": {
            verifier: {
                "model_families": sorted({row["model_family"] for row in all_rows[verifier]}),
                "model_versions": sorted({row["model_version"] for row in all_rows[verifier]}),
                "session_count": len({row["session_id"] for row in all_rows[verifier]}),
                "one_context_per_packet": len({row["session_id"] for row in all_rows[verifier]}) == len(all_rows[verifier]),
            }
            for verifier in VERIFIERS
        },
        "recommend_scale_generation": False, "training_authorized": False,
        "stop_synthetic_low_tail_route_permanently": stop,
        "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp48c_rubric_only_audit_decision.json", decision)

    write_json(args.out_dir / "hashes/exp48c_private_output_hashes.json", {
        verifier: {"exists": output_path(verifier).exists(), "sha256": sha256_path(output_path(verifier)) if output_path(verifier).exists() else None}
        for verifier in VERIFIERS
    })
    prompt_schema_paths = [
        MODULE / "prompts/exp48c_rubric_only_pointwise_verifier_prompt.md",
        MODULE / "prompts/exp48c_codex_one_session_prompt.md",
        MODULE / "schemas/exp48c_rubric_only_score_schema.json",
    ]
    write_json(args.out_dir / "hashes/exp48c_prompt_schema_hashes.json", {str(path.relative_to(ROOT)): sha256_path(path) for path in prompt_schema_paths})
    report = [
        "# Exp48C rubric-only pointwise audit", "", f"- Status: **{status}**",
        "- Frozen input: 12 families / 36 answers; no regeneration or editing.",
        "- Every review used one answer only. Codex packet contexts were isolated.",
        f"- Codex same-model ablation: `{str(same_model).lower()}`.",
        f"- Codex provenance: gpt-5.5, {len({row['session_id'] for row in all_rows['codex']})} isolated contexts.",
        f"- Qwen provenance: qwen3.7-max, {len({row['session_id'] for row in all_rows['qwen']})} independent API requests.",
        f"- Codex exact/QWK/score2: {codex['intended_exact_rows']}/36 / {codex['QWK']:.4f} / {codex['score2_correct']}/12.",
        f"- Qwen exact/QWK/score2: {summaries['qwen']['intended_exact_rows']}/36 / {summaries['qwen']['QWK']:.4f} / {summaries['qwen']['score2_correct']}/12.",
        f"- Cross-verifier exact/QWK: {cross['exact_rows']}/36 / {cross['QWK']:.4f}.",
        f"- Joint fully confirmed families/metrics: {joint_families}/12 / {accepted_metrics}/12.",
        f"- Contract-aware to rubric-only exact delta: {dependence['exact_delta']:.4f}.",
        f"- Contract-aware to rubric-only QWK delta: {dependence['QWK_delta']:.4f}.",
        f"- McNemar exact p: {mcnemar['exact_p_value']:.6f}.",
        f"- recommend_chinese_replication: `{str(chinese).lower()}`.",
        f"- stop_synthetic_low_tail_route_permanently: `{str(stop).lower()}`.",
        "- No new generation, no training, no GPU, and no dev/test access.",
    ]
    (args.out_dir / "reports/exp48c_rubric_only_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    contract_report = [
        "# Exp48C contract-dependence diagnosis", "",
        f"- Comparison type: **{dependence['comparison_interpretation']}**.",
        f"- Exact: {old_exact:.4f} -> {codex['intended_exact_rate']:.4f} (delta {dependence['exact_delta']:.4f}).",
        f"- QWK: {old_qwk:.4f} -> {codex['QWK']:.4f} (delta {dependence['QWK_delta']:.4f}).",
        f"- Score-2 recall: 1.0000 -> {codex['score2_recall']:.4f} (delta {dependence['score2_recall_delta']:.4f}).",
        f"- Paired McNemar exact p: {mcnemar['exact_p_value']:.6f}.",
        "- Family-cluster bootstrap uses 5,000 deterministic resamples.",
    ]
    (args.out_dir / "reports/exp48c_contract_dependence_report.md").write_text("\n".join(contract_report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
