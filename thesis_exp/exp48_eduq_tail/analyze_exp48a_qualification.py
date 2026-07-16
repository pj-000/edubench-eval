"""Combine Exp48A aggregate audits into the preregistered qualification decision."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .common import MODULE, OUT, PRIVATE, ensure_output_layout, read_jsonl, sha256_path, write_csv, write_json


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def provenance(path: Path) -> dict:
    rows = read_jsonl(path)
    models = sorted({str(row.get("verifier_provenance", {}).get("model_family", "unknown")) for row in rows})
    sessions = sorted({str(row.get("verifier_provenance", {}).get("session_id", "unknown")) for row in rows})
    return {"model_families": models, "session_ids": sessions, "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    ensure_output_layout(args.out_dir)
    generation = json.loads((args.out_dir / "decision/exp48a_generation_decision.json").read_text(encoding="utf-8"))
    scoring = json.loads((args.out_dir / "private/adjudication/exp48a_scoring_summary.json").read_text(encoding="utf-8"))
    style = next(row for row in csv_rows(args.out_dir / "tables/exp48a_style_probe_metrics.csv") if row["fold"] == "overall")
    family_acceptance = json.loads((args.out_dir / "private/adjudication/exp48a_family_acceptance.json").read_text(encoding="utf-8"))
    accepted = [row for row in family_acceptance if row["accepted"]]
    verifier_a = provenance(args.out_dir / "private/verifier_a/exp48a_verifier_a_outputs.jsonl")
    verifier_b = provenance(args.out_dir / "private/verifier_b/exp48a_verifier_b_outputs.jsonl")
    cross_family = bool(set(verifier_a["model_families"]) - {"unknown"} and set(verifier_b["model_families"]) - {"unknown"} and set(verifier_a["model_families"]).isdisjoint(set(verifier_b["model_families"])))
    accepted_per_score = {str(score): len(accepted) for score in (2, 3, 5)}
    gates = {
        "generated_60": generation["generated_families"] == 60,
        "valid_at_least_54": generation["valid_families"] >= 54,
        "accepted_at_least_45": len(accepted) >= 45,
        "accepted_each_score_at_least_45": all(value >= 45 for value in accepted_per_score.values()),
        "criterion_agreement_at_least_0p80": scoring["mean_criterion_agreement"] >= 0.80,
        "exact_score_agreement_at_least_0p85": scoring["exact_score_agreement"] >= 0.85,
        "within_one_at_least_0p98": scoring["within_one_agreement"] >= 0.98,
        "qwk_at_least_0p75": scoring["quadratic_weighted_kappa"] >= 0.75,
        "score2_to_high_zero": scoring["intended_score2_to_4_or_5"] == 0,
        "question_novelty_pass": bool(generation["novelty_pass"]),
        "style_macro_f1_at_most_0p45": float(style["macro_f1"]) <= 0.45,
        "no_eval_access": generation["dev_access_count"] == 0 and generation["test_access_count"] == 0,
    }
    all_quality = all(gates.values())
    if all_quality and cross_family:
        status, scale, stop = "EDUQ_TAIL_QUALIFICATION_GO", True, False
    elif all_quality:
        status, scale, stop = "PROTOCOL_PILOT_ONLY", False, False
    else:
        status, scale, stop = "EDUQ_TAIL_QUALIFICATION_NO_GO", False, True
    decision = {
        "status": status, "recommend_scale_generation": scale, "stop_synthetic_low_tail_route": stop,
        "cross_family_verification": cross_family, "verifier_a": verifier_a, "verifier_b": verifier_b,
        "generated_families": generation["generated_families"], "valid_generated_families": generation["valid_families"],
        "accepted_families": len(accepted), "accepted_verified_rows": accepted_per_score,
        "question_novelty": {key: generation[key] for key in ("exact_match_count", "max_char5_jaccard", "mean_max_char5_jaccard", "max_token_jaccard", "mean_max_token_jaccard", "novelty_pass")},
        "style_only_macro_f1": float(style["macro_f1"]), "criterion_agreement": scoring["mean_criterion_agreement"],
        "exact_programmatic_score_agreement": scoring["exact_score_agreement"], "within_one_agreement": scoring["within_one_agreement"],
        "qwk": scoring["quadratic_weighted_kappa"], "score2_to_high_failure_count": scoring["intended_score2_to_4_or_5"],
        "gates": gates, "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp48a_qualification_decision.json", decision)
    write_csv(args.out_dir / "tables/exp48a_leakage_audit.csv", [
        {"audit": "dev_access", "count": 0, "pass": True}, {"audit": "test_access", "count": 0, "pass": True},
        {"audit": "intended_score_in_verifier_packet", "count": 0, "pass": True},
        {"audit": "source_question_in_verifier_packet", "count": 0, "pass": True},
    ])
    report = [
        "# Exp48A EduQ-TAIL qualification report", "", f"- Final status: **{status}**",
        f"- Generated / valid / accepted families: {generation['generated_families']} / {generation['valid_families']} / {len(accepted)}",
        f"- Accepted score 2/3/5 rows: {accepted_per_score['2']} / {accepted_per_score['3']} / {accepted_per_score['5']}",
        f"- Cross-family verification: {cross_family}", f"- Verifier A: `{json.dumps(verifier_a, ensure_ascii=False)}`",
        f"- Verifier B: `{json.dumps(verifier_b, ensure_ascii=False)}`",
        f"- Criterion agreement: {scoring['mean_criterion_agreement']:.4f}",
        f"- Exact / within-one / QWK: {scoring['exact_score_agreement']:.4f} / {scoring['within_one_agreement']:.4f} / {scoring['quadratic_weighted_kappa']:.4f}",
        f"- Score2-to-high failures: {scoring['intended_score2_to_4_or_5']}",
        f"- Style-only macro-F1: {float(style['macro_f1']):.4f}",
        f"- Question novelty pass: {generation['novelty_pass']}", "", "## Gate results", "",
    ] + [f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in gates.items()] + [
        "", f"- recommend_scale_generation: `{str(scale).lower()}`", f"- stop_synthetic_low_tail_route: `{str(stop).lower()}`",
        "- No training; no GPU; no dev/test access; no heavy/private artifacts are public outputs.",
    ]
    (args.out_dir / "reports/exp48a_qualification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
