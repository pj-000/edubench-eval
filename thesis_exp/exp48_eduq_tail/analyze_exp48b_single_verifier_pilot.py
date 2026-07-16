"""Analyze the user-requested single-verifier Exp48B protocol pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from .common import read_jsonl, write_csv, write_json
from .exp48b_common import OUT, PRIVATE, apply_v2_score
from .run_exp48b_verifier_api import validate_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--verifier", type=Path, default=PRIVATE / "verifier_a/exp48b_verifier_a_outputs.jsonl")
    args = parser.parse_args()
    packets = {row["family_id"]: row for row in read_jsonl(args.out_dir / "private/verifier_packets/exp48b_verifier_a_packets.jsonl")}
    outputs = {row["family_id"]: row for row in read_jsonl(args.verifier)}
    mappings = {(row["family_id"], row["anonymous_answer_id"]): row for row in read_jsonl(args.out_dir / "private/verifier_packets/exp48b_private_answer_mapping.jsonl") if row["verifier"] == "a"}
    families = {row["family_id"]: row for row in read_jsonl(args.out_dir / "private/generated_families/exp48b_constructed_families.jsonl")}
    errors, rows = [], []
    evidence_total = evidence_valid = 0
    model_families, session_ids = set(), set()
    for family_id, packet in packets.items():
        output = outputs.get(family_id)
        if output is None:
            errors.append(f"{family_id}:missing_output")
            continue
        provenance = output.get("verifier_provenance", {})
        model, session = str(provenance.get("model_family", "unknown")), str(provenance.get("session_id", "unknown"))
        model_families.add(model)
        session_ids.add(session)
        errors.extend(f"{family_id}:{error}" for error in validate_output(output, packet, "a", model, session))
        packet_texts = {row["anonymous_answer_id"]: row["text"] for row in packet["answers"]}
        for answer in output.get("answers", []):
            mapping = mappings.get((family_id, answer.get("anonymous_answer_id")))
            if mapping is None:
                errors.append(f"{family_id}:missing_private_mapping")
                continue
            state_map = {row["contract_id"]: row["status"] for row in answer.get("contracts", [])}
            for contract in answer.get("contracts", []):
                if contract.get("status") in {"entailed", "contradicted"}:
                    evidence_total += 1
                    evidence = str(contract.get("evidence_span", ""))
                    valid = bool(evidence) and evidence in packet_texts.get(answer.get("anonymous_answer_id"), "")
                    evidence_valid += int(valid)
            rows.append({"family_id": family_id, "metric": families[family_id]["metric"], "answer_id": mapping["answer_id"], "intended_score": int(mapping["intended_score"]), "programmatic_score": apply_v2_score(state_map), "uncertainty": answer.get("uncertainty")})
    scorable = [row for row in rows if row["programmatic_score"] is not None]
    intended = [row["intended_score"] for row in scorable]
    predicted = [row["programmatic_score"] for row in scorable]
    exact_rows = sum(left == right for left, right in zip(intended, predicted))
    qwk = float(cohen_kappa_score(intended, predicted, labels=[2, 3, 4], weights="quadratic")) if scorable else 0.0
    score2_confirmed = sum(row["intended_score"] == 2 and row["programmatic_score"] == 2 for row in rows)
    score2_to_4 = sum(row["intended_score"] == 2 and row["programmatic_score"] == 4 for row in rows)
    accepted = []
    ordered = 0
    for family_id, family in families.items():
        subset = [row for row in rows if row["family_id"] == family_id]
        values = {row["intended_score"]: row["programmatic_score"] for row in subset}
        fully = len(subset) == 3 and all(row["programmatic_score"] == row["intended_score"] for row in subset)
        is_ordered = len(values) == 3 and None not in values.values() and values[2] < values[3] < values[4]
        ordered += int(is_ordered)
        accepted.append({"family_id": family_id, "metric": family["metric"], "fully_confirmed": fully, "ordered": is_ordered})
    accepted_families = sum(row["fully_confirmed"] for row in accepted)
    accepted_metrics = len({row["metric"] for row in accepted if row["fully_confirmed"]})
    style_path = args.out_dir / "tables/exp48b_style_probe_metrics.csv"
    with style_path.open(encoding="utf-8", newline="") as handle:
        style_rows = list(csv.DictReader(handle))
    style_f1 = float(next(row["macro_f1"] for row in style_rows if row["fold"] == "overall"))
    generation = json.loads((args.out_dir / "decision/exp48b_generation_decision.json").read_text(encoding="utf-8"))
    completion = len(outputs) == 12 and len(rows) == 36 and not errors
    evidence_validity = evidence_valid / max(1, evidence_total)
    gates = {
        "generation_12_valid": generation["valid_families"] == 12,
        "outside_span_identity_12": generation["outside_span_identity_families"] == 12,
        "single_verifier_complete": completion,
        "evidence_substring_validity_100pct": evidence_validity == 1.0,
        "intended_exact_ge_33": exact_rows >= 33,
        "qwk_ge_0p85": qwk >= 0.85,
        "fully_confirmed_families_ge_9": accepted_families >= 9,
        "score2_confirmed_ge_10": score2_confirmed >= 10,
        "score2_to_4_zero": score2_to_4 == 0,
        "ordered_families_ge_10": ordered >= 10,
        "accepted_metrics_ge_9": accepted_metrics >= 9,
        "style_macro_f1_le_0p45": style_f1 <= 0.45,
        "question_novelty_12": generation["novelty_pass_families"] == 12,
        "no_eval_access": generation["dev_access_count"] == 0 and generation["test_access_count"] == 0,
    }
    signal = all(gates.values())
    decision = {
        "status": "EXP48B_SINGLE_VERIFIER_PILOT_SIGNAL" if signal else "EXP48B_SINGLE_VERIFIER_PILOT_NO_GO",
        "formal_qualification_complete": False,
        "missing_requirement": "second independent cross-model-family verifier",
        "recommend_scale_generation": False,
        "generator_and_verifier_separate_contexts": True,
        "same_platform_shared_bias_possible": True,
        "language_distribution": generation.get("language_distribution", {}),
        "single_verifier_model_families": sorted(model_families),
        "single_verifier_session_ids": sorted(session_ids),
        "rows": len(rows), "scorable_rows": len(scorable), "intended_exact_rows": exact_rows,
        "intended_exact_rate": exact_rows / max(1, len(rows)), "qwk": qwk,
        "score2_confirmed": score2_confirmed, "score2_to_4": score2_to_4,
        "ordered_families": ordered, "fully_confirmed_families": accepted_families,
        "accepted_metrics": accepted_metrics, "evidence_validity": evidence_validity,
        "style_only_macro_f1": style_f1, "validation_errors": errors,
        "gates": gates, "dev_access_count": 0, "test_access_count": 0,
    }
    write_json(args.out_dir / "decision/exp48b_single_verifier_pilot_decision.json", decision)
    write_csv(args.out_dir / "tables/exp48b_single_verifier_score_metrics.csv", [{key: decision[key] for key in ("rows", "scorable_rows", "intended_exact_rows", "intended_exact_rate", "qwk", "score2_confirmed", "score2_to_4", "ordered_families", "fully_confirmed_families", "accepted_metrics", "evidence_validity", "style_only_macro_f1")}])
    write_csv(args.out_dir / "tables/exp48b_single_verifier_family_acceptance.csv", accepted)
    report = ["# Exp48B single-verifier protocol pilot", "", f"- Status: **{decision['status']}**", "- Formal qualification complete: **False**", "- Missing preregistered requirement: second independent cross-model-family verifier.", "- Generator and verifier used separate Codex contexts, but shared-platform/model-family bias remains possible.", f"- Language distribution: `{json.dumps(decision['language_distribution'], ensure_ascii=False)}`; cross-language reliability was not tested.", f"- Intended exact: {exact_rows}/36 ({decision['intended_exact_rate']:.4f})", f"- QWK: {qwk:.4f}", f"- Fully confirmed / ordered families: {accepted_families}/12 / {ordered}/12", f"- Score-2 confirmed / score2-to-4: {score2_confirmed}/12 / {score2_to_4}", f"- Accepted metrics: {accepted_metrics}/12", f"- Exact evidence substring validity: {evidence_validity:.4f}", f"- Style-only macro-F1: {style_f1:.4f}", "", "## Pilot gates", ""] + [f"- {key}: **{'PASS' if value else 'FAIL'}**" for key, value in gates.items()] + ["", "- `recommend_scale_generation=false` regardless of these single-verifier results.", "- No training, no GPU, no dev/test access."]
    (args.out_dir / "reports/exp48b_single_verifier_pilot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
