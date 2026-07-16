"""Validate independent Exp48B reviews and apply preregistered final gates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from .common import read_jsonl, write_csv, write_json, write_jsonl
from .exp48b_common import OUT, PRIVATE, apply_v2_score


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_verifier(verifier: str, path: Path, packet_path: Path) -> tuple[dict, dict, list[str]]:
    packets = {row["family_id"]: row for row in read_jsonl(packet_path)}
    outputs = {row["family_id"]: row for row in read_jsonl(path)}
    errors = []
    evidence_total = evidence_valid = 0
    models, sessions = set(), set()
    for family_id, packet in packets.items():
        output = outputs.get(family_id)
        if output is None:
            errors.append(f"{verifier}/{family_id}:missing_output")
            continue
        provenance = output.get("verifier_provenance", {})
        models.add(str(provenance.get("model_family", "unknown")))
        sessions.add(str(provenance.get("session_id", "unknown")))
        packet_answers = {row["anonymous_answer_id"]: row["text"] for row in packet["answers"]}
        answers = output.get("answers", [])
        if {row.get("anonymous_answer_id") for row in answers} != set(packet_answers):
            errors.append(f"{verifier}/{family_id}:answer_ids_mismatch")
        for answer in answers:
            aid, text = answer.get("anonymous_answer_id"), packet_answers.get(answer.get("anonymous_answer_id"), "")
            contracts = answer.get("contracts", [])
            if {row.get("contract_id") for row in contracts} != {"D2", "D3", "H4"}:
                errors.append(f"{verifier}/{family_id}/{aid}:contract_ids_mismatch")
            for row in contracts:
                status, evidence = row.get("status"), str(row.get("evidence_span", ""))
                if status in {"entailed", "contradicted"}:
                    evidence_total += 1
                    if evidence and evidence in text:
                        evidence_valid += 1
                    else:
                        errors.append(f"{verifier}/{family_id}/{aid}/{row.get('contract_id')}:invalid_evidence")
                elif status in {"absent", "unclear"}:
                    if evidence or not str(row.get("missing_reason", "")).strip():
                        errors.append(f"{verifier}/{family_id}/{aid}/{row.get('contract_id')}:invalid_missing_fields")
                else:
                    errors.append(f"{verifier}/{family_id}/{aid}/{row.get('contract_id')}:invalid_status")
    summary = {
        "verifier": verifier, "packet_families": len(packets), "output_families": len(outputs),
        "complete": len(errors) == 0 and len(outputs) == len(packets),
        "evidence_total": evidence_total, "evidence_valid": evidence_valid,
        "evidence_validity": evidence_valid / max(1, evidence_total),
        "model_families": sorted(models), "session_ids": sorted(sessions),
    }
    return packets, outputs, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--verifier-a", type=Path, default=PRIVATE / "verifier_a/exp48b_verifier_a_outputs.jsonl")
    parser.add_argument("--verifier-b", type=Path, default=PRIVATE / "verifier_b/exp48b_verifier_b_outputs.jsonl")
    args = parser.parse_args()
    families = {row["family_id"]: row for row in read_jsonl(args.out_dir / "private/generated_families/exp48b_constructed_families.jsonl")}
    mappings = read_jsonl(args.out_dir / "private/verifier_packets/exp48b_private_answer_mapping.jsonl")
    map_by = {(row["verifier"], row["family_id"], row["anonymous_answer_id"]): row for row in mappings}
    all_outputs, completion, validation_errors = {}, [], []
    for verifier, path in (("a", args.verifier_a), ("b", args.verifier_b)):
        packets, outputs, errors = load_verifier(verifier, path, args.out_dir / f"private/verifier_packets/exp48b_verifier_{verifier}_packets.jsonl")
        all_outputs[verifier] = outputs
        validation_errors.extend(errors)
        models = sorted({str(row.get("verifier_provenance", {}).get("model_family", "unknown")) for row in outputs.values()})
        sessions = sorted({str(row.get("verifier_provenance", {}).get("session_id", "unknown")) for row in outputs.values()})
        evidence_total = evidence_valid = 0
        for family_id, output in outputs.items():
            packet_texts = {row["anonymous_answer_id"]: row["text"] for row in packets[family_id]["answers"]}
            for answer in output.get("answers", []):
                text = packet_texts.get(answer.get("anonymous_answer_id"), "")
                for contract in answer.get("contracts", []):
                    if contract.get("status") in {"entailed", "contradicted"}:
                        evidence_total += 1
                        evidence_valid += int(bool(contract.get("evidence_span")) and str(contract["evidence_span"]) in text)
        completion.append({"verifier": verifier, "packet_families": len(packets), "output_families": len(outputs), "complete": not errors and len(outputs) == len(packets), "evidence_total": evidence_total, "evidence_valid": evidence_valid, "evidence_validity": evidence_valid / max(1, evidence_total), "model_families": "|".join(models), "session_ids": "|".join(sessions)})
    write_csv(args.out_dir / "tables/exp48b_verifier_completion.csv", completion)
    write_json(args.out_dir / "private/adjudication/exp48b_validation_errors.json", validation_errors)

    states, score_rows = {}, []
    for verifier in ("a", "b"):
        for family_id, output in all_outputs[verifier].items():
            for answer in output["answers"]:
                mapping = map_by[(verifier, family_id, answer["anonymous_answer_id"])]
                state_map = {row["contract_id"]: row["status"] for row in answer["contracts"]}
                states[(verifier, family_id, mapping["answer_id"])] = state_map
                score_rows.append({"verifier": verifier, "family_id": family_id, "answer_id": mapping["answer_id"], "metric": families[family_id]["metric"], "intended_score": int(mapping["intended_score"]), "programmatic_score": apply_v2_score(state_map), "uncertainty": answer["uncertainty"]})
    write_json(args.out_dir / "private/adjudication/exp48b_programmatic_scores.json", score_rows)

    agreement_cells = decisive_cells = agree_cells = decisive_agree = 0
    for family_id, family in families.items():
        for answer in family["answers"]:
            aid = answer["answer_id"]
            for contract_id in ("D2", "D3", "H4"):
                left = states.get(("a", family_id, aid), {}).get(contract_id)
                right = states.get(("b", family_id, aid), {}).get(contract_id)
                if left is not None and right is not None:
                    agreement_cells += 1
                    agree_cells += int(left == right)
                    if contract_id in {"D2", "D3"}:
                        decisive_cells += 1
                        decisive_agree += int(left == right)
    criterion_agreement = agree_cells / max(1, agreement_cells)
    decisive_agreement = decisive_agree / max(1, decisive_cells)
    write_csv(args.out_dir / "tables/exp48b_contract_agreement.csv", [{"scope": "all_D2_D3_H4", "cells": agreement_cells, "agreement": criterion_agreement}, {"scope": "decisive_D2_D3", "cells": decisive_cells, "agreement": decisive_agreement}])

    metrics_by_verifier, confusion_rows = {}, []
    for verifier in ("a", "b"):
        subset = [row for row in score_rows if row["verifier"] == verifier]
        scorable = [row for row in subset if row["programmatic_score"] is not None]
        intended = [row["intended_score"] for row in scorable]
        predicted = [row["programmatic_score"] for row in scorable]
        exact = sum(a == b for a, b in zip(intended, predicted))
        qwk = float(cohen_kappa_score(intended, predicted, labels=[2, 3, 4], weights="quadratic")) if scorable else 0.0
        ordered = 0
        for family_id in families:
            values = {row["intended_score"]: row["programmatic_score"] for row in subset if row["family_id"] == family_id}
            ordered += int(None not in values.values() and values.get(2, 99) < values.get(3, -1) < values.get(4, -1))
        score2_confirmed = sum(row["intended_score"] == 2 and row["programmatic_score"] == 2 for row in subset)
        score2_to_4 = sum(row["intended_score"] == 2 and row["programmatic_score"] == 4 for row in subset)
        metrics_by_verifier[verifier] = {"verifier": verifier, "rows": len(subset), "scorable_rows": len(scorable), "intended_exact_rows": exact, "intended_exact_rate": exact / max(1, len(subset)), "qwk": qwk, "ordered_families": ordered, "score2_confirmed": score2_confirmed, "score2_to_4": score2_to_4}
        counter = Counter((row["intended_score"], row["programmatic_score"]) for row in subset)
        for (gold, pred), count in sorted(counter.items(), key=lambda item: (item[0][0], str(item[0][1]))):
            confusion_rows.append({"verifier": verifier, "intended_score": gold, "programmatic_score": pred if pred is not None else "unscorable", "count": count})
    write_csv(args.out_dir / "tables/exp48b_intended_score_metrics.csv", list(metrics_by_verifier.values()))
    write_csv(args.out_dir / "tables/exp48b_intended_score_confusion.csv", confusion_rows)

    pair_rows = []
    for family_id, family in families.items():
        for answer in family["answers"]:
            aid = answer["answer_id"]
            a = next((row["programmatic_score"] for row in score_rows if row["verifier"] == "a" and row["family_id"] == family_id and row["answer_id"] == aid), None)
            b = next((row["programmatic_score"] for row in score_rows if row["verifier"] == "b" and row["family_id"] == family_id and row["answer_id"] == aid), None)
            pair_rows.append({"family_id": family_id, "answer_id": aid, "intended_score": answer["intended_score"], "score_a": a, "score_b": b, "exact": a is not None and a == b})
    ab_exact = sum(row["exact"] for row in pair_rows) / max(1, len(pair_rows))
    write_csv(args.out_dir / "tables/exp48b_ab_programmatic_agreement.csv", [{"answer_rows": len(pair_rows), "exact_rows": sum(row["exact"] for row in pair_rows), "exact_rate": ab_exact}])

    acceptance = []
    for family_id, family in families.items():
        rows = [row for row in score_rows if row["family_id"] == family_id]
        fully = len(rows) == 6 and all(row["programmatic_score"] == row["intended_score"] for row in rows)
        acceptance.append({"family_id": family_id, "metric": family["metric"], "fully_confirmed": fully})
    write_csv(args.out_dir / "tables/exp48b_family_acceptance.csv", acceptance)
    accepted_families = sum(row["fully_confirmed"] for row in acceptance)
    metrics_accepted = len({row["metric"] for row in acceptance if row["fully_confirmed"]})
    style_rows = csv_rows(args.out_dir / "tables/exp48b_style_probe_metrics.csv")
    style_f1 = float(next(row["macro_f1"] for row in style_rows if row["fold"] == "overall"))
    generation = json.loads((args.out_dir / "decision/exp48b_generation_decision.json").read_text(encoding="utf-8"))
    model_sets = []
    for row in completion:
        model_sets.append(set(str(row["model_families"]).split("|")) - {"", "unknown"})
    cross_family = len(model_sets) == 2 and bool(model_sets[0]) and bool(model_sets[1]) and model_sets[0].isdisjoint(model_sets[1])
    gates = {
        "generation_12_valid": generation["valid_families"] == 12,
        "outside_span_identity_12": generation["outside_span_identity_families"] == 12,
        "verifier_completeness_100pct": all(bool(row["complete"]) for row in completion),
        "evidence_substring_validity_100pct": all(float(row["evidence_validity"]) == 1.0 for row in completion),
        "criterion_agreement_ge_0p90": criterion_agreement >= 0.90,
        "decisive_agreement_ge_0p90": decisive_agreement >= 0.90,
        "ab_programmatic_exact_ge_0p90": ab_exact >= 0.90,
        "intended_exact_ge_33_each": all(row["intended_exact_rows"] >= 33 for row in metrics_by_verifier.values()),
        "qwk_ge_0p85_each": all(row["qwk"] >= 0.85 for row in metrics_by_verifier.values()),
        "fully_confirmed_families_ge_9": accepted_families >= 9,
        "score2_confirmed_ge_10_each": all(row["score2_confirmed"] >= 10 for row in metrics_by_verifier.values()),
        "score2_to_4_zero": all(row["score2_to_4"] == 0 for row in metrics_by_verifier.values()),
        "ordered_families_ge_10_each": all(row["ordered_families"] >= 10 for row in metrics_by_verifier.values()),
        "accepted_metrics_ge_9": metrics_accepted >= 9,
        "style_macro_f1_le_0p45": style_f1 <= 0.45,
        "question_novelty_12": generation["novelty_pass_families"] == 12,
        "no_eval_access": generation["dev_access_count"] == 0 and generation["test_access_count"] == 0,
        "cross_model_family_verification": cross_family,
    }
    quality_without_cross = all(value for key, value in gates.items() if key != "cross_model_family_verification")
    if all(gates.values()):
        status, scale, stop = "EXP48B_QUALIFICATION_GO", True, False
    elif quality_without_cross:
        status, scale, stop = "EXP48B_PROTOCOL_PILOT_ONLY", False, False
    else:
        status, scale, stop = "SYNTHETIC_LOW_TAIL_ROUTE_PERMANENT_STOP", False, True
    decision = {"status": status, "recommend_scale_generation": scale, "permanent_stop": stop, "cross_model_family_verification": cross_family, "accepted_families": accepted_families, "accepted_metrics": metrics_accepted, "criterion_agreement": criterion_agreement, "decisive_agreement": decisive_agreement, "ab_programmatic_exact": ab_exact, "verifier_a": metrics_by_verifier.get("a"), "verifier_b": metrics_by_verifier.get("b"), "style_only_macro_f1": style_f1, "validation_error_count": len(validation_errors), "gates": gates, "dev_access_count": 0, "test_access_count": 0}
    write_json(args.out_dir / "decision/exp48b_qualification_decision.json", decision)
    write_jsonl(args.out_dir / "private/final_silver/exp48b_accepted_families.jsonl", [families[row["family_id"]] for row in acceptance if row["fully_confirmed"]])
    report = ["# Exp48B metric-specific local-edit qualification", "", f"- Final status: **{status}**", f"- Fully confirmed families: {accepted_families}/12", f"- Accepted metrics: {metrics_accepted}/12", f"- Criterion / decisive agreement: {criterion_agreement:.4f} / {decisive_agreement:.4f}", f"- A/B programmatic exact: {ab_exact:.4f}", f"- Verifier A: `{json.dumps(metrics_by_verifier.get('a'), ensure_ascii=False)}`", f"- Verifier B: `{json.dumps(metrics_by_verifier.get('b'), ensure_ascii=False)}`", f"- Cross-model-family verification: {cross_family}", f"- Style-only macro-F1: {style_f1:.4f}", "", "## Gates", ""] + [f"- {key}: **{'PASS' if value else 'FAIL'}**" for key, value in gates.items()] + ["", f"- recommend_scale_generation: `{str(scale).lower()}`", f"- permanent_stop: `{str(stop).lower()}`", "- No training, no GPU, no dev/test access, and no raw private artifacts are public."]
    (args.out_dir / "reports/exp48b_qualification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    if validation_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
