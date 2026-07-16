"""Apply locked rubric programs to blind verifier criterion states."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from .common import OUT, PRIVATE, apply_score_program, ensure_output_layout, read_jsonl, write_csv, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=Path, default=PRIVATE / "generated_families/exp48a_generated_families.jsonl")
    parser.add_argument("--verifier-a", type=Path, default=PRIVATE / "verifier_a/exp48a_verifier_a_outputs.jsonl")
    parser.add_argument("--verifier-b", type=Path, default=PRIVATE / "verifier_b/exp48a_verifier_b_outputs.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    ensure_output_layout(args.out_dir)
    families = {row["family_id"]: row for row in read_jsonl(args.families)}
    mappings = read_jsonl(args.out_dir / "private/verifier_packets/exp48a_private_answer_mapping.jsonl")
    map_by = {(row["verifier"], row["family_id"], row["anonymous_answer_id"]): row for row in mappings}
    outputs = {
        "a": {row["family_id"]: row for row in read_jsonl(args.verifier_a)},
        "b": {row["family_id"]: row for row in read_jsonl(args.verifier_b)},
    }
    scores = []
    states_by: dict[tuple[str, str, str, str], str] = {}
    uncertainty_by: dict[tuple[str, str, str], str] = {}
    for verifier in ("a", "b"):
        for family_id, row in outputs[verifier].items():
            family = families[family_id]
            for answer in row["answers"]:
                mapping = map_by[(verifier, family_id, answer["anonymous_answer_id"])]
                state_map = {item["criterion_id"]: item["status"] for item in answer["criteria"]}
                score = apply_score_program(family, state_map)
                uncertainty_by[(verifier, family_id, mapping["answer_id"])] = answer["uncertainty"]
                for criterion_id, state in state_map.items():
                    states_by[(verifier, family_id, mapping["answer_id"], criterion_id)] = state
                scores.append({
                    "verifier": verifier, "family_id": family_id, "answer_id": mapping["answer_id"],
                    "intended_score": int(mapping["intended_score"]), "programmatic_score": score,
                    "uncertainty": answer["uncertainty"], "metric": family["metric"], "language": family["language"],
                })
    agreement_rows = []
    family_acceptance = []
    rejection_reasons = Counter()
    private_silver = []
    for family_id, family in families.items():
        answer_ids = [row["answer_id"] for row in family["answers"]]
        criterion_ids = [row["id"] for row in family["criteria"]]
        pairs = [(states_by.get(("a", family_id, aid, cid)), states_by.get(("b", family_id, aid, cid))) for aid in answer_ids for cid in criterion_ids]
        comparable = [(a, b) for a, b in pairs if a is not None and b is not None]
        criterion_agreement = sum(a == b for a, b in comparable) / max(1, len(comparable))
        family_scores = [row for row in scores if row["family_id"] == family_id]
        reasons = []
        if len(family_scores) != 6:
            reasons.append("incomplete_programmatic_scores")
        if any(row["uncertainty"] == "high" for row in family_scores):
            reasons.append("high_uncertainty")
        if criterion_agreement < 0.80:
            reasons.append("criterion_agreement_below_0p80")
        for intended in (2, 3, 5):
            derived = [row["programmatic_score"] for row in family_scores if row["intended_score"] == intended]
            if derived != [intended, intended]:
                reasons.append(f"score{intended}_not_confirmed_by_both")
        by_verifier = defaultdict(dict)
        for row in family_scores:
            by_verifier[row["verifier"]][row["intended_score"]] = row["programmatic_score"]
        if any(not (values.get(2, 99) < values.get(3, -1) < values.get(5, -1)) for values in by_verifier.values()):
            reasons.append("ordering_inconsistent")
        if any(row["intended_score"] == 2 and row["programmatic_score"] in {4, 5} for row in family_scores):
            reasons.append("score2_mapped_to_high")
        accepted = not reasons
        for reason in set(reasons):
            rejection_reasons[reason] += 1
        family_acceptance.append({
            "family_id": family_id, "metric": family["metric"], "language": family["language"],
            "criterion_agreement": criterion_agreement, "accepted": accepted,
            "rejection_reasons": reasons,
        })
        agreement_rows.append({"family_id": family_id, "metric": family["metric"], "language": family["language"], "criterion_agreement": criterion_agreement})
        if accepted:
            private_silver.append(family)
    write_jsonl(args.out_dir / "private/final_silver/exp48a_accepted_silver_families.jsonl", private_silver)
    write_json(args.out_dir / "private/adjudication/exp48a_family_acceptance.json", family_acceptance)
    write_json(args.out_dir / "private/adjudication/exp48a_programmatic_scores.json", scores)

    criterion_summary = [{
        "scope": "overall", "family_count": len(agreement_rows),
        "mean_criterion_agreement": sum(row["criterion_agreement"] for row in agreement_rows) / max(1, len(agreement_rows)),
        "families_ge_0p80": sum(row["criterion_agreement"] >= 0.80 for row in agreement_rows),
    }]
    write_csv(args.out_dir / "tables/exp48a_criterion_agreement.csv", criterion_summary)
    intended = [row["intended_score"] for row in scores if row["programmatic_score"] is not None]
    predicted = [row["programmatic_score"] for row in scores if row["programmatic_score"] is not None]
    exact = sum(a == b for a, b in zip(intended, predicted)) / max(1, len(intended))
    within = sum(abs(a - b) <= 1 for a, b in zip(intended, predicted)) / max(1, len(intended))
    qwk = float(cohen_kappa_score(intended, predicted, weights="quadratic", labels=[1, 2, 3, 4, 5])) if intended else 0.0
    score_summary = [{
        "verifier_answer_rows": len(scores), "scorable_rows": len(predicted), "exact_score_agreement": exact,
        "within_one_agreement": within, "quadratic_weighted_kappa": qwk,
        "intended_score2_to_4_or_5": sum(row["intended_score"] == 2 and row["programmatic_score"] in {4, 5} for row in scores),
    }]
    write_csv(args.out_dir / "tables/exp48a_programmatic_score_agreement.csv", score_summary)
    for dimension in ("metric", "language"):
        output = []
        for value in sorted({row[dimension] for row in family_acceptance}):
            subset = [row for row in family_acceptance if row[dimension] == value]
            output.append({dimension: value, "families": len(subset), "accepted_families": sum(row["accepted"] for row in subset), "acceptance_rate": sum(row["accepted"] for row in subset) / len(subset)})
        write_csv(args.out_dir / f"tables/exp48a_acceptance_by_{dimension}.csv", output)
    write_csv(args.out_dir / "tables/exp48a_rejection_reason_distribution.csv", [{"rejection_reason": key, "family_count": value} for key, value in sorted(rejection_reasons.items())])
    summary = {"accepted_families": len(private_silver), **score_summary[0], **criterion_summary[0]}
    write_json(args.out_dir / "private/adjudication/exp48a_scoring_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
