"""Create separately shuffled, score-blind Exp48B verifier packets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .common import read_jsonl, stable_id, write_json, write_jsonl
from .exp48b_common import OUT, PRIVATE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=Path, default=PRIVATE / "generated_families/exp48b_constructed_families.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=4802)
    args = parser.parse_args()
    families = read_jsonl(args.families)
    if len(families) != 12:
        raise ValueError(f"Expected 12 valid families, got {len(families)}")
    mappings, hashes = [], {}
    for verifier_index, verifier in enumerate(("a", "b"), 1):
        rng = random.Random(args.seed + verifier_index)
        packets = []
        family_order = list(families)
        rng.shuffle(family_order)
        for family in family_order:
            answers = list(family["answers"])
            rng.shuffle(answers)
            packet_id = stable_id(f"exp48b_{verifier}", family["family_id"])
            anonymous = []
            for answer in answers:
                anonymous_id = stable_id(f"v{verifier}", packet_id, answer["answer_id"])
                anonymous.append({"anonymous_answer_id": anonymous_id, "text": answer["text"]})
                mappings.append({
                    "verifier": verifier, "packet_id": packet_id, "family_id": family["family_id"],
                    "anonymous_answer_id": anonymous_id, "answer_id": answer["answer_id"],
                    "intended_score": answer["intended_score"],
                })
            packets.append({
                "packet_id": packet_id, "family_id": family["family_id"], "metric": family["metric"],
                "language": family["language"], "synthetic_question": family["synthetic_question"],
                "rubric_levels": family["rubric_levels"], "metric_contract": family["metric_contract"],
                "answers": anonymous,
            })
        path = args.out_dir / f"private/verifier_packets/exp48b_verifier_{verifier}_packets.jsonl"
        write_jsonl(path, packets)
        serialized = path.read_text(encoding="utf-8").lower()
        forbidden = [term for term in ('"intended_score"', '"target_score"', '"edit"', '"source_question"', '"base_answer"', '"rubric_grounded_reason"') if term in serialized]
        if forbidden:
            raise AssertionError(f"Blind packet {verifier} leaks fields: {forbidden}")
        hashes[verifier] = {"families": len(packets), "answers": sum(len(row["answers"]) for row in packets)}
    write_jsonl(args.out_dir / "private/verifier_packets/exp48b_private_answer_mapping.jsonl", mappings)
    write_json(args.out_dir / "hashes/exp48b_blind_packet_counts.json", hashes)
    decision = {"status": "EXP48B_BLIND_PACKETS_READY", "verifier_a_families": hashes["a"]["families"], "verifier_b_families": hashes["b"]["families"], "intended_score_leakage": 0, "source_question_leakage": 0, "dev_access_count": 0, "test_access_count": 0}
    write_json(args.out_dir / "decision/exp48b_blind_packet_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
