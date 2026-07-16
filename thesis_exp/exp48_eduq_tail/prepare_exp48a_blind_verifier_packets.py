"""Create independently shuffled, score-blind packets for two verifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .common import OUT, PRIVATE, ensure_output_layout, read_jsonl, stable_id, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=Path, default=PRIVATE / "generated_families/exp48a_generated_families.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    ensure_output_layout(args.out_dir)
    decision = json.loads((args.out_dir / "decision/exp48a_generation_decision.json").read_text(encoding="utf-8"))
    if decision.get("status") != "GENERATION_VALID":
        raise RuntimeError("Generation must pass validation before blind packet construction")
    families = read_jsonl(args.families)
    mappings = []
    hashes = {}
    for verifier_index, verifier in enumerate(("a", "b"), 1):
        packets = []
        for family in families:
            shuffled = list(family["answers"])
            random.Random(4800 + verifier_index * 1000 + int(hashlib.sha1(str(family["family_id"]).encode()).hexdigest()[:8], 16)).shuffle(shuffled)
            answers = []
            for position, answer in enumerate(shuffled, 1):
                anonymous_id = stable_id(f"v{verifier}", family["family_id"], position)
                answers.append({"anonymous_answer_id": anonymous_id, "text": answer["text"]})
                mappings.append({
                    "verifier": verifier, "family_id": family["family_id"], "anonymous_answer_id": anonymous_id,
                    "answer_id": answer["answer_id"], "intended_score": answer["intended_score"],
                })
            packets.append({
                "packet_id": stable_id(f"packet_{verifier}", family["family_id"]),
                "family_id": family["family_id"], "synthetic_question": family["synthetic_question"],
                "metric": family["metric"], "language": family["language"], "criteria": family["criteria"],
                "score_program": family["score_program"], "answers": answers,
            })
        path = args.out_dir / f"private/verifier_packets/exp48a_verifier_{verifier}_packets.jsonl"
        write_jsonl(path, packets)
        hashes[f"verifier_{verifier}_packets_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_jsonl(args.out_dir / "private/verifier_packets/exp48a_private_answer_mapping.jsonl", mappings)
    write_json(args.out_dir / "hashes/exp48a_verifier_packet_hashes.json", hashes)
    print(json.dumps({"status": "BLIND_PACKETS_PREPARED", "families_per_verifier": len(families), "mapping_rows": len(mappings)}, sort_keys=True))


if __name__ == "__main__":
    main()
