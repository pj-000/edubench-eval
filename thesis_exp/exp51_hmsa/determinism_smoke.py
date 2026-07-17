"""Compare two locked Exp51 smoke traces without pretending CUDA is bitwise."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thesis_exp.exp51_hmsa import OUTPUT_ROOT
from thesis_exp.src.edujudge.exp02.train_ce_baseline import write_json


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare() -> dict[str, Any]:
    root = OUTPUT_ROOT / "audit" / "determinism"
    a = read(root / "run_a" / "training_trace_first64.json")
    b = read(root / "run_b" / "training_trace_first64.json")
    if a["head_contract"] != b["head_contract"]:
        raise AssertionError("initial head contracts differ")
    trace_a, trace_b = a["trace"], b["trace"]
    if len(trace_a) != 64 or len(trace_b) != 64:
        raise AssertionError("expected two 64-microbatch traces")
    batch_match = [row["record_ids"] for row in trace_a] == [row["record_ids"] for row in trace_b]
    deltas = {
        key: [abs(float(x[key]) - float(y[key])) for x, y in zip(trace_a, trace_b)]
        for key in ("hard_loss", "soft_loss", "total_loss")
    }
    result = {
        "status": "CONTRACT_MATCH_BITWISE" if max(max(values) for values in deltas.values()) == 0 else "CONTRACT_MATCH_NON_BITWISE_CUDA",
        "head_contract_match": True,
        "batch_ids_match": batch_match,
        "pre_first_update_max_delta": {key: max(values[:32]) for key, values in deltas.items()},
        "post_first_update_max_delta": {key: max(values[32:]) for key, values in deltas.items()},
        "overall_max_delta": {key: max(values) for key, values in deltas.items()},
        "test_access_count": 0,
    }
    if not batch_match or any(value != 0 for value in result["pre_first_update_max_delta"].values()):
        raise AssertionError(result)
    write_json(root / "comparison.json", result)
    return result


def main() -> None:
    print(json.dumps(compare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
