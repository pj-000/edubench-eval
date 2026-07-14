"""Create the immutable Exp43 final-test lock after both dev gates pass."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from thesis_exp.exp43_rubimor.common import ROOT, TEST_PATH, sha256_file, stable_hash, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out-dir", type=Path, default=ROOT); parser.add_argument("--test", type=Path, default=TEST_PATH); args = parser.parse_args()
    consumed = args.out_dir / "decision/exp43_final_test_consumed.json"
    if consumed.exists() and json.loads(consumed.read_text(encoding="utf-8")).get("final_test_consumed"):
        raise SystemExit("Exp43 final test was already consumed")
    groupcv = json.loads((args.out_dir / "decision/exp43_groupcv_decision.json").read_text(encoding="utf-8"))
    headline = json.loads((args.out_dir / "decision/exp43_headline_dev_decision.json").read_text(encoding="utf-8"))
    if groupcv["status"] not in {"RUBIMOR_FULL_GROUPCV_GO", "RUBIMOR_OVERALL_GROUPCV_GO"} or headline["status"] != "HEADLINE_DEV_GO":
        raise SystemExit("Final test remains sealed because GroupCV/headline gates did not both pass")
    checkpoint_hashes = json.loads((args.out_dir / "hashes/exp43_checkpoint_hashes.json").read_text(encoding="utf-8"))
    code_files = sorted(Path("thesis_exp/exp43_rubimor").glob("*.py"))
    lock = {"git_commit": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(), "code_hashes": {str(path): sha256_file(path) for path in code_files}, "data_hashes": json.loads((args.out_dir/"hashes/exp43_split_hashes.json").read_text(encoding="utf-8")), "metric_mapping_hash": sha256_file(args.out_dir/"configs/exp43_metric_mapping.json"), "checkpoint_hashes": checkpoint_hashes, "variants": ["E0","E3","E5","E6","E6N"], "seeds": [42,43,44], "success_criteria": {"groupcv": groupcv["status"], "headline": headline["status"]}, "final_test_consumed": False, "test_path": str(args.test), "test_file_hash": sha256_file(args.test)}
    lock["lock_hash"] = stable_hash(lock)
    write_json(args.out_dir / "configs/exp43_final_test_lock.json", lock)
    print(json.dumps({"status":"FINAL_TEST_LOCKED","lock_hash":lock["lock_hash"]},sort_keys=True))


if __name__ == "__main__": main()

