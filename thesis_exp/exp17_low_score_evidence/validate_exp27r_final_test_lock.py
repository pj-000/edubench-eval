"""Validate the Exp27R phase-1 lock without reading test."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_exp.src.edujudge.exp27r import OUTPUT_DIR, SEEDS, VARIANTS


def validate(out_dir: Path) -> dict[str, object]:
    manifest = json.loads((out_dir / "configs/exp27r_final_lock_manifest.json").read_text(encoding="utf-8"))
    registry = list(csv.DictReader((out_dir / "tables/exp27r_checkpoint_registry.csv").open("r", encoding="utf-8", newline="")))
    selected = [row for row in registry if row["checkpoint_kind"] == "selected"]
    expected = {(variant, str(seed)) for variant in VARIANTS for seed in SEEDS}
    actual = {(row["variant"], row["seed"]) for row in selected}
    passed = (
        manifest["methods_frozen"] is True and manifest["training_frozen"] is True
        and manifest["data_frozen"] is True and manifest["test_access_before_campaign"] == 0
        and len(selected) == 15 and actual == expected
        and all(Path(row["checkpoint_path"]).exists() for row in selected)
    )
    result = {"status": "PASS" if passed else "FAIL", "selected_checkpoints": len(selected),
              "pure_min_sensitivity_enabled": manifest["pure_min_sensitivity_enabled"],
              "test_access_before_campaign": 0, "methods_frozen": manifest["methods_frozen"]}
    if not passed:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    print(json.dumps(validate(parser.parse_args().out_dir), sort_keys=True))

