"""Audit all nine train-only LR=5e-6 SORC-DPO follow-up runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import sha256_file
from thesis_exp.exp54_rar_sft.audit_sorc_dpo_formal_results import (
    _adapter_spec,
)
from thesis_exp.exp54_rar_sft.run_user_sorc_dpo_lr_followup_smoke import (
    ARMS,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_formal import (
    DEFAULT_FROZEN_TRAINING_LOCK,
    DEFAULT_TRAINING_CONFIG,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_lr_followup import (
    MULTISEED_OVERLAY_PATH,
    OUTPUT_ROOT,
    verify_all_smokes,
    verify_multiseed_overlay,
)


SEEDS = (42, 43, 44)
EXPECTED_RUNS = {(arm, seed) for arm in ARMS for seed in SEEDS}
DEFAULT_REPORT = OUTPUT_ROOT.parent / "training_audit_report.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit() -> dict[str, Any]:
    verify_multiseed_overlay()
    smoke_bindings = verify_all_smokes()
    common_spec = None
    adapter_hashes = set()
    runs = {}
    for arm, seed in sorted(EXPECTED_RUNS):
        run_dir = OUTPUT_ROOT / arm.lower() / f"seed_{seed}"
        paths = {
            "progress": run_dir / "progress.json",
            "result": run_dir / "result.json",
            "adapter_config": run_dir / "adapter/adapter_config.json",
            "adapter_model": run_dir / "adapter/adapter_model.safetensors",
        }
        if any(not path.is_file() or path.is_symlink() for path in paths.values()):
            raise ValueError(f"{arm}/{seed}: missing or invalid artifact")
        result = _read_json(paths["result"])
        progress = _read_json(paths["progress"])
        exact = {
            "schema_version": "exp54-sorc-dpo-lr5e6-multiseed-result-v1",
            "status": "SORC_DPO_FORMAL_TRAINING_COMPLETE",
            "arm": arm,
            "seed": seed,
            "preference_epochs": 1,
            "optimizer_steps": 27,
            "learning_rate": 5e-6,
            "physical_micro_batch_pairs": 4,
            "protocol_overlay_sha256": sha256_file(
                MULTISEED_OVERLAY_PATH
            ),
            "dev_accessed": False,
            "test_accessed": False,
        }
        for field, expected in exact.items():
            if result.get(field) != expected:
                raise ValueError(f"{arm}/{seed}: result {field} differs")
        if (
            progress.get("status") != "COMPLETE"
            or progress.get("optimizer_step") != 27
            or progress.get("optimizer_steps") != 27
            or progress.get("overall_percent") != 100.0
            or progress.get("dev_accessed") is not False
            or progress.get("test_accessed") is not False
        ):
            raise ValueError(f"{arm}/{seed}: progress differs")
        losses = [float(value) for value in result.get("losses") or []]
        gradients = [
            float(value)
            for value in result.get("gradient_norms_before_clip") or []
        ]
        if (
            len(losses) != 27
            or len(gradients) != 27
            or not all(math.isfinite(value) for value in losses)
            or not all(math.isfinite(value) and value > 0 for value in gradients)
        ):
            raise ValueError(f"{arm}/{seed}: loss or gradient trace differs")
        if (
            result.get("training_config_sha256")
            != sha256_file(DEFAULT_TRAINING_CONFIG)
            or result.get("frozen_training_lock_sha256")
            != sha256_file(DEFAULT_FROZEN_TRAINING_LOCK)
            or result.get("output_adapter_config_sha256")
            != sha256_file(paths["adapter_config"])
            or result.get("output_adapter_model_sha256")
            != sha256_file(paths["adapter_model"])
            or result.get("smoke_bindings") != smoke_bindings
        ):
            raise ValueError(f"{arm}/{seed}: artifact binding differs")
        spec = _adapter_spec(paths["adapter_model"])
        if (
            spec["tensor_count"] != 504
            or spec["total_parameters"] != 33030144
            or spec["dtypes"] != ["torch.float32"]
        ):
            raise ValueError(f"{arm}/{seed}: adapter tensor spec differs")
        if common_spec is None:
            common_spec = spec
        elif spec != common_spec:
            raise ValueError(f"{arm}/{seed}: adapter shape spec differs")
        adapter_hash = sha256_file(paths["adapter_model"])
        if adapter_hash in adapter_hashes:
            raise ValueError(f"{arm}/{seed}: duplicate adapter bytes")
        adapter_hashes.add(adapter_hash)
        runs[f"{arm}/seed_{seed}"] = {
            "status": "PASS",
            "optimizer_steps": 27,
            "learning_rate": 5e-6,
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "loss_trace_finite": True,
            "gradient_trace_finite_positive": True,
            "adapter_model_sha256": adapter_hash,
            "result_sha256": sha256_file(paths["result"]),
        }
    assert common_spec is not None
    return {
        "schema_version": "exp54-sorc-dpo-lr5e6-training-audit-v1",
        "status": "SORC_DPO_LR5E6_TRAINING_AUDIT_PASS",
        "run_count": len(runs),
        "all_runs_complete": True,
        "all_runs_have_27_optimizer_steps": True,
        "all_learning_rates_equal_5e6": True,
        "all_adapter_hashes_unique": True,
        "common_adapter_spec": common_spec,
        "runs": runs,
        "source_hashes": {
            "auditor": sha256_file(Path(__file__)),
            "training_config": sha256_file(DEFAULT_TRAINING_CONFIG),
            "frozen_training_lock": sha256_file(
                DEFAULT_FROZEN_TRAINING_LOCK
            ),
            "multiseed_overlay": sha256_file(MULTISEED_OVERLAY_PATH),
        },
        "dev_accessed": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = audit()
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_count": report["run_count"],
                "all_adapter_hashes_unique": report[
                    "all_adapter_hashes_unique"
                ],
                "report_sha256": sha256_file(args.output),
                "dev_accessed": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
