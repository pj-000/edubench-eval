"""One-shot Exp62 test evaluator; disabled until a separate authorization freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp62_summeval_routing_confirmation import (
    CONFIG_ROOT,
    OUTPUT_ROOT,
    REPO_ROOT,
    SEEDS,
    SPLIT_MANIFEST,
    VARIANTS,
)
from thesis_exp.exp62_summeval_routing_confirmation.analyze_test import paired_analysis
from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import sha256_file
from thesis_exp.exp62_summeval_routing_confirmation.contract import (
    SOURCE_LOCK_PATH,
    verify_source_lock,
)
from thesis_exp.exp62_summeval_routing_confirmation.data import load_test_rows_once
from thesis_exp.exp62_summeval_routing_confirmation.finalize_training import ARTIFACT_ROOT
from thesis_exp.exp62_summeval_routing_confirmation.model import ModelConfig, load_model_and_tokenizer
from thesis_exp.exp62_summeval_routing_confirmation.train import FormalConfig, evaluate_dev, make_loader


TEST_PROTOCOL_PATH = CONFIG_ROOT / "test_protocol.json"
INTEGRITY_PATH = OUTPUT_ROOT / "decision/formal_training_integrity.json"
ACCESS_RECORD = OUTPUT_ROOT / "decision/test_access_record.json"


def verify_test_authorization() -> dict[str, Any]:
    if ACCESS_RECORD.exists():
        raise PermissionError("Exp62 test has already been accessed")
    protocol = json.loads(TEST_PROTOCOL_PATH.read_text(encoding="utf-8"))
    integrity = json.loads(INTEGRITY_PATH.read_text(encoding="utf-8"))
    frozen = protocol.get("frozen_artifacts", {})
    checks = {
        "protocol_status": protocol.get("status") == "EXP62_TEST_PROTOCOL_FROZEN_ONE_TIME_AUTHORIZED",
        "authorized": protocol.get("authorization", {}).get("one_time_test_evaluation") is True,
        "zero_before": protocol.get("test_access_count_before_run") == 0,
        "integrity": integrity.get("status")
        == "EXP62_FORMAL_TRAINING_INTEGRITY_PASS_READY_TO_AUTHORIZE_TEST",
        "twenty_runs": len(integrity.get("runs", [])) == 20,
        "evaluator_hash": frozen.get("evaluate_test_once_sha256")
        == sha256_file(Path(__file__)),
        "analysis_hash": frozen.get("analyze_test_sha256")
        == sha256_file(Path(__file__).with_name("analyze_test.py")),
        "integrity_hash": frozen.get("formal_training_integrity_sha256")
        == sha256_file(INTEGRITY_PATH),
        "training_source_lock_hash": frozen.get("training_source_lock_sha256")
        == sha256_file(SOURCE_LOCK_PATH),
        "split_manifest_hash": frozen.get("split_manifest_sha256")
        == sha256_file(SPLIT_MANIFEST),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Exp62 one-shot test gate failed: {checks}")
    return {"protocol": protocol, "integrity": integrity}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    authorization = verify_test_authorization()
    if not torch.cuda.is_available():
        raise RuntimeError("Exp62 one-shot test evaluation requires CUDA")
    verify_source_lock(Path(args.model_name_or_path))
    test_rows = load_test_rows_once(args.annotations)
    all_metrics: dict[str, dict[str, Any]] = {}
    all_predictions: dict[str, dict[int, list[dict[str, Any]]]] = {
        variant: {} for variant in VARIANTS
    }
    device = torch.device("cuda")
    for variant in VARIANTS:
        all_metrics[variant] = {}
        for seed in SEEDS:
            set_seed(seed)
            model, tokenizer, _ = load_model_and_tokenizer(
                ModelConfig(args.model_name_or_path)
            )
            checkpoint_dir = ARTIFACT_ROOT / variant / f"seed_{seed}" / "epoch10"
            checkpoint = json.loads((checkpoint_dir / "checkpoint.json").read_text())
            state_path = checkpoint_dir / "state_dict.pt"
            if sha256_file(state_path) != checkpoint["state_dict_sha256"]:
                raise RuntimeError(f"Exp62 state hash mismatch: {variant} seed {seed}")
            state = torch.load(state_path, map_location="cpu")
            model.load_state_dict(state, strict=True)
            model.to(device)
            config = FormalConfig(
                annotations=args.annotations,
                model_name_or_path=args.model_name_or_path,
                variant=variant,
                seed=seed,
                output_dir=Path("ONE_SHOT_TEST"),
                checkpoint_dir=checkpoint_dir,
            )
            loader = make_loader(test_rows, tokenizer, config, shuffle=False)
            metrics, predictions = evaluate_dev(model, loader, device)
            all_metrics[variant][str(seed)] = metrics
            all_predictions[variant][seed] = predictions
            del model, state
            torch.cuda.empty_cache()
    analysis = paired_analysis(all_predictions)
    output = {
        "status": "EXP62_ONE_TIME_TEST_COMPLETE",
        "test_access_count": 1,
        "test_protocol_sha256": sha256_file(TEST_PROTOCOL_PATH),
        "training_integrity_sha256": sha256_file(INTEGRITY_PATH),
        "metrics": all_metrics,
        "paired_analysis": analysis,
    }
    output_dir = OUTPUT_ROOT / "test_once"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for variant, seeds in all_predictions.items():
        for seed, rows in seeds.items():
            (output_dir / f"predictions_{variant}_seed_{seed}.json").write_text(
                json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    ACCESS_RECORD.write_text(
        json.dumps(
            {
                "status": "EXP62_TEST_ACCESSED_ONCE",
                "test_access_count": 1,
                "test_results_sha256": sha256_file(output_dir / "test_results.json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
