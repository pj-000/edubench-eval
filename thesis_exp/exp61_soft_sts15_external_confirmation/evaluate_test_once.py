"""One-shot sealed-test prediction for the nine frozen epoch-10 checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from thesis_exp.exp61_soft_sts15_external_confirmation import (
    ARTIFACT_ROOT,
    FROZEN_PROTOCOL,
    OUTPUT_ROOT,
    SEEDS,
    VARIANTS,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.contract import (
    SOURCE_LOCK,
    directory_manifest,
    sha256_file,
    verify_model_against_lock,
    verify_source_lock,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.model import (
    ModelConfig,
    load_model_and_tokenizer,
    parameter_sha256,
)
from thesis_exp.exp61_soft_sts15_external_confirmation.runtime import collate
from thesis_exp.exp61_soft_sts15_external_confirmation.sealed_test_data import (
    load_sealed_test_rows,
)


FINAL_TEST_ROOT = OUTPUT_ROOT / "final_test"
ACCESS_MARKER = FINAL_TEST_ROOT / "test_access.json"
PREDICTION_ROOT = FINAL_TEST_ROOT / "predictions"


def checkpoint_dir(variant: str, seed: int) -> Path:
    return ARTIFACT_ROOT / variant / f"seed_{seed}" / "epoch10"


def validate_checkpoint_grid(
    *,
    protocol_sha256: str,
    source_lock_sha256: str,
    model_manifest_sha256: str | None = None,
    mapping_semantic_sha256: str | None = None,
) -> dict[str, Any]:
    grid: dict[str, Any] = {}
    batch_order_by_seed: dict[int, str] = {}
    initialization_by_seed: dict[int, str] = {}
    for variant in VARIANTS:
        grid[variant] = {}
        for seed in SEEDS:
            root = checkpoint_dir(variant, seed)
            metadata_path = root / "exp61_checkpoint.json"
            state_path = root / "dual_head_state_dict.pt"
            if not metadata_path.is_file() or not state_path.is_file():
                raise FileNotFoundError(f"missing frozen epoch-10 checkpoint: {root}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            required = {
                "status": metadata.get("status") == "EXP61_FIXED_EPOCH10_CHECKPOINT",
                "variant": metadata.get("variant") == variant,
                "seed": metadata.get("seed") == seed,
                "epoch": metadata.get("epoch") == 10,
                "state_hash": metadata.get("state_dict_sha256") == sha256_file(state_path),
                "protocol_hash": metadata.get("provenance", {}).get("protocol_sha256") == protocol_sha256,
                "source_lock_hash": metadata.get("provenance", {}).get("source_lock_sha256") == source_lock_sha256,
                "test_access_zero": metadata.get("test_access_count") == 0,
            }
            if model_manifest_sha256 is not None:
                required["model_manifest"] = (
                    metadata.get("provenance", {}).get("model_manifest_sha256")
                    == model_manifest_sha256
                )
            if mapping_semantic_sha256 is not None:
                required["mapping_semantic"] = (
                    metadata.get("provenance", {}).get("mapping_semantic_sha256")
                    == mapping_semantic_sha256
                )
            if not all(required.values()):
                raise RuntimeError(f"checkpoint contract failed for {variant}/seed{seed}: {required}")
            order_hash = str(metadata["provenance"]["train_batch_order_sha256"])
            if seed in batch_order_by_seed and batch_order_by_seed[seed] != order_hash:
                raise RuntimeError(f"three arms used different batch order for seed {seed}")
            batch_order_by_seed[seed] = order_hash
            initial_hash = str(metadata["provenance"]["initial_parameter_sha256"])
            if seed in initialization_by_seed and initialization_by_seed[seed] != initial_hash:
                raise RuntimeError(f"three arms used different initialization for seed {seed}")
            initialization_by_seed[seed] = initial_hash
            grid[variant][str(seed)] = {
                "checkpoint_directory_manifest": directory_manifest(root),
                "checkpoint_metadata_sha256": sha256_file(metadata_path),
                "state_dict_sha256": sha256_file(state_path),
                "initial_parameter_sha256": initial_hash,
                "train_batch_order_sha256": order_hash,
            }
    return grid


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _predict_one(
    *,
    source_model: str,
    variant: str,
    seed: int,
    rows: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    device: Any,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, tokenizer, _ = load_model_and_tokenizer(ModelConfig(source_model))
    if parameter_sha256(model) != checkpoint["initial_parameter_sha256"]:
        raise RuntimeError(f"test evaluator initialization mismatch for {variant}/seed{seed}")
    state_path = checkpoint_dir(variant, seed) / "dual_head_state_dict.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    loader = DataLoader(
        rows,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate(tokenizer, batch, 256),
    )
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            outputs = model(**inputs, labels=labels, aux_route="ordinary_hmsa")
            probabilities = torch.softmax(outputs["hard_logits"].float(), dim=-1).cpu().numpy()
            for row, values in zip(metadata, probabilities):
                predictions.append(
                    {
                        "record_id": row["record_id"],
                        "row_id": row["row_id"],
                        "split": "test",
                        "component_sha256": row["component_sha256"],
                        "human_mean": row["human_mean"],
                        "hard_label": row["label"],
                        "hard_head_probabilities": values.tolist(),
                        "hard_head_expectation": float(values @ np.arange(6)),
                        "hard_head_argmax": int(values.argmax()),
                        "variant": variant,
                        "seed": seed,
                        "checkpoint_state_dict_sha256": checkpoint["state_dict_sha256"],
                    }
                )
    del model, tokenizer, state
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def run_once(source_repo: Path, model_name_or_path: str) -> dict[str, Any]:
    import torch

    source_lock = verify_source_lock(require_formal_authorization=True)
    model_manifest = verify_model_against_lock(Path(model_name_or_path), source_lock)
    protocol_sha256 = sha256_file(FROZEN_PROTOCOL)
    source_lock_sha256 = sha256_file(SOURCE_LOCK)
    checkpoints = validate_checkpoint_grid(
        protocol_sha256=protocol_sha256,
        source_lock_sha256=source_lock_sha256,
        model_manifest_sha256=source_lock["model"]["manifest_sha256"],
        mapping_semantic_sha256=source_lock["mapping_semantic_sha256"],
    )
    if ACCESS_MARKER.exists() or PREDICTION_ROOT.exists():
        raise FileExistsError("Exp61 one-shot test has already been opened or attempted")
    if not torch.cuda.is_available():
        raise RuntimeError("Exp61 one-shot test evaluation requires CUDA")

    # This marker is written before loading the test split. A failed attempt
    # remains an irreversible access record and cannot be silently retried.
    marker: dict[str, Any] = {
        "status": "EXP61_TEST_ACCESS_IN_PROGRESS_NO_RETRY",
        "test_access_count": 1,
        "protocol_sha256": protocol_sha256,
        "source_lock_sha256": source_lock_sha256,
        "model_manifest_sha256": model_manifest["manifest_sha256"],
        "checkpoints": checkpoints,
        "expected_test_rows": 1578,
    }
    _write_json_exclusive(ACCESS_MARKER, marker)
    test_rows = load_sealed_test_rows(source_repo)
    device = torch.device("cuda")
    prediction_hashes: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        prediction_hashes[variant] = {}
        for seed in SEEDS:
            rows = _predict_one(
                source_model=model_name_or_path,
                variant=variant,
                seed=seed,
                rows=test_rows,
                checkpoint=checkpoints[variant][str(seed)],
                device=device,
            )
            if len(rows) != 1578:
                raise RuntimeError("Exp61 test prediction row count mismatch")
            output = PREDICTION_ROOT / variant / f"seed_{seed}.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            prediction_hashes[variant][str(seed)] = sha256_file(output)
    marker.update(
        {
            "status": "EXP61_TEST_ACCESS_COMPLETE",
            "prediction_sha256": prediction_hashes,
            "observed_test_rows_per_run": 1578,
        }
    )
    ACCESS_MARKER.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_repo", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    args = parser.parse_args()
    result = run_once(args.source_repo, args.model_name_or_path)
    print(json.dumps({"status": result["status"], "test_access_count": 1}, indent=2))


if __name__ == "__main__":
    main()
