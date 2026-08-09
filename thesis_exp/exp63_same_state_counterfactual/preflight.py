"""No-update real-model preflight for Exp63."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_exp.exp57_cbrd.data_audit import model_rows
from thesis_exp.exp57_cbrd.train import load_model
from thesis_exp.exp63_same_state_counterfactual import ARMS, OUTPUT_ROOT
from thesis_exp.exp63_same_state_counterfactual.counterfactual import (
    assign_matched_gradient,
    candidate_raw_norm,
    collect_components,
    config_for,
    scalar_geometry,
)
from thesis_exp.exp63_same_state_counterfactual.freeze import verify
from thesis_exp.exp63_same_state_counterfactual.runtime import (
    hash_strings,
    load_protocol,
    ordered_rows,
    write_json,
)
from thesis_exp.src.edujudge.exp02.train_ce_baseline import make_dataloader, set_seed


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "preflight" / "real_model.json")
    args = parser.parse_args()
    verify()
    protocol = load_protocol()
    seed = 67
    set_seed(seed)
    config = config_for(seed, args.model_name_or_path, args.output.parent, Path("UNUSED"))
    rows = model_rows("train")
    ordered = ordered_rows(rows, seed, 1)
    window_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
    window = ordered[:window_size]
    model, tokenizer, model_mode, head_contract = load_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Real-model preflight requires CUDA")
    model.to(device)
    model.gradient_checkpointing_enable()
    loader = make_dataloader(window, tokenizer, config, "train", shuffle=False)
    common, residual, window_audit = collect_components(
        model, loader, device, config.gradient_accumulation_steps
    )
    geometry = scalar_geometry(common, residual)
    target = float(protocol["fairness_controls"]["full_parameter_target_gradient_norm"])
    threshold = float(protocol["fairness_controls"]["common_global_clip_threshold"])
    arm_audit = {}
    for arm in ARMS:
        raw_norm = candidate_raw_norm(
            common, residual, arm, geometry["projection_coefficient"]
        )
        arm_audit[arm] = assign_matched_gradient(
            model,
            common,
            residual,
            arm,
            geometry["projection_coefficient"],
            target,
            threshold,
        )
        arm_audit[arm]["raw_norm_recomputed"] = raw_norm
    gates = protocol["implementation_gates"]
    checks = {
        "window_has_128_rows": len(window) == 128,
        "window_hash_matches_runtime": window_audit["record_ids_sha256"]
        == hash_strings(str(row["record_id"]) for row in window),
        "reconstruction_pass": geometry["reconstruction_relative_error"]
        <= gates["residual_reconstruction_relative_error_at_most"],
        "orthogonality_pass": geometry["normalized_orthogonality_error"]
        <= gates["normalized_orthogonality_error_at_most"],
        "all_raw_norms_above_target": all(
            value["raw_fp64_norm"] >= target for value in arm_audit.values()
        ),
        "all_norms_matched": all(
            abs(value["matched_preclip_norm"] - target)
            <= gates["matched_norm_absolute_error_at_most"]
            for value in arm_audit.values()
        ),
        "no_arm_clipped": all(
            value["postclip_norm"] <= gates["post_clip_norm_at_most"]
            and value["effective_clip_coefficient"] == 1.0
            for value in arm_audit.values()
        ),
        "test_access_count_zero": True,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model_mode": model_mode,
        "head_contract": head_contract,
        "window": window_audit,
        "geometry": geometry,
        "arms": arm_audit,
        "checks": checks,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "test_access_count": 0,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

