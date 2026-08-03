"""Real-Qwen BF16 Exp60 preflight with no optimizer and no parameter update.

This entrypoint is intentionally separate from ``train.py``.  It may run while
the protocol is still marked preflight-only, never reads test data, never
constructs an optimizer, and never calls an optimizer step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import model_rows, write_json
from thesis_exp.exp57_cbrd.losses import cbrd_objective
from thesis_exp.exp57_cbrd.train import aggregate_text_hash, load_model
from thesis_exp.exp60_geometry_matched_shuffle import (
    MAPPING_AUDIT_PATH,
    MAPPING_PATH,
    OUTPUT_ROOT,
    PROTOCOL_PATH,
)
from thesis_exp.exp60_geometry_matched_shuffle.contract import (
    require_finite_scalar,
    stable_gpu_identity,
    verify_preflight_source_lock,
)
from thesis_exp.exp60_geometry_matched_shuffle.mapping import (
    mapping_sha256,
    mapping_target_lookup,
)
from thesis_exp.exp60_geometry_matched_shuffle.train import (
    _capture_rng,
    _install_residual_capture_hooks,
    _restore_rng,
    assert_formal_config_matches_protocol,
    compose_geometry_step,
    dataset_contract_sha256,
    file_manifest,
    parameter_snapshot_sha256,
)
from thesis_exp.src.edujudge.exp02.train_ce_baseline import (
    TrainConfig,
    make_dataloader,
    set_seed,
)


WINDOWS = {"full_32": 32, "partial_24": 24}


def parse_args() -> tuple[TrainConfig, int]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Exp60 real-Qwen BF16 no-update preflight")
    parser.add_argument("--seed", required=True, type=int, choices=(47, 48, 49))
    parser.add_argument("--gpu_slot", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--model_name_or_path", default=protocol["model_name_or_path"])
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    fixed = protocol["fixed_training"]
    output = args.output_dir or OUTPUT_ROOT / "real_model_preflight" / f"seed_{args.seed}"
    config = TrainConfig(
        model_name_or_path=args.model_name_or_path,
        data_dir=Path("EXP60_REAL_PREFLIGHT_TRAIN_ONLY"),
        output_dir=output,
        checkpoint_output_dir=output / "NO_CHECKPOINTS_ALLOWED",
        max_length=int(fixed["max_length"]),
        num_train_epochs=float(fixed["epochs"]),
        learning_rate=float(fixed["learning_rate"]),
        weight_decay=float(fixed["weight_decay"]),
        warmup_ratio=float(fixed["warmup_ratio"]),
        per_device_train_batch_size=int(fixed["micro_batch_size"]),
        per_device_eval_batch_size=int(fixed["eval_batch_size"]),
        gradient_accumulation_steps=int(fixed["gradient_accumulation_steps"]),
        max_grad_norm=float(fixed["max_grad_norm"]),
        seed=args.seed,
        bf16="true",
        fp16=False,
        gradient_checkpointing=True,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
        max_train_samples=None,
        max_eval_samples=None,
        eval_only=False,
        checkpoint_dir=None,
        num_workers=0,
        log_steps=1,
        progress_bar=False,
        evaluate_test=False,
    )
    return config, args.gpu_slot


def tensor_digest(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        digest.update(f"{name}\n".encode("utf-8"))
        if gradient is None:
            digest.update(b"NONE\n")
        else:
            value = gradient.detach().cpu().contiguous()
            digest.update(value.view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def clone_gradients_to_cpu(model: Any) -> dict[str, Any]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def restore_gradients(model: Any, gradients: dict[str, Any]) -> None:
    for name, parameter in model.named_parameters():
        if name in gradients:
            parameter.grad = gradients[name].to(
                device=parameter.device, dtype=parameter.dtype
            )
        else:
            parameter.grad = None


def gradient_mismatches_against_cpu(
    model: Any, reference: dict[str, Any]
) -> list[str]:
    import torch

    mismatches: list[str] = []
    for name, parameter in model.named_parameters():
        if name not in reference:
            if parameter.grad is not None:
                mismatches.append(name)
            continue
        if parameter.grad is None or not torch.equal(
            parameter.grad.detach().float().cpu(), reference[name]
        ):
            mismatches.append(name)
    return mismatches


def _batch_values(batch: dict[str, Any], device: Any, target_lookup: dict[str, list[float]]) -> tuple[Any, ...]:
    import torch

    copied = dict(batch)
    metadata = copied.pop("metadata")
    labels = copied.pop("labels").to(device)
    inputs = {key: value.to(device) for key, value in copied.items()}
    aligned = torch.tensor(
        [row["soft_target_5"] for row in metadata], dtype=torch.float32, device=device
    )
    shuffled = torch.tensor(
        [target_lookup[str(row["record_id"])] for row in metadata],
        dtype=torch.float32,
        device=device,
    )
    return metadata, labels, inputs, aligned, shuffled


def _backward_route(
    model: Any,
    batch: dict[str, Any],
    device: Any,
    target_lookup: dict[str, list[float]],
    *,
    route: str,
    loss_scale: float,
) -> tuple[Any, Any]:
    _, labels, inputs, aligned, shuffled = _batch_values(batch, device, target_lookup)
    kwargs: dict[str, Any] = {}
    if route == "shuffled_residual":
        kwargs["residual_targets"] = shuffled
    outputs = model(
        **inputs,
        labels=labels,
        soft_targets=aligned,
        aux_route=route,
        route_loss_scale=loss_scale,
        **kwargs,
    )
    objective = cbrd_objective(
        outputs,
        labels,
        aligned,
        variant=route,
        residual_targets=shuffled if route == "shuffled_residual" else None,
    )
    (objective["optimization_loss"] * loss_scale).backward()
    return (
        outputs["hard_logits"].detach().float().cpu(),
        outputs["aux_logits"].detach().float().cpu(),
        float(objective["reported_aux_soft_ce"].detach().cpu()),
    )


def _diagnostic(
    model: Any,
    batch: dict[str, Any],
    device: Any,
    target_lookup: dict[str, list[float]],
    targets_kind: str,
    buffers: dict[str, Any],
    loss_scale: float,
) -> tuple[Any, Any]:
    _, labels, inputs, aligned, shuffled = _batch_values(batch, device, target_lookup)
    targets = aligned if targets_kind == "aligned" else shuffled
    handles = _install_residual_capture_hooks(model, buffers)
    try:
        outputs = model(
            **inputs,
            labels=labels,
            soft_targets=targets,
            aux_route="residual_only",
            route_loss_scale=loss_scale,
        )
        objective = cbrd_objective(outputs, labels, targets, variant="residual_only")
        (objective["reported_aux_soft_ce"] * loss_scale).backward()
        return (
            outputs["hard_logits"].detach().float().cpu(),
            outputs["aux_logits"].detach().float().cpu(),
            float(objective["reported_aux_soft_ce"].detach().cpu()),
        )
    finally:
        for handle in handles:
            handle.remove()


def _captured_route_difference_error(
    route: dict[str, Any],
    consensus: dict[str, Any],
    captured: dict[str, Any],
) -> dict[str, float]:
    sums = {
        group: {"difference_sq": 0.0, "reference_sq": 0.0}
        for group in ("backbone", "hard_head", "soft_head", "global")
    }
    for name, route_value in route.items():
        expected = route_value - consensus[name]
        actual = (
            captured[name].detach().float().cpu()
            if name in captured
            else expected.new_zeros(expected.shape)
        )
        group = "backbone" if name.startswith("backbone.") else (
            "hard_head" if name.startswith("hard_head.") else "soft_head"
        )
        difference_sq = float((expected - actual).double().square().sum())
        reference_sq = float(expected.double().square().sum())
        for destination in (group, "global"):
            sums[destination]["difference_sq"] += difference_sq
            sums[destination]["reference_sq"] += reference_sq
    return {
        group: math.sqrt(values["difference_sq"])
        / (math.sqrt(values["reference_sq"]) + 1e-12)
        for group, values in sums.items()
    }


def _route_gradients(
    model: Any,
    batches: list[dict[str, Any]],
    states: list[Any],
    final_state: Any,
    device: Any,
    target_lookup: dict[str, list[float]],
    route: str,
    loss_scale: float,
) -> dict[str, Any]:
    import torch

    model.zero_grad(set_to_none=True)
    for batch, state in zip(batches, states):
        _restore_rng(torch, device, state)
        _backward_route(
            model, batch, device, target_lookup, route=route, loss_scale=loss_scale
        )
    _restore_rng(torch, device, final_state)
    return clone_gradients_to_cpu(model)


def audit_window(
    model: Any,
    batches: list[dict[str, Any]],
    device: Any,
    target_lookup: dict[str, list[float]],
    name: str,
) -> dict[str, Any]:
    import torch

    model.train()
    model.zero_grad(set_to_none=True)
    loss_scale = 1.0 / 32.0
    backbone = {
        key: parameter
        for key, parameter in model.named_parameters()
        if key.startswith("backbone.") and parameter.requires_grad
    }
    aligned_buffers = {
        key: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for key, parameter in backbone.items()
    }
    shuffled_buffers = {
        key: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for key, parameter in backbone.items()
    }
    states: list[Any] = []
    main_logits: list[tuple[Any, Any, float]] = []
    started = time.time()
    torch.cuda.synchronize(device)
    main_started = time.time()
    for batch in batches:
        before = _capture_rng(torch, device)
        states.append(before)
        main_logits.append(
            _backward_route(
                model,
                batch,
                device,
                target_lookup,
                route="routed_hmsa",
                loss_scale=loss_scale,
            )
        )
    torch.cuda.synchronize(device)
    main_seconds = time.time() - main_started
    final_state = _capture_rng(torch, device)
    routed_cpu = clone_gradients_to_cpu(model)
    digest_before = tensor_digest(model)

    parity = {"hard": True, "aux": True, "aligned_soft_ce": True}
    torch.cuda.synchronize(device)
    aligned_diagnostic_started = time.time()
    for index, (batch, state) in enumerate(zip(batches, states)):
        _restore_rng(torch, device, state)
        logits = _diagnostic(
            model, batch, device, target_lookup, "aligned", aligned_buffers, loss_scale
        )
        parity["hard"] &= torch.equal(logits[0], main_logits[index][0])
        parity["aux"] &= torch.equal(logits[1], main_logits[index][1])
        parity["aligned_soft_ce"] &= logits[2] == main_logits[index][2]
    _restore_rng(torch, device, final_state)
    torch.cuda.synchronize(device)
    aligned_diagnostic_seconds = time.time() - aligned_diagnostic_started
    digest_after_aligned = tensor_digest(model)
    mismatches_after_aligned = gradient_mismatches_against_cpu(model, routed_cpu)
    torch.cuda.synchronize(device)
    shuffled_diagnostic_started = time.time()
    for index, (batch, state) in enumerate(zip(batches, states)):
        _restore_rng(torch, device, state)
        logits = _diagnostic(
            model, batch, device, target_lookup, "shuffled", shuffled_buffers, loss_scale
        )
        parity["hard"] &= torch.equal(logits[0], main_logits[index][0])
        # The shuffled target changes only the loss target, not forward logits.
        parity["aux"] &= torch.equal(logits[1], main_logits[index][1])
    _restore_rng(torch, device, final_state)
    torch.cuda.synchronize(device)
    shuffled_diagnostic_seconds = time.time() - shuffled_diagnostic_started
    digest_after_shuffled = tensor_digest(model)
    mismatches_after_shuffled = gradient_mismatches_against_cpu(model, routed_cpu)
    rng_restored = all(
        torch.equal(left, right) for left, right in zip(_capture_rng(torch, device), final_state)
    )

    consensus_cpu = _route_gradients(
        model, batches, states, final_state, device, target_lookup, "consensus_only", loss_scale
    )
    aligned_error = _captured_route_difference_error(
        routed_cpu, consensus_cpu, aligned_buffers
    )

    shuffled_route_cpu = _route_gradients(
        model, batches, states, final_state, device, target_lookup, "shuffled_residual", loss_scale
    )
    shuffled_error = _captured_route_difference_error(
        shuffled_route_cpu, consensus_cpu, shuffled_buffers
    )

    restore_gradients(model, routed_cpu)
    torch.cuda.synchronize(device)
    geometry_started = time.time()
    geometry = compose_geometry_step(
        model,
        aligned_buffers,
        shuffled_buffers,
        variant="aligned_orthogonal_only",
        max_norm=1.0,
    )
    torch.cuda.synchronize(device)
    geometry_seconds = time.time() - geometry_started
    model.zero_grad(set_to_none=True)
    return {
        "name": name,
        "microbatches": len(batches),
        "loss_scale": loss_scale,
        "diagnostic_gradient_digest_before": digest_before,
        "diagnostic_gradient_digest_after_aligned": digest_after_aligned,
        "diagnostic_gradient_digest_after_shuffled": digest_after_shuffled,
        "diagnostic_gradients_exactly_unchanged": digest_before
        == digest_after_aligned
        == digest_after_shuffled
        and not mismatches_after_aligned
        and not mismatches_after_shuffled,
        "diagnostic_gradient_mismatches_after_aligned": mismatches_after_aligned,
        "diagnostic_gradient_mismatches_after_shuffled": mismatches_after_shuffled,
        "forward_logit_parity": parity,
        "rng_restored": rng_restored,
        "residual_buffers_have_no_head_entries": all(
            key.startswith("backbone.")
            for key in (*aligned_buffers.keys(), *shuffled_buffers.keys())
        ),
        "aligned_explicit_route_relative_error": aligned_error,
        "shuffled_explicit_route_relative_error": shuffled_error,
        "geometry": geometry,
        "elapsed_seconds": time.time() - started,
        "formal_path_timing": {
            "main_forward_backward_seconds": main_seconds,
            "aligned_diagnostic_seconds": aligned_diagnostic_seconds,
            "shuffled_diagnostic_seconds": shuffled_diagnostic_seconds,
            "geometry_seconds": geometry_seconds,
            "mean_microbatch_main_plus_two_diagnostics_seconds": (
                main_seconds
                + aligned_diagnostic_seconds
                + shuffled_diagnostic_seconds
            )
            / len(batches),
        },
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run(config: TrainConfig, gpu_slot: int) -> dict[str, Any]:
    import torch
    import transformers

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "EXP60_DRAFT_FOR_INDEPENDENT_REVIEW_NOT_AUTHORIZED_FOR_TRAINING":
        raise RuntimeError("Real-model preflight requires the reviewed preflight-only draft status")
    contract_binding = verify_preflight_source_lock()
    expected_slot = int(protocol["formal_runs"]["real_preflight_gpu_schedule"][str(config.seed)])
    if gpu_slot != expected_slot:
        raise RuntimeError(
            f"Exp60 seed {config.seed} preflight requires gpu_slot {expected_slot}, got {gpu_slot}"
        )
    # This checks all formal hyperparameters but deliberately does not call the
    # formal source-lock verifier, which remains unavailable until preflight passes.
    assert_formal_config_matches_protocol(config, "consensus_only", protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp60 real-model preflight requires CUDA")
    device = torch.device("cuda")
    gpu_identity = stable_gpu_identity(torch, device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    mapping_audit = json.loads(MAPPING_AUDIT_PATH.read_text(encoding="utf-8"))
    mapping_rows = [
        json.loads(line) for line in MAPPING_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual_mapping_sha = mapping_sha256(mapping_rows)
    if actual_mapping_sha != mapping_audit["mapping_sha256"]:
        raise RuntimeError("Preflight actual mapping differs from mapping audit")
    if actual_mapping_sha != protocol["mapping"]["canonical_sha256"]:
        raise RuntimeError("Preflight actual mapping differs from protocol")
    if not all(mapping_audit["checks"].values()):
        raise RuntimeError("Preflight mapping integrity audit is incomplete")
    target_lookup = mapping_target_lookup(mapping_rows)
    rows = model_rows("train")
    train_ids = {str(row["record_id"]) for row in rows}
    if set(target_lookup) != train_ids or len(target_lookup) != len(rows):
        raise RuntimeError("Preflight mapping recipient coverage differs from frozen train")
    fixed = protocol["fixed_training"]
    if len(rows) != fixed["train_rows"] or aggregate_text_hash(rows) != fixed["train_text_hash"]:
        raise RuntimeError("Preflight frozen training data mismatch")
    if dataset_contract_sha256(rows) != fixed["train_dataset_contract_sha256"]:
        raise RuntimeError("Preflight frozen training dataset-contract mismatch")
    set_seed(config.seed)
    model_manifest = file_manifest(Path(config.model_name_or_path))
    model, tokenizer, model_mode, head_contract = load_model(config)
    initial_model_hash = parameter_snapshot_sha256(model)
    model.to(device)
    model_loaded_allocated = int(torch.cuda.memory_allocated(device))
    model_loaded_reserved = int(torch.cuda.memory_reserved(device))
    loader = make_dataloader(rows, tokenizer, config, "train", shuffle=True)
    first: list[dict[str, Any]] = []
    tail: deque[dict[str, Any]] = deque(maxlen=24)
    for index, batch in enumerate(loader):
        if index < 32:
            first.append(batch)
        tail.append(batch)
    if len(first) != 32 or len(tail) != 24:
        raise RuntimeError("Could not construct the frozen 32/24 preflight windows")
    residual_bytes = 2 * sum(
        parameter.numel() * 4
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.cuda.reset_peak_memory_stats(device)
    windows = []
    for offset, (name, batches) in enumerate((
        ("full_32", first),
        ("partial_24", list(tail)),
    )):
        set_seed(config.seed * 100 + offset)
        windows.append(audit_window(model, batches, device, target_lookup, name))
    final_model_hash = parameter_snapshot_sha256(model)
    explicit_limit = float(protocol["implementation_gates_before_training"][
        "explicit_route_residual_relative_error_at_most"
    ])
    bf16_limit = float(protocol["implementation_gates_before_training"][
        "bf16_storage_space_relative_error_at_most"
    ])
    all_cosines = [
        float(window["geometry"]["storage_aligned_shuffled_component_cosine"])
        for window in windows
    ]
    all_distances = [
        float(window["geometry"]["storage_aligned_shuffled_component_relative_distance"])
        for window in windows
    ]
    all_activity_ratios = [
        float(window["geometry"]["storage_component_activity_ratio"])
        for window in windows
    ]
    finite_preflight_scalars = [
        *all_cosines,
        *all_distances,
        *all_activity_ratios,
        *(
            float(value)
            for window in windows
            for key in (
                "aligned_explicit_route_relative_error",
                "shuffled_explicit_route_relative_error",
            )
            for value in window[key].values()
        ),
    ]
    for index, value in enumerate(finite_preflight_scalars):
        require_finite_scalar(f"real_preflight_scalar_{index}", value)
    checks = {
        "full_and_partial_windows_present": [row["microbatches"] for row in windows]
        == [32, 24],
        "diagnostic_gradients_exactly_unchanged": all(
            row["diagnostic_gradients_exactly_unchanged"] for row in windows
        ),
        "forward_logits_identical": all(all(row["forward_logit_parity"].values()) for row in windows),
        "rng_restored": all(row["rng_restored"] for row in windows),
        "diagnostic_residual_buffers_exclude_heads": all(
            row["residual_buffers_have_no_head_entries"] for row in windows
        ),
        "explicit_route_residual_equivalence": all(
            max(row[key].values()) <= explicit_limit
            for row in windows
            for key in (
                "aligned_explicit_route_relative_error",
                "shuffled_explicit_route_relative_error",
            )
        ),
        "construction_and_storage_geometry_gates": all(
            float(row["geometry"]["component_norm_relative_error"]) <= 1e-4
            and float(row["geometry"]["preclip_total_norm_relative_error"]) <= 1e-4
            and float(row["geometry"]["clip_coefficient_relative_error"]) <= 1e-4
            and float(row["geometry"]["storage_component_norm_relative_error"])
            <= bf16_limit
            and float(row["geometry"]["storage_preclip_total_norm_relative_error"])
            <= bf16_limit
            and float(row["geometry"]["storage_clip_coefficient_relative_error"])
            <= bf16_limit
            for row in windows
        ),
        "mapping_integrity": all(mapping_audit["checks"].values()),
        "actual_mapping_sha_and_train_coverage_verified": actual_mapping_sha
        == protocol["mapping"]["canonical_sha256"]
        and set(target_lookup) == train_ids,
        "hard_soft_initial_heads_identical_and_storage_independent": (
            head_contract["hard_head_hash"] == head_contract["soft_head_hash"]
            and bool(head_contract["storage_independent"])
        ),
        "no_optimizer_constructed_or_step_taken": True,
        "model_parameters_exactly_unchanged": final_model_hash == initial_model_hash,
        "no_test_access": True,
    }
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    conservative_adam_fp32_bytes = 2 * parameter_count * 4
    peak = max(row["peak_reserved_bytes"] for row in windows)
    projected_peak = peak + conservative_adam_fp32_bytes
    mean_microbatch_seconds = sum(
        row["formal_path_timing"]["mean_microbatch_main_plus_two_diagnostics_seconds"]
        for row in windows
    ) / len(windows)
    mean_geometry_seconds = sum(
        row["formal_path_timing"]["geometry_seconds"] for row in windows
    ) / len(windows)
    estimated_training_seconds_excluding_eval = (
        mean_microbatch_seconds * 664 * 10 + mean_geometry_seconds * 210
    )
    checks["projected_formal_memory_has_10_percent_margin"] = projected_peak <= 0.9 * total_memory
    report = {
        "status": "EXP60_REAL_MODEL_NO_UPDATE_PREFLIGHT_PASS"
        if all(checks.values())
        else "EXP60_REAL_MODEL_NO_UPDATE_PREFLIGHT_FAIL",
        "seed": config.seed,
        "gpu_slot": gpu_slot,
        "gpu_identity": gpu_identity,
        "preflight_contract_binding": contract_binding,
        "model_mode": model_mode,
        "initial_head_contract": head_contract,
        "initial_model_snapshot_sha256": initial_model_hash,
        "final_model_snapshot_sha256": final_model_hash,
        "model_input_manifest": model_manifest,
        "mapping_sha256": actual_mapping_sha,
        "windows": windows,
        "treatment_separation_observation": {
            "storage_component_cosines": all_cosines,
            "storage_component_relative_distances": all_distances,
            "storage_component_activity_ratios": all_activity_ratios,
            "per_seed_rule_is_evaluated_after_all_three_seed_reports": True,
        },
        "memory": {
            "total_bytes": total_memory,
            "model_loaded_allocated_bytes": model_loaded_allocated,
            "model_loaded_reserved_bytes": model_loaded_reserved,
            "two_fp32_residual_buffers_bytes": residual_bytes,
            "conservative_adam_fp32_state_bytes": conservative_adam_fp32_bytes,
            "projected_peak_bytes": projected_peak,
            "projected_free_fraction": (total_memory - projected_peak) / total_memory,
        },
        "runtime_estimate": {
            "mean_formal_microbatch_seconds": mean_microbatch_seconds,
            "mean_geometry_window_seconds": mean_geometry_seconds,
            "estimated_10_epoch_training_seconds_excluding_eval_and_checkpoint_io": estimated_training_seconds_excluding_eval,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": gpu_identity["cuda_visible_devices"],
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_tracked_diff_empty": subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--"], check=False
            ).returncode
            == 0,
        },
        "checks": checks,
        "optimizer_steps": 0,
        "allowed_splits": ["train"],
        "test_access_count": 0,
    }
    write_json(config.output_dir / "real_model_no_update_preflight.json", report)
    return report


def main() -> None:
    config, gpu_slot = parse_args()
    print(json.dumps(run(config, gpu_slot), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
