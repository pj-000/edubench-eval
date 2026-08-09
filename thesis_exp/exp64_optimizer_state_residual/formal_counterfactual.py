"""Frozen train/dev-only optimizer-aware counterfactual audit for Exp64."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from thesis_exp.exp62_summeval_routing_confirmation.audit_dataset import (
    EXPECTED_ANNOTATION_SHA256,
    sha256_file,
)
from thesis_exp.exp62_summeval_routing_confirmation.data import load_model_rows
from thesis_exp.exp62_summeval_routing_confirmation.model import (
    ModelConfig,
    load_model_and_tokenizer,
)
from thesis_exp.exp62_summeval_routing_confirmation.runtime import collate, objective
from thesis_exp.exp64_optimizer_state_residual import (
    ARMS,
    ARTIFACT_ROOT,
    CONFIG_ROOT,
    OUTPUT_ROOT,
    SEEDS,
    STAGE_EPOCHS,
)
from thesis_exp.exp64_optimizer_state_residual.mechanics import (
    dot,
    exact_adamw_displacement,
    fixed_denominator_attributable_displacement,
    l2_norm,
    shared_scale_from_norms,
    subtract_displacements,
)
from thesis_exp.exp64_optimizer_state_residual.state import optimizer_contract, restore_rng


PROTOCOL_PATH = CONFIG_ROOT / "protocol_draft.json"
PROBE_MANIFEST = OUTPUT_ROOT / "stage0" / "probe_manifest.jsonl"
PRIMARY_CANDIDATES = ("full_residual", "parallel_only", "orthogonal_only")
PROBE_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal Exp64 per-seed counterfactual audit")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--artifact_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_protocol(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    checks = {
        "formal_status": protocol.get("status")
        == "EXP64_PROTOCOL_FROZEN_BEFORE_INDEPENDENT_OUTCOMES",
        "seed": args.seed in tuple(protocol.get("base_trajectory", {}).get("seeds", ())),
        "stages": tuple(
            protocol.get("base_trajectory", {}).get("stage_checkpoints_after_epochs", ())
        )
        == STAGE_EPOCHS,
        "probe_hash": sha256_file(PROBE_MANIFEST)
        == protocol.get("probe", {}).get("manifest_sha256"),
        "annotation_hash": sha256_file(args.annotations) == EXPECTED_ANNOTATION_SHA256,
        "counterfactual_authorized": protocol.get("authorization", {}).get(
            "formal_counterfactual_updates"
        )
        is True,
        "test_forbidden": protocol.get("authorization", {}).get("test_evaluation") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Exp64 formal counterfactual gate failed: {checks}")
    return protocol


def _probe_assignments() -> dict[str, str]:
    assignments: dict[str, str] = {}
    for line in PROBE_MANIFEST.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        record_id = str(item["record_id"])
        if record_id in assignments:
            raise RuntimeError(f"duplicate Exp64 probe record: {record_id}")
        assignments[record_id] = str(item["probe"])
    if len(assignments) != 480:
        raise RuntimeError("Exp64 probe manifest must contain 480 development records")
    return assignments


def _make_loader(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int, batch_size: int
) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate(tokenizer, batch, max_length),
    )


def _collect_components(
    model: Any, loader: Any, device: Any
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Collect complete DRB gradient and the analytic direct residual VJP."""

    import torch

    from thesis_exp.exp62_summeval_routing_confirmation.runtime import (
        accumulate_residual_vjp,
        capture_rng,
        restore_rng as restore_torch_rng,
    )

    named_backbone = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("backbone.") and parameter.requires_grad
    }
    residual_buffers = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for name, parameter in named_backbone.items()
    }
    record_ids: list[str] = []
    model.train()
    model.zero_grad(set_to_none=True)
    for batch in loader:
        metadata = batch.pop("metadata")
        record_ids.extend(str(row["record_id"]) for row in metadata)
        labels = batch.pop("labels").to(device)
        inputs = {name: value.to(device) for name, value in batch.items()}
        targets = torch.tensor(
            [row["soft_target"] for row in metadata], dtype=torch.float32, device=device
        )
        before = capture_rng(torch, device)
        outputs = model(
            **inputs,
            labels=labels,
            soft_targets=targets,
            aux_route="consensus_only",
            route_loss_scale=1.0,
        )
        losses = objective(outputs, labels, targets, "consensus_only")
        losses["optimization_loss"].backward()
        after = capture_rng(torch, device)
        restore_torch_rng(torch, device, before)
        accumulate_residual_vjp(model, inputs, labels, targets, residual_buffers, 1.0)
        restore_torch_rng(torch, device, after)
    common = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    residual = {name: value.detach().cpu().clone() for name, value in residual_buffers.items()}
    model.zero_grad(set_to_none=True)
    del residual_buffers
    torch.cuda.empty_cache()
    return common, residual, record_ids


def _projection_coefficient(common: dict[str, Any], residual: dict[str, Any]) -> float:
    numerator = denominator = 0.0
    for name, residual_value in residual.items():
        common_value = common[name]
        numerator += float((common_value.double() * residual_value.double()).sum())
        denominator += float(common_value.double().square().sum())
    return numerator / denominator if denominator > 0.0 else 0.0


def _residual_component(
    name: str, arm: str, common: Any, residual: Any | None, coefficient: float
) -> Any | None:
    if residual is None or not name.startswith("backbone."):
        return None
    if arm == "blocked":
        return residual.new_zeros(residual.shape)
    if arm == "full_residual":
        return residual
    if arm == "parallel_only":
        return coefficient * common
    if arm == "orthogonal_only":
        return residual - coefficient * common
    if arm == "sign_flipped_residual":
        return -residual
    raise ValueError(arm)


def _candidate_gradient(
    common: dict[str, Any],
    residual: dict[str, Any],
    coefficient: float,
    arm: str,
    scale: float,
) -> dict[str, Any]:
    result = {}
    for name, common_value in common.items():
        component = _residual_component(
            name, arm, common_value, residual.get(name), coefficient
        )
        value = common_value if component is None else common_value + component
        result[name] = value.mul(scale)
    return result


def _residual_gradient(
    common: dict[str, Any],
    residual: dict[str, Any],
    coefficient: float,
    arm: str,
    scale: float,
) -> dict[str, Any]:
    result = {}
    for name, common_value in common.items():
        component = _residual_component(
            name, arm, common_value, residual.get(name), coefficient
        )
        result[name] = (
            common_value.new_zeros(common_value.shape)
            if component is None
            else component.mul(scale)
        )
    return result


def _candidate_norms(
    common: dict[str, Any], residual: dict[str, Any], coefficient: float
) -> dict[str, float]:
    norms = {}
    for arm in ARMS:
        squared = 0.0
        for name, common_value in common.items():
            component = _residual_component(
                name, arm, common_value, residual.get(name), coefficient
            )
            value = common_value if component is None else common_value + component
            squared += float(value.double().square().sum())
        norms[arm] = math.sqrt(max(0.0, squared))
    return norms


def _raw_common_cosine(
    common: dict[str, Any], residual_gradient: dict[str, Any]
) -> float:
    numerator = dot(common, residual_gradient)
    denominator = l2_norm(common) * l2_norm(residual_gradient)
    return numerator / denominator if denominator > 0.0 else 0.0


def _probe_gradient(
    model: Any,
    loader: Any,
    device: Any,
    total_rows: int,
) -> tuple[dict[str, Any], float]:
    import torch

    model.eval()
    model.zero_grad(set_to_none=True)
    loss_sum = 0.0
    seen = 0
    for batch in loader:
        metadata = batch.pop("metadata")
        labels = batch.pop("labels").to(device)
        inputs = {name: value.to(device) for name, value in batch.items()}
        outputs = model(**inputs, aux_route="ordinary_hmsa")
        summed = torch.nn.functional.cross_entropy(
            outputs["hard_logits"].float(), labels, reduction="sum"
        )
        (summed / total_rows).backward()
        loss_sum += float(summed.detach().cpu())
        seen += len(metadata)
    if seen != total_rows:
        raise RuntimeError(f"Exp64 probe row mismatch: {seen} != {total_rows}")
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradients[name] = (
            parameter.grad.detach().float().cpu().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
        )
    model.zero_grad(set_to_none=True)
    return gradients, loss_sum / total_rows


def _hard_loss(model: Any, loader: Any, device: Any, total_rows: int) -> float:
    import torch

    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            outputs = model(**inputs, aux_route="ordinary_hmsa")
            total += float(
                torch.nn.functional.cross_entropy(
                    outputs["hard_logits"].float(), labels, reduction="sum"
                ).cpu()
            )
            seen += len(metadata)
    if seen != total_rows:
        raise RuntimeError("Exp64 hard-loss probe row mismatch")
    return total / total_rows


def _apply_displacement(model: Any, displacement: dict[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.add_(displacement[name].to(parameter.device, parameter.dtype))


def _single_learning_rate(groups: dict[str, dict[str, Any]]) -> float:
    values = {float(group["lr"]) for group in groups.values()}
    if len(values) != 1:
        raise RuntimeError(f"Exp64 expected one current learning rate, found {values}")
    return values.pop()


def _verify_optimizer_groups(
    groups: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> None:
    frozen = protocol["optimizer"]
    expected = {
        "betas": tuple(float(value) for value in frozen["betas"]),
        "eps": float(frozen["epsilon"]),
        "weight_decay": float(protocol["training"]["weight_decay"]),
        "amsgrad": bool(frozen["amsgrad"]),
        "maximize": bool(frozen["maximize"]),
        "foreach": bool(frozen["foreach"]),
        "fused": bool(frozen["fused"]),
        "capturable": bool(frozen["capturable"]),
    }
    for name, group in groups.items():
        actual = {
            "betas": tuple(float(value) for value in group["betas"]),
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            "amsgrad": bool(group.get("amsgrad", False)),
            "maximize": bool(group.get("maximize", False)),
            "foreach": bool(group.get("foreach", False)),
            "fused": bool(group.get("fused", False)),
            "capturable": bool(group.get("capturable", False)),
        }
        if actual != expected:
            raise RuntimeError(f"Exp64 optimizer setting mismatch for {name}: {actual}")


def _run_stage(
    *,
    checkpoint: Path,
    stage: int,
    seed: int,
    model: Any,
    tokenizer: Any,
    train_by_id: dict[str, dict[str, Any]],
    probe_rows: dict[str, list[dict[str, Any]]],
    device: Any,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from thesis_exp.exp64_optimizer_state_residual.source_lock import LOCK_PATH

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checks = {
        "format": payload.get("format") == "EXP64_COMPLETE_DRB_STAGE_STATE_V1",
        "seed": payload.get("seed") == seed,
        "stage": payload.get("epoch") == stage,
        "protocol": payload.get("protocol_sha256") == sha256_file(PROTOCOL_PATH),
        "source_lock": payload.get("source_lock_sha256") == sha256_file(LOCK_PATH),
        "test_zero": payload.get("test_access_count") == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Exp64 stage-checkpoint contract failed: {checks}")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    expected_ids = [str(value) for value in payload["next_window_record_ids"]]
    window_rows = [train_by_id[record_id] for record_id in expected_ids]
    window_loader = _make_loader(
        window_rows,
        tokenizer,
        int(protocol["training"]["max_length"]),
        int(protocol["training"]["micro_batch_size"]),
    )
    restore_rng(torch, device, payload["rng"])
    common, residual, observed_ids = _collect_components(model, window_loader, device)
    if observed_ids != expected_ids:
        raise RuntimeError("Exp64 counterfactual next-window mismatch")
    coefficient = _projection_coefficient(common, residual)
    raw_norms = _candidate_norms(common, residual, coefficient)
    scale = shared_scale_from_norms(raw_norms, target_norm=0.95)
    scaled_norms = {name: value * scale for name, value in raw_norms.items()}
    if max(scaled_norms.values()) > 0.9500005:
        raise RuntimeError("Exp64 shared scaling failed")

    model.load_state_dict(payload["model"], strict=True)
    probe_loaders = {
        key: _make_loader(
            rows,
            tokenizer,
            int(protocol["training"]["max_length"]),
            PROBE_BATCH_SIZE,
        )
        for key, rows in probe_rows.items()
    }
    probe_gradients = {}
    base_probe_losses = {}
    for key in ("A", "B"):
        model.load_state_dict(payload["model"], strict=True)
        probe_gradients[key], base_probe_losses[key] = _probe_gradient(
            model, probe_loaders[key], device, len(probe_rows[key])
        )

    model.load_state_dict(payload["model"], strict=True)
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    states, groups = optimizer_contract(
        model, payload["optimizer"], payload.get("optimizer_parameter_names")
    )
    _verify_optimizer_groups(groups, protocol)
    learning_rate = _single_learning_rate(groups)
    blocked_gradient = _candidate_gradient(common, residual, coefficient, "blocked", scale)
    blocked_step = exact_adamw_displacement(parameters, blocked_gradient, states, groups)
    _apply_displacement(model, blocked_step)
    blocked_losses = {
        key: _hard_loss(model, probe_loaders[key], device, len(probe_rows[key]))
        for key in ("A", "B")
    }

    results: dict[str, Any] = {}
    exact_attributable: dict[str, dict[str, Any]] = {}
    for arm in ARMS[1:]:
        model.load_state_dict(payload["model"], strict=True)
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        gradient = _candidate_gradient(common, residual, coefficient, arm, scale)
        candidate_step = exact_adamw_displacement(parameters, gradient, states, groups)
        attributable = subtract_displacements(candidate_step, blocked_step)
        attributable_cpu = {
            name: value.detach().float().cpu() for name, value in attributable.items()
        }
        residual_gradient = _residual_gradient(common, residual, coefficient, arm, scale)
        raw_displacement = {
            name: value.mul(-learning_rate) for name, value in residual_gradient.items()
        }
        fixed = fixed_denominator_attributable_displacement(
            blocked_gradient, gradient, states, groups
        )
        fixed_cpu = {name: value.detach().float().cpu() for name, value in fixed.items()}
        predictions = {}
        for source, target in (("A", "B"), ("B", "A")):
            predictions[f"{source}_to_{target}"] = {
                "q_exact": dot(probe_gradients[source], attributable_cpu),
                "q_raw_validation": dot(probe_gradients[source], raw_displacement),
                "q_fixed_denominator": dot(probe_gradients[source], fixed_cpu),
            }
        _apply_displacement(model, candidate_step)
        candidate_losses = {
            key: _hard_loss(model, probe_loaders[key], device, len(probe_rows[key]))
            for key in ("A", "B")
        }
        outcomes = {
            "A_to_B": candidate_losses["B"] - blocked_losses["B"],
            "B_to_A": candidate_losses["A"] - blocked_losses["A"],
        }
        exact_norm = l2_norm(attributable_cpu)
        results[arm] = {
            "predictions": predictions,
            "outcomes": outcomes,
            "candidate_probe_losses": candidate_losses,
            "exact_attributable_norm": exact_norm,
            "magnitude_only_score": -exact_norm,
            "fixed_denominator_norm": l2_norm(fixed_cpu),
            "raw_common_cosine": _raw_common_cosine(common, residual_gradient),
        }
        if arm in PRIMARY_CANDIDATES:
            exact_attributable[arm] = attributable_cpu
        del gradient, candidate_step, attributable, residual_gradient, raw_displacement, fixed
        gc.collect()
        torch.cuda.empty_cache()

    interaction = {
        name: exact_attributable["full_residual"][name]
        - exact_attributable["parallel_only"][name]
        - exact_attributable["orthogonal_only"][name]
        for name in exact_attributable["full_residual"]
    }
    interaction_predictions = {
        f"{source}_to_{target}": dot(probe_gradients[source], interaction)
        for source, target in (("A", "B"), ("B", "A"))
    }
    interaction_outcomes = {
        direction: results["full_residual"]["outcomes"][direction]
        - results["parallel_only"]["outcomes"][direction]
        - results["orthogonal_only"]["outcomes"][direction]
        for direction in ("A_to_B", "B_to_A")
    }
    return {
        "status": "EXP64_FORMAL_STAGE_COMPLETE",
        "seed": seed,
        "stage": stage,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "next_window_record_ids": expected_ids,
        "projection_coefficient": coefficient,
        "candidate_raw_norms": raw_norms,
        "shared_scale": scale,
        "candidate_scaled_norms": scaled_norms,
        "learning_rate": learning_rate,
        "base_probe_losses": base_probe_losses,
        "blocked_probe_losses": blocked_losses,
        "arms": results,
        "nonadditivity": {
            "norm": l2_norm(interaction),
            "predictions": interaction_predictions,
            "outcomes": interaction_outcomes,
        },
        "checks": {
            "checkpoint_rng_restored": True,
            "one_shared_scale": True,
            "all_clip_coefficients_one": max(scaled_norms.values()) < 1.0,
            "cross_probe_only": True,
            "probe_batch_size": PROBE_BATCH_SIZE,
            "optimizer_step_calls": 0,
            "test_access_count": 0,
        },
        "test_access_count": 0,
    }


def main() -> None:
    import torch

    args = parse_args()
    protocol = _verify_protocol(args)
    from thesis_exp.exp64_optimizer_state_residual.source_lock import (
        FORMAL_LOCK_PATH,
        verify_formal_source_lock,
    )

    source_lock = verify_formal_source_lock(Path(args.model_name_or_path))
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Exp64 counterfactual requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    train_rows = load_model_rows(args.annotations, "train")
    dev_rows = load_model_rows(args.annotations, "dev")
    train_by_id = {str(row["record_id"]): row for row in train_rows}
    assignments = _probe_assignments()
    probe_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev_rows:
        probe_rows[assignments[str(row["record_id"])]].append(row)
    if {key: len(value) for key, value in probe_rows.items()} != {"A": 256, "B": 224}:
        raise RuntimeError("Exp64 probe partition row counts changed")
    model, tokenizer, head_contract = load_model_and_tokenizer(
        ModelConfig(args.model_name_or_path)
    )
    model.to(device)
    artifact_dir = args.artifact_dir or ARTIFACT_ROOT / f"seed_{args.seed}"
    output_dir = args.output_dir or OUTPUT_ROOT / "formal" / f"seed_{args.seed}"
    stages = []
    for stage in STAGE_EPOCHS:
        result = _run_stage(
            checkpoint=artifact_dir / f"after_epoch_{stage}.pt",
            stage=stage,
            seed=args.seed,
            model=model,
            tokenizer=tokenizer,
            train_by_id=train_by_id,
            probe_rows=dict(probe_rows),
            device=device,
            protocol=protocol,
        )
        _write_json(output_dir / f"after_epoch_{stage}.json", result)
        stages.append(result)
        print(
            f"[exp64:formal:seed{args.seed}] completed stage {stage}; "
            f"Full mean Y="
            f"{sum(result['arms']['full_residual']['outcomes'].values())/2:.8f}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "status": "EXP64_FORMAL_SEED_COMPLETE",
        "seed": args.seed,
        "head_contract": head_contract,
        "stages": stages,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_lock_sha256": sha256_file(FORMAL_LOCK_PATH),
        "model_manifest_sha256": source_lock["model"]["manifest_sha256"],
        "probe_manifest_sha256": sha256_file(PROBE_MANIFEST),
        "test_access_count": 0,
    }
    _write_json(output_dir / "seed_summary.json", summary)


if __name__ == "__main__":
    main()
