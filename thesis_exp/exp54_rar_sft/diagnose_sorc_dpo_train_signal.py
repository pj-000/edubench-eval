"""Diagnose whether frozen SORC-DPO adapters learned their train preferences.

This is a train-only, read-only diagnostic.  It compares each trained
P1/P2/P3 adapter with its seed-matched frozen R3 cold start on the exact frozen
preference manifests.  Row-level values stay in a private JSONL file; the
public report contains only grouped aggregate statistics and artifact hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from thesis_exp.exp54_rar_sft import REPO_ROOT
from thesis_exp.exp54_rar_sft.actual_failure_bank import read_jsonl, sha256_file
from thesis_exp.exp54_rar_sft.build_sorc_dpo_training_manifests import (
    DEFAULT_PAIR_LOCK,
    PAIR_FILES,
)
from thesis_exp.exp54_rar_sft.probe_sorc_dpo_capacity import _device_identity
from thesis_exp.exp54_rar_sft.run_sorc_dpo_dev_inference_vllm import (
    _validate_adapter,
)
from thesis_exp.exp54_rar_sft.sorc_dpo_loss import (
    SORCDPOPairCollator,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_formal import (
    DEFAULT_BASE_TRAINING_CONFIGURATION,
    DEFAULT_TRAINING_CONFIG,
    _checkpoint_for_seed,
    load_formal_rows,
)
from thesis_exp.exp54_rar_sft.train_sorc_dpo_smoke import (
    read_json,
    verify_base_model_snapshot,
)


ARMS = ("P1_FIELD_DPO", "P2_SORC_SCORE", "P3_JOINT_SORC")
OUTPUT_ROOT = (
    REPO_ROOT
    / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
    "preference_train_signal_diagnostics"
)
PRIVATE_METRIC_FIELDS = (
    "reference_chosen_logp",
    "reference_rejected_logp",
    "policy_chosen_logp",
    "policy_rejected_logp",
    "reference_raw_margin",
    "policy_raw_margin",
    "chosen_logp_change",
    "rejected_logp_change",
    "preference_contrast",
    "beta_scaled_contrast",
    "risk_adjusted_margin",
)
RATE_FIELDS = (
    "reference_prefers_chosen",
    "policy_prefers_chosen",
    "contrast_positive",
    "offset_satisfied",
    "chosen_logp_increased",
    "rejected_logp_decreased",
)


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_compact_json(row) + "\n")
    os.replace(temporary, path)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability is outside [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate train-signal statistics without row identifiers."""
    if not rows:
        raise ValueError("cannot summarize an empty diagnostic group")
    numeric = {}
    for field in PRIVATE_METRIC_FIELDS:
        values = [float(row[field]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{field} contains a non-finite value")
        numeric[field] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p10": _quantile(values, 0.10),
            "p90": _quantile(values, 0.90),
            "minimum": min(values),
            "maximum": max(values),
        }
    rates = {
        field: statistics.fmean(bool(row[field]) for row in rows)
        for field in RATE_FIELDS
    }
    score_rows = [row for row in rows if row["pair_task"] == "score"]
    score = None
    if score_rows:
        score = {
            "count": len(score_rows),
            "reference_gold_accuracy": statistics.fmean(
                int(row["reference_predicted_score"]) == int(row["gold_label"])
                for row in score_rows
            ),
            "policy_gold_accuracy": statistics.fmean(
                int(row["policy_predicted_score"]) == int(row["gold_label"])
                for row in score_rows
            ),
            "reference_rejected_rate": statistics.fmean(
                int(row["reference_predicted_score"])
                == int(row["rejected_score"])
                for row in score_rows
            ),
            "policy_rejected_rate": statistics.fmean(
                int(row["policy_predicted_score"]) == int(row["rejected_score"])
                for row in score_rows
            ),
            "rejected_to_gold_flip_rate": statistics.fmean(
                int(row["reference_predicted_score"])
                == int(row["rejected_score"])
                and int(row["policy_predicted_score"]) == int(row["gold_label"])
                for row in score_rows
            ),
            "gold_to_rejected_flip_rate": statistics.fmean(
                int(row["reference_predicted_score"]) == int(row["gold_label"])
                and int(row["policy_predicted_score"])
                == int(row["rejected_score"])
                for row in score_rows
            ),
        }
    return {
        "count": len(rows),
        "numeric": numeric,
        "rates": rates,
        "score_decision": score,
    }


def grouped_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "pair_task": lambda row: str(row["pair_task"]),
        "pair_type": lambda row: str(row["pair_type"]),
        "pair_source": lambda row: str(row["pair_source"]),
        "gold_label": lambda row: str(row["gold_label"]),
        "metric_id": lambda row: str(row["metric_id"]),
        "language": lambda row: str(row["language"]),
    }
    output = {"overall": summarize_rows(rows), "by": {}}
    for name, key_function in dimensions.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[key_function(row)].append(row)
        output["by"][name] = {
            key: summarize_rows(group)
            for key, group in sorted(groups.items())
        }
    return output


def _source_metadata() -> dict[str, dict[str, Any]]:
    pair_lock = read_json(DEFAULT_PAIR_LOCK)
    if (
        pair_lock.get("status")
        != "PAIR_PROTOCOL_FROZEN_IMPLEMENTATION_ALLOWED"
        or pair_lock.get("pair_protocol_frozen") is not True
        or pair_lock.get("dev_accessed") is not False
        or pair_lock.get("test_accessed") is not False
    ):
        raise ValueError("frozen pair-protocol lock differs")
    expected_hashes = dict(pair_lock["private_artifact_hashes"])
    output = {}
    for name, path in PAIR_FILES.items():
        if sha256_file(path) != str(expected_hashes[name]):
            raise ValueError(f"{name}: frozen pair source hash differs")
        for row in read_jsonl(path):
            pair_id = str(row["pair_hash"])
            if pair_id in output:
                raise ValueError("duplicate pair hash across frozen pair sources")
            output[pair_id] = {
                "pair_task": (
                    "rationale"
                    if str(row["pair_type"]) == "rationale_alignment"
                    else "score"
                ),
                "pair_type": str(row["pair_type"]),
                "pair_source": str(row["pair_source"]),
                "gold_label": int(row["gold_label"]),
                "rejected_score": int(row["rejected"]["score"]),
                "metric_id": str(row["metric_id"]),
                "language": str(row["language"]),
            }
    return output


def _score_token_ids(
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> dict[int, int]:
    candidates: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        source = metadata[str(row["pair_id"])]
        if source["pair_task"] != "score":
            continue
        chosen_positions = list(row["chosen_field_token_positions"])
        rejected_positions = list(row["rejected_field_token_positions"])
        if len(chosen_positions) != 1 or len(rejected_positions) != 1:
            raise ValueError("score diagnostic requires one score token per side")
        candidates[int(source["gold_label"])].add(
            int(row["chosen_input_ids"][chosen_positions[0]])
        )
        candidates[int(source["rejected_score"])].add(
            int(row["rejected_input_ids"][rejected_positions[0]])
        )
    if set(candidates) != set(range(1, 6)):
        raise ValueError("frozen score pairs do not identify all five score tokens")
    if any(len(values) != 1 for values in candidates.values()):
        raise ValueError("one score has multiple token IDs in frozen manifests")
    return {score: next(iter(values)) for score, values in candidates.items()}


def _batch_measurements(
    model: Any,
    batch: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    score_token_ids: dict[int, int],
) -> tuple[list[float], list[float], list[list[float] | None]]:
    import torch

    batch_size = len(rows)
    input_ids = torch.cat(
        [batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0
    ).to(model.device)
    attention_mask = torch.cat(
        [batch["chosen_attention_mask"], batch["rejected_attention_mask"]],
        dim=0,
    ).to(model.device)
    field_mask = torch.cat(
        [batch["chosen_field_mask"], batch["rejected_field_mask"]], dim=0
    ).to(model.device)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    logps = memory_efficient_field_mean_logps(logits, input_ids, field_mask)
    chosen = logps[:batch_size].detach().float().cpu().tolist()
    rejected = logps[batch_size:].detach().float().cpu().tolist()
    ordered_score_tokens = torch.tensor(
        [score_token_ids[score] for score in range(1, 6)],
        dtype=torch.long,
        device=logits.device,
    )
    score_vectors: list[list[float] | None] = []
    for index, row in enumerate(rows):
        if str(row["pair_task"]) != "score":
            score_vectors.append(None)
            continue
        position = int(row["chosen_field_token_positions"][0])
        vector = torch.log_softmax(
            logits[index, position - 1].float(), dim=-1
        ).index_select(0, ordered_score_tokens)
        score_vectors.append(vector.detach().cpu().tolist())
    return chosen, rejected, score_vectors


def memory_efficient_field_mean_logps(
    logits: Any,
    input_ids: Any,
    field_mask: Any,
) -> Any:
    """Match ``field_mean_logps`` without materializing full-vocabulary logps.

    Formal training uses the shared audited implementation.  This read-only
    diagnostic selects active causal positions first, then applies
    ``target_logit - logsumexp(logits)``.  It is mathematically identical to
    log-softmax followed by target gather, while avoiding a second
    [batch, sequence, vocabulary] tensor.
    """
    import torch

    if (
        logits.ndim != 3
        or input_ids.ndim != 2
        or field_mask.ndim != 2
        or logits.shape[:2] != input_ids.shape
        or field_mask.shape != input_ids.shape
    ):
        raise ValueError("diagnostic logits, IDs, and field mask shapes differ")
    field_mask = field_mask.to(dtype=torch.bool)
    if torch.any(field_mask[:, 0]):
        raise ValueError("position zero cannot be scored by a causal predictor")
    values = []
    for row_index in range(int(input_ids.shape[0])):
        positions = torch.nonzero(
            field_mask[row_index], as_tuple=False
        ).flatten()
        if positions.numel() == 0:
            raise ValueError("diagnostic field mask cannot be empty")
        active_logits = logits[
            row_index,
            positions - 1,
            :,
        ].float()
        targets = input_ids[row_index].index_select(0, positions)
        target_logits = active_logits.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        token_logps = target_logits - torch.logsumexp(
            active_logits, dim=-1
        )
        values.append(token_logps.mean())
    return torch.stack(values)


def _evaluate_adapter(
    model: Any,
    adapter_name: str,
    rows: list[dict[str, Any]],
    *,
    batch_pairs: int,
    collator: SORCDPOPairCollator,
    score_token_ids: dict[int, int],
    on_chunk_complete: Callable[[], None] | None = None,
) -> dict[str, dict[str, Any]]:
    import torch

    model.set_adapter(adapter_name)
    model.eval()
    output = {}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_pairs):
            chunk = rows[start : start + batch_pairs]
            batch = collator(chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chosen, rejected, score_vectors = _batch_measurements(
                    model,
                    batch,
                    chunk,
                    score_token_ids=score_token_ids,
                )
            for row, chosen_value, rejected_value, vector in zip(
                chunk, chosen, rejected, score_vectors, strict=True
            ):
                pair_id = str(row["pair_id"])
                if pair_id in output:
                    raise ValueError("adapter evaluation saw a duplicate pair")
                output[pair_id] = {
                    "chosen_logp": float(chosen_value),
                    "rejected_logp": float(rejected_value),
                    "score_logps_1_to_5": vector,
                }
            if on_chunk_complete is not None:
                on_chunk_complete()
    if len(output) != len(rows):
        raise ValueError("adapter evaluation pair inventory differs")
    return output


def _load_adapter_tensors(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            key: handle.get_tensor(key).float()
            for key in sorted(handle.keys())
        }


def _tensor_update_report(
    reference: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if set(reference) != set(policy):
        raise ValueError("cold-start and trained adapter tensor sets differ")
    accumulators: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "reference_squared_l2": 0.0,
            "delta_squared_l2": 0.0,
            "maximum_absolute_delta": 0.0,
            "parameter_count": 0.0,
            "changed_parameter_count": 0.0,
        }
    )
    deltas = {}
    for key in sorted(reference):
        if reference[key].shape != policy[key].shape:
            raise ValueError(f"adapter tensor shape differs: {key}")
        delta = policy[key] - reference[key]
        deltas[key] = delta
        layer_match = re.search(r"\.layers\.(\d+)\.", key)
        layer = layer_match.group(1) if layer_match else "non_transformer"
        component = "lora_A" if ".lora_A." in key else "lora_B"
        for group in ("global", f"component:{component}", f"layer:{layer}:{component}"):
            value = accumulators[group]
            value["reference_squared_l2"] += float(
                torch.sum(reference[key].double().square())
            )
            value["delta_squared_l2"] += float(
                torch.sum(delta.double().square())
            )
            value["maximum_absolute_delta"] = max(
                value["maximum_absolute_delta"],
                float(delta.abs().max()),
            )
            value["parameter_count"] += float(delta.numel())
            value["changed_parameter_count"] += float(
                torch.count_nonzero(delta)
            )

    def finish(value: dict[str, float]) -> dict[str, Any]:
        reference_l2 = math.sqrt(value["reference_squared_l2"])
        delta_l2 = math.sqrt(value["delta_squared_l2"])
        count = int(value["parameter_count"])
        return {
            "reference_l2": reference_l2,
            "delta_l2": delta_l2,
            "relative_delta_l2": delta_l2 / reference_l2,
            "maximum_absolute_delta": value["maximum_absolute_delta"],
            "parameter_count": count,
            "changed_parameter_fraction": (
                value["changed_parameter_count"] / count
            ),
        }

    return {
        "global": finish(accumulators["global"]),
        "by_component": {
            name.split(":", 1)[1]: finish(value)
            for name, value in sorted(accumulators.items())
            if name.startswith("component:")
        },
        "by_layer_and_component": {
            name.split(":", 1)[1]: finish(value)
            for name, value in sorted(accumulators.items())
            if name.startswith("layer:")
        },
    }, deltas


def _delta_cosine(first: dict[str, Any], second: dict[str, Any]) -> float:
    import torch

    if set(first) != set(second):
        raise ValueError("adapter update tensor sets differ")
    dot = 0.0
    first_squared = 0.0
    second_squared = 0.0
    for key in sorted(first):
        dot += float(torch.sum(first[key].double() * second[key].double()))
        first_squared += float(torch.sum(first[key].double().square()))
        second_squared += float(torch.sum(second[key].double().square()))
    denominator = math.sqrt(first_squared * second_squared)
    if denominator == 0.0:
        raise ValueError("cannot compare a zero adapter update")
    return dot / denominator


def _pair_result(
    *,
    arm: str,
    row: dict[str, Any],
    source: dict[str, Any],
    reference: dict[str, Any],
    policy: dict[str, Any],
    beta: float,
) -> dict[str, Any]:
    reference_margin = float(reference["chosen_logp"]) - float(
        reference["rejected_logp"]
    )
    policy_margin = float(policy["chosen_logp"]) - float(
        policy["rejected_logp"]
    )
    contrast = policy_margin - reference_margin
    beta_contrast = beta * contrast
    offset = float(row["odpo_offset"])
    value = {
        "arm": arm,
        "pair_id": str(row["pair_id"]),
        "record_id": str(row["record_id"]),
        **source,
        "odpo_offset": offset,
        "reference_chosen_logp": float(reference["chosen_logp"]),
        "reference_rejected_logp": float(reference["rejected_logp"]),
        "policy_chosen_logp": float(policy["chosen_logp"]),
        "policy_rejected_logp": float(policy["rejected_logp"]),
        "reference_raw_margin": reference_margin,
        "policy_raw_margin": policy_margin,
        "chosen_logp_change": float(policy["chosen_logp"])
        - float(reference["chosen_logp"]),
        "rejected_logp_change": float(policy["rejected_logp"])
        - float(reference["rejected_logp"]),
        "preference_contrast": contrast,
        "beta_scaled_contrast": beta_contrast,
        "risk_adjusted_margin": beta_contrast - offset,
        "reference_prefers_chosen": reference_margin > 0.0,
        "policy_prefers_chosen": policy_margin > 0.0,
        "contrast_positive": contrast > 0.0,
        "offset_satisfied": beta_contrast > offset,
        "chosen_logp_increased": (
            float(policy["chosen_logp"]) > float(reference["chosen_logp"])
        ),
        "rejected_logp_decreased": (
            float(policy["rejected_logp"]) < float(reference["rejected_logp"])
        ),
    }
    if source["pair_task"] == "score":
        reference_scores = list(reference["score_logps_1_to_5"])
        policy_scores = list(policy["score_logps_1_to_5"])
        value.update(
            {
                "reference_score_logps_1_to_5": reference_scores,
                "policy_score_logps_1_to_5": policy_scores,
                "reference_predicted_score": 1
                + max(range(5), key=reference_scores.__getitem__),
                "policy_predicted_score": 1
                + max(range(5), key=policy_scores.__getitem__),
            }
        )
    return value


def run_seed(
    *,
    seed: int,
    cuda_device_uuid: str,
    batch_pairs: int,
    output_dir: Path,
) -> dict[str, Any]:
    if seed not in (42, 43, 44):
        raise ValueError("diagnostic seed must be 42, 43, or 44")
    if batch_pairs < 1:
        raise ValueError("batch_pairs must be positive")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()

    manifests = {}
    config = None
    for arm in ARMS:
        rows, current_config, _frozen = load_formal_rows(arm=arm, seed=seed)
        manifests[arm] = rows
        if config is None:
            config = current_config
        elif config != current_config:
            raise ValueError("arm training configurations differ")
    assert config is not None
    source = _source_metadata()
    for arm, rows in manifests.items():
        for row in rows:
            pair_id = str(row["pair_id"])
            if pair_id not in source:
                raise ValueError(f"{arm}: manifest pair absent from frozen source")
            if str(row["pair_task"]) != source[pair_id]["pair_task"]:
                raise ValueError(f"{arm}: pair task differs from frozen source")

    p3_rows = manifests["P3_JOINT_SORC"]
    score_token_ids = _score_token_ids(p3_rows, source)
    base_path, _base_config = verify_base_model_snapshot(
        DEFAULT_BASE_TRAINING_CONFIGURATION
    )
    cold_adapter_dir, cold_hashes = _checkpoint_for_seed(seed)
    trained_paths = {}
    trained_hashes = {}
    for arm in ARMS:
        path, hashes = _validate_adapter(
            arm=arm,
            seed=seed,
            training_root=(
                REPO_ROOT
                / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
                "preference_formal_runs"
            ),
            audit_report_path=(
                REPO_ROOT
                / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
                "preference_formal_runs/formal_training_audit_report.json"
            ),
        )
        trained_paths[arm] = path
        trained_hashes[arm] = hashes

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    identity = _device_identity(torch, cuda_device_uuid)
    base = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base,
        str(cold_adapter_dir),
        adapter_name="reference",
        is_trainable=False,
    )
    adapter_names = {}
    for arm in ARMS:
        name = arm.lower()
        model.load_adapter(
            str(trained_paths[arm]),
            adapter_name=name,
            is_trainable=False,
        )
        adapter_names[arm] = name
    collator = SORCDPOPairCollator(pad_token_id=151643, cutoff_len=2048)

    total_forward_chunks = math.ceil(len(p3_rows) / batch_pairs) + sum(
        math.ceil(len(rows) / batch_pairs) for rows in manifests.values()
    )
    completed_forward_chunks = 0
    current_phase = "reference"

    def record_progress() -> None:
        nonlocal completed_forward_chunks
        completed_forward_chunks += 1
        elapsed = time.time() - started
        fraction = completed_forward_chunks / total_forward_chunks
        _atomic_json(
            output_dir / "progress.json",
            {
                "schema_version": (
                    "exp54-sorc-dpo-train-signal-diagnostic-progress-v1"
                ),
                "status": "RUNNING",
                "seed": seed,
                "phase": current_phase,
                "completed_forward_chunks": completed_forward_chunks,
                "total_forward_chunks": total_forward_chunks,
                "percent_complete": 100.0 * fraction,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": (
                    elapsed * (1.0 - fraction) / fraction
                ),
                "dev_accessed": False,
                "test_accessed": False,
            },
        )

    _atomic_json(
        output_dir / "progress.json",
        {
            "schema_version": (
                "exp54-sorc-dpo-train-signal-diagnostic-progress-v1"
            ),
            "status": "LOADING_MODEL",
            "seed": seed,
            "phase": "model_and_adapter_loading",
            "completed_forward_chunks": 0,
            "total_forward_chunks": total_forward_chunks,
            "percent_complete": 0.0,
            "elapsed_seconds": time.time() - started,
            "estimated_remaining_seconds": None,
            "dev_accessed": False,
            "test_accessed": False,
        },
    )
    reference_values = _evaluate_adapter(
        model,
        "reference",
        p3_rows,
        batch_pairs=batch_pairs,
        collator=collator,
        score_token_ids=score_token_ids,
        on_chunk_complete=record_progress,
    )
    beta = float(config["loss"]["beta"])
    arm_reports = {}
    all_private_rows = []
    for arm in ARMS:
        current_phase = arm
        rows = manifests[arm]
        policy_values = _evaluate_adapter(
            model,
            adapter_names[arm],
            rows,
            batch_pairs=batch_pairs,
            collator=collator,
            score_token_ids=score_token_ids,
            on_chunk_complete=record_progress,
        )
        private_rows = [
            _pair_result(
                arm=arm,
                row=row,
                source=source[str(row["pair_id"])],
                reference=reference_values[str(row["pair_id"])],
                policy=policy_values[str(row["pair_id"])],
                beta=beta,
            )
            for row in rows
        ]
        all_private_rows.extend(private_rows)
        arm_reports[arm] = {
            "pair_count": len(private_rows),
            "aggregates": grouped_summaries(private_rows),
            "manifest_sha256": sha256_file(
                REPO_ROOT
                / "thesis_exp/outputs/exp54_rar_sft/rar_v2/"
                f"preference_training_candidate/private/{arm.lower()}.jsonl"
            ),
            "trained_adapter_hashes": trained_hashes[arm],
        }

    private_path = output_dir / "private/pair_diagnostics.jsonl"
    _atomic_jsonl(private_path, all_private_rows)

    reference_tensors = _load_adapter_tensors(
        cold_adapter_dir / "adapter_model.safetensors"
    )
    delta_by_arm = {}
    for arm in ARMS:
        policy_tensors = _load_adapter_tensors(
            trained_paths[arm] / "adapter_model.safetensors"
        )
        update_report, deltas = _tensor_update_report(
            reference_tensors, policy_tensors
        )
        arm_reports[arm]["adapter_update"] = update_report
        delta_by_arm[arm] = deltas
    update_cosines = {}
    for first_index, first in enumerate(ARMS):
        for second in ARMS[first_index + 1 :]:
            update_cosines[f"{first}__{second}"] = _delta_cosine(
                delta_by_arm[first], delta_by_arm[second]
            )

    report = {
        "schema_version": "exp54-sorc-dpo-train-signal-diagnostic-v1",
        "status": "SORC_DPO_TRAIN_SIGNAL_DIAGNOSTIC_COMPLETE",
        "seed": seed,
        "beta": beta,
        "batch_pairs": batch_pairs,
        "elapsed_seconds": time.time() - started,
        "device_identity": identity,
        "arms": arm_reports,
        "adapter_update_cosines": update_cosines,
        "source_hashes": {
            "diagnostic_source": sha256_file(Path(__file__)),
            "training_config": sha256_file(DEFAULT_TRAINING_CONFIG),
            "cold_start_adapter": cold_hashes,
            "private_pair_diagnostics": sha256_file(private_path),
            "frozen_pair_sources": {
                name: sha256_file(path) for name, path in PAIR_FILES.items()
            },
        },
        "row_level_values_public": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    report_path = output_dir / "aggregate_report.json"
    _atomic_json(report_path, report)
    _atomic_json(
        output_dir / "progress.json",
        {
            "schema_version": (
                "exp54-sorc-dpo-train-signal-diagnostic-progress-v1"
            ),
            "status": "COMPLETE",
            "seed": seed,
            "phase": "complete",
            "completed_forward_chunks": total_forward_chunks,
            "total_forward_chunks": total_forward_chunks,
            "percent_complete": 100.0,
            "elapsed_seconds": report["elapsed_seconds"],
            "estimated_remaining_seconds": 0.0,
            "dev_accessed": False,
            "test_accessed": False,
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "seed": seed,
                "elapsed_seconds": report["elapsed_seconds"],
                "report_sha256": sha256_file(report_path),
                "dev_accessed": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--cuda-device-uuid", required=True)
    parser.add_argument("--batch-pairs", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_seed(
        seed=args.seed,
        cuda_device_uuid=args.cuda_device_uuid,
        batch_pairs=args.batch_pairs,
        output_dir=args.output_root / f"seed_{args.seed}",
    )


if __name__ == "__main__":
    main()
