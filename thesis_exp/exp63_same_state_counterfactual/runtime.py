"""Shared deterministic runtime utilities for Exp63."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from thesis_exp.exp63_same_state_counterfactual import PROTOCOL_PATH, REPO_ROOT


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def hash_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "EXP63_PROTOCOL_FROZEN_BEFORE_ANY_REAL_MODEL_UPDATE_RESULT":
        raise RuntimeError("Exp63 protocol is not frozen")
    if protocol.get("allowed_splits") != ["train", "dev"]:
        raise RuntimeError("Exp63 split contract changed")
    if protocol.get("test_access_count") != 0:
        raise RuntimeError("Exp63 test access must remain zero")
    return protocol


def source_hashes(relative_paths: Iterable[str]) -> dict[str, str]:
    return {relative: sha256_file(REPO_ROOT / relative) for relative in relative_paths}


def epoch_order(row_count: int, seed: int, epoch: int) -> list[int]:
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed * 1000 + epoch)
    return torch.randperm(row_count, generator=generator).tolist()


def ordered_rows(rows: list[dict[str, Any]], seed: int, epoch: int) -> list[dict[str, Any]]:
    return [rows[index] for index in epoch_order(len(rows), seed, epoch)]


def capture_rng(torch: Any, device: Any) -> dict[str, Any]:
    import random

    import numpy as np

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(torch: Any, device: Any, state: dict[str, Any]) -> None:
    import random

    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)

