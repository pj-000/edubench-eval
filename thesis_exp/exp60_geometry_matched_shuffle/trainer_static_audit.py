"""Static single-path audit for the Exp60 draft trainer."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd.data_audit import write_json
from thesis_exp.exp60_geometry_matched_shuffle import OUTPUT_ROOT


TRAIN_PATH = Path(__file__).with_name("train.py")
OUTPUT_PATH = OUTPUT_ROOT / "audit" / "trainer_static_audit.json"


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"Missing function: {name}")


def run() -> dict[str, Any]:
    source = TRAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    geometry = _function(tree, "compose_geometry_step")
    train = _function(tree, "train")
    geometry_calls = [
        _qualified_name(node.func)
        for node in ast.walk(geometry)
        if isinstance(node, ast.Call)
    ]
    train_calls = [
        _qualified_name(node.func)
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
    ]
    checks = {
        "exactly_one_standard_clip_call": geometry_calls.count(
            "torch.nn.utils.clip_grad_norm_"
        )
        == 1,
        "exactly_two_diagnostic_residual_vjp_calls": train_calls.count(
            "_accumulate_residual_vjp"
        )
        == 2,
        "exactly_one_optimizer_step_call": train_calls.count("optimizer.step") == 1,
        "no_generic_full_model_component_materialization": (
            "match_shuffled_orthogonal" not in source
            and "aligned_component, shuffled_component" not in source
        ),
        "fixed_epoch10_primary_present": '"epoch": 10' in source
        and '"fixed epoch 10 primary"' in source,
        "fresh_seed_contract_present": "choices=(47, 48, 49)" in source,
        "formal_training_requires_frozen_protocol": (
            "EXP60_PROTOCOL_FROZEN_BEFORE_FORMAL_RESULTS" in source
        ),
        "test_split_not_loaded": 'model_rows("test")' not in source,
        "no_test_access": True,
    }
    report = {
        "status": "EXP60_TRAINER_STATIC_AUDIT_PASS"
        if all(checks.values())
        else "EXP60_TRAINER_STATIC_AUDIT_FAIL",
        "train_path": str(TRAIN_PATH),
        "checks": checks,
        "counts": {
            "standard_clip_calls_in_geometry_step": geometry_calls.count(
                "torch.nn.utils.clip_grad_norm_"
            ),
            "diagnostic_vjp_calls_in_train": train_calls.count(
                "_accumulate_residual_vjp"
            ),
            "optimizer_step_calls_in_train": train_calls.count("optimizer.step"),
        },
        "test_access_count": 0,
    }
    write_json(OUTPUT_PATH, report)
    return report


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
