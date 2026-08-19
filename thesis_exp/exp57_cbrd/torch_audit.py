"""PyTorch-only CBRD scalar and gradient equivalence audit.

This file is runnable on the training environment.  It intentionally uses a
small deterministic two-layer toy backbone first; the later real-model audit
will re-use the same assertions with the frozen Qwen3 head architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_exp.exp57_cbrd import OUTPUT_ROOT


TARGET_THIRDS = (
    (3, 0, 0, 0, 0), (2, 1, 0, 0, 0),
    (1, 2, 0, 0, 0), (0, 3, 0, 0, 0), (0, 2, 1, 0, 0),
    (0, 1, 2, 0, 0), (0, 0, 3, 0, 0), (0, 0, 2, 1, 0),
    (0, 0, 1, 2, 0), (0, 0, 0, 3, 0), (0, 0, 0, 2, 1),
    (0, 0, 0, 1, 2), (0, 0, 0, 0, 3),
)


def _torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required; run this audit in the training environment") from error
    return torch, nn, functional


def _losses(logits: Any, labels: Any, targets: Any) -> tuple[Any, Any, Any]:
    """Dtype-preserving implementation used to audit the algebra itself.

    The production implementation deliberately casts losses to FP32, matching
    Exp51.  This audit keeps the caller's dtype so the FP64 numerical claim is
    genuinely tested rather than silently reduced to FP32.
    """

    torch, _, functional = _torch()
    log_probs = functional.log_softmax(logits, dim=-1)
    soft = -(targets.to(dtype=logits.dtype) * log_probs).sum(dim=-1).mean()
    consensus = functional.cross_entropy(logits, labels)
    hard = functional.one_hot(labels, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
    residual = ((hard - targets.to(dtype=logits.dtype)).detach() * logits).sum(dim=-1).mean()
    return soft, consensus, residual


def _max_abs(left: Any, right: Any) -> float:
    return float((left.detach() - right.detach()).abs().max().cpu())


def scalar_identity(dtype_name: str) -> dict[str, Any]:
    torch, _, _ = _torch()
    dtype = {"float64": torch.float64, "float32": torch.float32}[dtype_name]
    torch.manual_seed(5701)
    labels = torch.tensor([max(range(5), key=target.__getitem__) for target in TARGET_THIRDS], dtype=torch.long)
    targets = torch.tensor(TARGET_THIRDS, dtype=dtype) / 3.0
    logits = torch.randn(len(TARGET_THIRDS), 5, dtype=dtype, requires_grad=True)
    soft, consensus, residual = _losses(logits, labels, targets)
    grad_soft = torch.autograd.grad(soft, logits, retain_graph=True)[0]
    grad_reconstructed = torch.autograd.grad(consensus + residual, logits)[0]
    max_scalar_error = float((soft - consensus - residual).abs().detach().cpu())
    max_gradient_error = _max_abs(grad_soft, grad_reconstructed)
    tolerance = 1e-10 if dtype == torch.float64 else 1e-6
    return {
        "dtype": dtype_name,
        "target_vectors": len(TARGET_THIRDS),
        "max_scalar_error": max_scalar_error,
        "max_logit_gradient_error": max_gradient_error,
        "tolerance": tolerance,
        "pass": max(max_scalar_error, max_gradient_error) <= tolerance,
    }


def toy_routing_identity() -> dict[str, Any]:
    torch, nn, functional = _torch()

    class ToyDual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bottom = nn.Linear(7, 11, bias=False)
            self.top = nn.Linear(11, 13, bias=False)
            self.hard_head = nn.Linear(13, 5, bias=False)
            self.soft_head = nn.Linear(13, 5, bias=False)
            self.soft_head.load_state_dict(self.hard_head.state_dict())

        def encoded(self, x: Any) -> Any:
            return self.top(torch.tanh(self.bottom(x)))

    torch.manual_seed(5702)
    model = ToyDual().double().eval()
    x = torch.randn(13, 7, dtype=torch.float64)
    labels = torch.tensor([max(range(5), key=target.__getitem__) for target in TARGET_THIRDS], dtype=torch.long)
    targets = torch.tensor(TARGET_THIRDS, dtype=torch.float64) / 3.0

    def ordinary() -> tuple[dict[str, Any], Any]:
        model.zero_grad(set_to_none=True)
        hidden = model.encoded(x)
        hidden.retain_grad()
        hard_logits = model.hard_head(hidden)
        soft_logits = model.soft_head(hidden)
        loss = functional.cross_entropy(hard_logits, labels) + _losses(soft_logits, labels, targets)[0]
        loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
        return gradients, hidden.grad.detach().clone()

    def routed() -> tuple[dict[str, Any], Any, dict[str, float]]:
        model.zero_grad(set_to_none=True)
        hidden = model.encoded(x)
        hidden.retain_grad()
        hard_logits = model.hard_head(hidden)
        head_logits = model.soft_head(hidden.detach())
        route_logits = functional.linear(
            hidden,
            model.soft_head.weight.detach(),
            None if model.soft_head.bias is None else model.soft_head.bias.detach(),
        )
        aux_head_soft, _, _ = _losses(head_logits, labels, targets)
        _, consensus, residual = _losses(route_logits, labels, targets)
        # This scalar is intentionally numerically hard + 2*soft.  Detach
        # routing ensures each parameter group receives the ordinary
        # hard+soft derivative exactly once.
        optimization_loss = functional.cross_entropy(hard_logits, labels) + aux_head_soft + consensus + residual
        optimization_loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
        scalar = {
            "aux_head_soft": float(aux_head_soft.detach()),
            "route_consensus_plus_residual": float((consensus + residual).detach()),
            "optimization_loss": float(optimization_loss.detach()),
        }
        return gradients, hidden.grad.detach().clone(), scalar

    ordinary_grads, ordinary_hidden = ordinary()
    routed_grads, routed_hidden, scalar = routed()
    per_tensor = {name: _max_abs(ordinary_grads[name], routed_grads[name]) for name in ordinary_grads}
    report = {
        "max_parameter_gradient_error": max(per_tensor.values()),
        "per_parameter_gradient_error": per_tensor,
        "hidden_gradient_error": _max_abs(ordinary_hidden, routed_hidden),
        "checks": {
            "bottom_backbone_checked": "bottom.weight" in per_tensor,
            "top_backbone_checked": "top.weight" in per_tensor,
            "auxiliary_head_checked": "soft_head.weight" in per_tensor,
            "hard_head_checked": "hard_head.weight" in per_tensor,
        },
        "scalars": scalar,
    }
    report["pass"] = max(report["max_parameter_gradient_error"], report["hidden_gradient_error"]) <= 1e-10 and all(report["checks"].values())
    return report


def toy_identity_hook_parity() -> dict[str, Any]:
    """Verify that an identity hook preserves the original one-CE gradients."""

    torch, nn, functional = _torch()

    class ToyDual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bottom = nn.Linear(7, 11, bias=False)
            self.top = nn.Linear(11, 13, bias=False)
            self.hard_head = nn.Linear(13, 5, bias=False)
            self.soft_head = nn.Linear(13, 5, bias=False)
            self.soft_head.load_state_dict(self.hard_head.state_dict())

        def encoded(self, x: Any) -> Any:
            return self.top(torch.tanh(self.bottom(x)))

    torch.manual_seed(5704)
    model = ToyDual().double().eval()
    x = torch.randn(13, 7, dtype=torch.float64)
    labels = torch.tensor(
        [max(range(5), key=target.__getitem__) for target in TARGET_THIRDS],
        dtype=torch.long,
    )
    targets = torch.tensor(TARGET_THIRDS, dtype=torch.float64) / 3.0

    def capture(*, identity_hook: bool) -> tuple[dict[str, Any], Any]:
        model.zero_grad(set_to_none=True)
        hidden = model.encoded(x)
        hidden.retain_grad()
        auxiliary_hidden = hidden.clone() if identity_hook else hidden
        if identity_hook:
            auxiliary_hidden.register_hook(lambda gradient: gradient)
        hard_logits = model.hard_head(hidden)
        soft_logits = model.soft_head(auxiliary_hidden)
        loss = functional.cross_entropy(hard_logits, labels) + _losses(
            soft_logits,
            labels,
            targets,
        )[0]
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        return gradients, hidden.grad.detach().clone()

    ordinary_gradients, ordinary_hidden = capture(identity_hook=False)
    hooked_gradients, hooked_hidden = capture(identity_hook=True)
    per_tensor = {
        name: _max_abs(ordinary_gradients[name], hooked_gradients[name])
        for name in ordinary_gradients
    }
    hidden_error = _max_abs(ordinary_hidden, hooked_hidden)
    return {
        "pass": max(max(per_tensor.values()), hidden_error) <= 1e-10,
        "max_parameter_gradient_error": max(per_tensor.values()),
        "hidden_gradient_error": hidden_error,
        "per_parameter_gradient_error": per_tensor,
    }


def amp_scalar_identity() -> dict[str, Any]:
    """Run the same scalar/gradient identity inside CUDA BF16 autocast."""

    torch, _, _ = _torch()
    if not torch.cuda.is_available():
        return {"status": "SKIPPED_NO_CUDA", "pass": False}
    torch.manual_seed(5703)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    labels = torch.tensor([max(range(5), key=target.__getitem__) for target in TARGET_THIRDS], dtype=torch.long, device=device)
    targets = torch.tensor(TARGET_THIRDS, dtype=torch.float32, device=device) / 3.0
    logits = torch.randn(len(TARGET_THIRDS), 5, device=device, dtype=torch.float32, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        soft, consensus, residual = _losses(logits, labels, targets)
    grad_soft = torch.autograd.grad(soft, logits, retain_graph=True)[0]
    grad_reconstructed = torch.autograd.grad(consensus + residual, logits)[0]
    max_scalar_error = float((soft - consensus - residual).abs().detach().cpu())
    max_gradient_error = _max_abs(grad_soft, grad_reconstructed)
    tolerance = 1e-5
    return {
        "status": "PASS" if max(max_scalar_error, max_gradient_error) <= tolerance else "FAIL",
        "device": str(device),
        "autocast_dtype": str(amp_dtype).replace("torch.", ""),
        "max_scalar_error": max_scalar_error,
        "max_logit_gradient_error": max_gradient_error,
        "tolerance": tolerance,
        "pass": max(max_scalar_error, max_gradient_error) <= tolerance,
    }


def run() -> dict[str, Any]:
    scalar64 = scalar_identity("float64")
    scalar32 = scalar_identity("float32")
    two_branch_routing = toy_routing_identity()
    identity_hook = toy_identity_hook_parity()
    amp = amp_scalar_identity()
    report = {
        "status": "PASS" if scalar64["pass"] and scalar32["pass"] and two_branch_routing["pass"] and identity_hook["pass"] and amp.get("pass", False) else "PARTIAL_PASS_NEEDS_CUDA" if scalar64["pass"] and scalar32["pass"] and two_branch_routing["pass"] and identity_hook["pass"] else "FAIL",
        "scalar_identity": {"float64": scalar64, "float32": scalar32},
        "toy_two_branch_routing_identity": two_branch_routing,
        "toy_identity_hook_parity": identity_hook,
        "gpu_amp_identity": amp,
        "real_qwen3_routing_status": "PENDING_GPU_RUNTIME_AUDIT",
        "test_access_count": 0,
    }
    output = OUTPUT_ROOT / "audit" / "stage0_torch_identity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
