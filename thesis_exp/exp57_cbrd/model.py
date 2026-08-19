"""Qwen3 dual-head model with an explicit CBRD backbone-routing branch."""

from __future__ import annotations

import copy
import hashlib
from typing import Any


def module_hash(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_cbrd_dual_head_classifier(base_model: Any) -> Any:
    """Wrap the Exp51-compatible classifier with auxiliary-gradient routing.

    The auxiliary head always receives the original full soft-CE derivative.
    A hook on a cloned hidden-state branch controls only the gradient returned
    to the shared backbone.  This avoids the BF16 rounding drift produced by
    adding separately backpropagated consensus and residual branches.
    """

    import torch
    from torch import nn
    from torch.nn import functional as functional

    if not hasattr(base_model, "model") or not hasattr(base_model, "score"):
        raise TypeError("CBRD requires the Qwen3 sequence-classification model/model.score path")

    class CbrdDualHeadClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = base_model.model
            self.hard_head = base_model.score
            self.soft_head = copy.deepcopy(base_model.score)
            self.config = base_model.config
            self.num_labels = int(base_model.num_labels)

        def gradient_checkpointing_enable(self) -> None:
            if hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()

        def _last_token(self, token_logits: Any, input_ids: Any, inputs_embeds: Any) -> Any:
            indices = self._last_token_indices(token_logits, input_ids, inputs_embeds)
            batch_size = token_logits.shape[0]
            return token_logits[torch.arange(batch_size, device=token_logits.device), indices]

        def _last_token_indices(self, token_values: Any, input_ids: Any, inputs_embeds: Any) -> Any:
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 without pad_token_id")
            if self.config.pad_token_id is None or input_ids is None:
                last_non_pad_token: Any = -1
            else:
                non_pad_mask = (input_ids != self.config.pad_token_id).to(token_values.device, torch.int32)
                token_indices = torch.arange(input_ids.shape[-1], device=token_values.device, dtype=torch.int32)
                last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
            return last_non_pad_token

        def forward(
            self,
            input_ids: Any = None,
            attention_mask: Any = None,
            labels: Any = None,
            soft_targets: Any = None,
            residual_targets: Any = None,
            aux_route: str = "ordinary_hmsa",
            route_loss_scale: float = 1.0,
            inputs_embeds: Any = None,
            retain_backbone_hidden: bool = False,
            **kwargs: Any,
        ) -> dict[str, Any]:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            hidden = outputs.last_hidden_state
            if retain_backbone_hidden:
                hidden.retain_grad()
            hard_logits = self._last_token(self.hard_head(hidden), input_ids, inputs_embeds)
            valid_routes = {
                "ordinary_hmsa",
                "routed_hmsa",
                "dual_hard",
                "consensus_only",
                "residual_only",
                "sign_flipped",
                "shuffled_residual",
                "detached_soft",
            }
            if aux_route not in valid_routes:
                raise ValueError(f"Unknown auxiliary route: {aux_route}")

            # Ordinary HMSA and DualHard use the unmodified graph.  All other
            # controls keep the same forward values and auxiliary-head loss but
            # route only this cloned branch's gradient back to the backbone.
            routed = aux_route not in {"ordinary_hmsa", "dual_hard"}
            aux_hidden = hidden.clone() if routed else hidden
            if routed:
                if labels is None or soft_targets is None:
                    raise ValueError(f"{aux_route} requires labels and soft_targets")
                if aux_route == "shuffled_residual" and residual_targets is None:
                    raise ValueError("shuffled_residual requires residual_targets")
                last_indices = self._last_token_indices(aux_hidden, input_ids, inputs_embeds)
                batch_size = int(aux_hidden.shape[0])
                head_weight = self.soft_head.weight.detach().float()
                labels_for_route = labels.detach().long()
                original_targets = soft_targets.detach().float()
                alternative_targets = (
                    original_targets
                    if residual_targets is None
                    else residual_targets.detach().float()
                )
                if not float(route_loss_scale) > 0.0:
                    raise ValueError("route_loss_scale must be positive")

                def residual_hidden_gradient(target_values: Any, gradient: Any) -> Any:
                    hard = functional.one_hot(
                        labels_for_route,
                        num_classes=self.num_labels,
                    ).float()
                    # The hook receives the derivative of the *scaled* loss.
                    # The analytic residual must use the same scale, notably
                    # 1 / gradient_accumulation_steps during formal training.
                    selected = (
                        ((hard - target_values) @ head_weight)
                        / float(batch_size)
                        * float(route_loss_scale)
                    )
                    routed_gradient = torch.zeros_like(gradient)
                    selected = selected.to(dtype=gradient.dtype)
                    row_indices = torch.arange(batch_size, device=gradient.device)
                    routed_gradient[row_indices, last_indices] = selected
                    return routed_gradient

                def route_hook(gradient: Any) -> Any:
                    if aux_route == "routed_hmsa":
                        return gradient
                    original_residual = residual_hidden_gradient(original_targets, gradient)
                    if aux_route == "consensus_only":
                        return gradient - original_residual
                    if aux_route == "residual_only":
                        return original_residual
                    if aux_route == "sign_flipped":
                        return gradient - 2.0 * original_residual
                    if aux_route == "shuffled_residual":
                        shuffled_residual = residual_hidden_gradient(alternative_targets, gradient)
                        return gradient - original_residual + shuffled_residual
                    if aux_route == "detached_soft":
                        return torch.zeros_like(gradient)
                    raise AssertionError(f"Unhandled route: {aux_route}")

                aux_hidden.register_hook(route_hook)

            aux_logits = self._last_token(self.soft_head(aux_hidden), input_ids, inputs_embeds)
            loss = functional.cross_entropy(hard_logits.float(), labels) if labels is not None else None
            result = {
                "loss": loss,
                "logits": hard_logits,
                "hard_logits": hard_logits,
                "aux_logits": aux_logits,
                # Aliases keep diagnostic consumers simple; there is now only
                # one numerical auxiliary forward and one soft-CE backward.
                "aux_head_logits": aux_logits,
                "aux_route_logits": aux_logits,
                "aux_route": aux_route,
            }
            if retain_backbone_hidden:
                result["backbone_hidden"] = hidden
            return result

    return CbrdDualHeadClassifier()


def head_contract(model: Any) -> dict[str, Any]:
    hard_parameters = list(model.hard_head.parameters())
    soft_parameters = list(model.soft_head.parameters())
    if len(hard_parameters) != len(soft_parameters):
        raise AssertionError("hard/soft head parameter structures differ")
    storage_independent = all(hard.data_ptr() != soft.data_ptr() for hard, soft in zip(hard_parameters, soft_parameters))
    return {
        "hard_head_hash": module_hash(model.hard_head),
        "soft_head_hash": module_hash(model.soft_head),
        "storage_independent": storage_independent,
        "hard_parameter_count": sum(value.numel() for value in hard_parameters),
        "soft_parameter_count": sum(value.numel() for value in soft_parameters),
        "hard_bias": model.hard_head.bias is not None,
        "soft_bias": model.soft_head.bias is not None,
    }
