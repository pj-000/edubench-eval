"""Human-anchor and distillation losses for Exp46A."""

from __future__ import annotations

from typing import Any


def distribution_ce(logits: Any, targets: Any) -> Any:
    import torch
    return -(targets.float() * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()


def ordinal_cdf_mse_from_probabilities(probabilities: Any, targets: Any) -> Any:
    import torch
    return torch.mean((torch.cumsum(probabilities.float(), dim=-1)[:, :4] - torch.cumsum(targets.float(), dim=-1)[:, :4]) ** 2)


def human_anchor_loss(logits: Any, targets: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    distribution = distribution_ce(logits, targets)
    ordinal = ordinal_cdf_mse_from_probabilities(torch.softmax(logits.float(), dim=-1), targets)
    total = distribution + 0.5 * ordinal
    return total, {"human_distribution": distribution, "human_ordinal": ordinal}


def standard_kd_loss(student_logits: Any, teacher_logits: Any, temperature: float = 2.0) -> Any:
    import torch
    teacher = torch.softmax(teacher_logits.float() / temperature, dim=-1)
    student_log = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    return (temperature ** 2) * torch.nn.functional.kl_div(student_log, teacher, reduction="batchmean")


def ordinal_kd_loss(student_logits: Any, teacher_logits: Any, temperature: float = 2.0) -> Any:
    import torch
    teacher = torch.softmax(teacher_logits.float() / temperature, dim=-1)
    student = torch.softmax(student_logits.float() / temperature, dim=-1)
    teacher_survival = 1.0 - torch.cumsum(teacher, dim=-1)[:, :4]
    student_survival = 1.0 - torch.cumsum(student, dim=-1)[:, :4]
    return torch.mean((teacher_survival - student_survival) ** 2)
