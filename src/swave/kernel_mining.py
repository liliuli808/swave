"""Exact training-row mining and a bounded-influence derivative objective."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.func import jvp


@torch.no_grad()
def mine_kernel_rows(
    forward: Callable[[Tensor], Tensor],
    vs: Tensor,
    labels: Tensor,
    mask: Tensor,
    pool_indices: Tensor,
    *,
    batch_size: int = 256,
    threshold: float = 0.03,
) -> tuple[Tensor, dict[str, object]]:
    """Find difficult valid rows using all input directions, with no AD graph.

    Returns global (model index, mode, frequency) triples. The caller supplies
    training data only. A new random model pool can be supplied at each refresh.
    """
    candidates = []
    valid_by_mode = torch.zeros(mask.shape[1], dtype=torch.long, device=vs.device)
    passed_by_mode = torch.zeros_like(valid_by_mode)
    for start in range(0, len(pool_indices), batch_size):
        index = pool_indices[start : start + batch_size]
        inputs = vs[index]
        target = labels[index].to(inputs.dtype)
        valid = mask[index]
        squared_error = torch.zeros_like(target[..., 0])
        # Accumulate the exact squared row error without materializing a second
        # (batch, mode, frequency, layer) Jacobian or retaining training graphs.
        for layer in range(inputs.shape[1]):
            tangent = torch.zeros_like(inputs)
            tangent[:, layer] = 1
            _, derivative = jvp(forward, (inputs,), (tangent,))
            squared_error += (derivative - target[..., layer]).square()
        norm = target.square().sum(-1).sqrt().clamp_min(1e-12)
        relative = squared_error.sqrt() / norm
        if not torch.isfinite(relative[valid]).all().item():
            raise FloatingPointError("non-finite kernel error during training-row mining")
        valid_by_mode += valid.sum(dim=(0, 2))
        passed_by_mode += (valid & (relative < 0.05)).sum(dim=(0, 2))
        selected = (valid & (relative >= threshold)).nonzero()
        if len(selected):
            selected[:, 0] = index[selected[:, 0]]
            candidates.append(selected)
    bank = (torch.cat(candidates) if candidates else
            torch.empty((0, 3), device=vs.device, dtype=torch.long))
    valid_counts = valid_by_mode.cpu().tolist()
    passed_counts = passed_by_mode.cpu().tolist()
    total = sum(valid_counts)
    return bank, {
        "pool_models": len(pool_indices),
        "valid_rows": total,
        "selected_rows": len(bank),
        "selection_threshold": threshold,
        "selected_rows_by_mode": torch.bincount(
            bank[:, 1], minlength=mask.shape[1],
        ).cpu().tolist(),
        "pool_rows_within_5pct": sum(passed_counts) / total if total else None,
        "pool_rows_within_5pct_by_mode": [
            passed / count if count else None
            for passed, count in zip(passed_counts, valid_counts, strict=True)
        ],
    }


def exact_row_kernel_loss(
    forward: Callable[[Tensor], Tensor],
    vs: Tensor,
    labels: Tensor,
    modes: Tensor,
    frequencies: Tensor,
    *,
    kernel_floor: float = 1e-2,
    percent_scale: float = 1e-2,
) -> tuple[Tensor, Tensor]:
    """Fit one exact Jacobian row per model with twice pseudo-Huber loss.

    Rows in the batch must be independent (as in FourHeadForwardModel).
    The robust transition is at 5% relative error. Small errors have the same
    squared-percent scale as the ordinary kernel objective; large errors grow
    linearly, so a few extreme rows do not dominate this additional objective.
    """
    inputs = vs.detach().requires_grad_(True)
    prediction = forward(inputs)
    selected = prediction[torch.arange(len(inputs), device=inputs.device),
                          modes, frequencies]
    derivative = torch.autograd.grad(selected.sum(), inputs, create_graph=True)[0]
    target = labels.to(derivative.dtype)
    norm2 = target.square().sum(-1)
    squared_error = (derivative - target).square().sum(-1)
    percent_squared = squared_error / (norm2 + kernel_floor**2) / percent_scale**2
    # 2 * delta^2 * (sqrt(1 + r^2/delta^2) - 1), delta = 5 percent.
    # Rationalize to retain accuracy near a zero residual.
    loss = (2 * percent_squared / ((1 + percent_squared / 25).sqrt() + 1)).mean()
    relative = squared_error.detach().sqrt() / norm2.detach().sqrt().clamp_min(1e-12)
    return loss, relative
