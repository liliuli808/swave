from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import pseudo_huber

from swave.kernel_mining import exact_row_kernel_loss, mine_kernel_rows
from swave.network import FourHeadForwardModel


def test_mining_uses_exact_errors_and_preserves_global_indices_and_valid_mask():
    def forward(vs):
        return vs[:, :1, None].expand(-1, 4, 3)

    vs = torch.ones(4, 3)
    labels = torch.zeros(4, 4, 3, 3)
    labels[..., 0] = 1
    labels[1, 3, 2, 0] = 0.5
    labels[3, 2, 0, 0] = 0.5  # invalid cell
    labels[0, 0, 0, 0] = 0.5  # outside the mining pool
    mask = torch.ones(4, 4, 3, dtype=torch.bool)
    mask[3, 2, 0] = False
    state = torch.get_rng_state().clone()
    bank, report = mine_kernel_rows(
        forward, vs, labels, mask, torch.tensor([3, 1]), batch_size=1,
    )
    torch.testing.assert_close(bank, torch.tensor([[1, 3, 2]]))
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert report["valid_rows"] == 23
    assert report["pool_rows_within_5pct"] == pytest.approx(22 / 23)
    assert report["selected_rows_by_mode"] == [0, 0, 0, 1]
    assert not bank.requires_grad


def test_mining_handles_no_difficult_rows():
    def forward(vs):
        return vs[:, :1, None].expand(-1, 4, 3)

    vs = torch.ones(2, 3)
    labels = torch.zeros(2, 4, 3, 3)
    labels[..., 0] = 1
    bank, report = mine_kernel_rows(
        forward, vs, labels, torch.ones(2, 4, 3, dtype=torch.bool), torch.arange(2),
    )
    assert bank.shape == (0, 3)
    assert report["selected_rows"] == 0
    assert report["pool_rows_within_5pct"] == 1.0


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is unavailable",
    ))],
)
def test_exact_row_loss_matches_full_jacobian_and_parameter_finite_difference(device):
    torch.manual_seed(31)
    model = FourHeadForwardModel(
        input_size=3, width=8, blocks=1, frequencies=2,
    ).to(device=device, dtype=torch.float64)
    vs = torch.randn(2, 3, dtype=torch.float64, device=device)
    target = torch.randn(2, 3, dtype=torch.float64, device=device)
    modes = torch.tensor([1, 3], device=device)
    frequencies = torch.tensor([0, 1], device=device)

    def objective():
        return exact_row_kernel_loss(model, vs, target, modes, frequencies)

    loss, relative = objective()
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    jacobian = torch.vmap(torch.func.jacrev(lambda x: model(x[None])[0]))(vs)
    rows = jacobian[torch.arange(2, device=device), modes, frequencies]
    error2 = (rows - target).square().sum(-1)
    norm2 = target.square().sum(-1)
    torch.testing.assert_close(relative, error2.sqrt() / norm2.sqrt())
    percent = (error2 / (norm2 + 1e-4)).sqrt() * 100
    # Independent library evaluation of the intended robust scalar objective.
    assert loss.item() == pytest.approx(
        2 * pseudo_huber(5, percent.detach().cpu().numpy()).mean(), rel=1e-12,
    )
    reference = (50 * ((1 + percent.square() / 25).sqrt() - 1)).mean()
    expected = torch.autograd.grad(reference, parameters, allow_unused=True)
    for actual, wanted in zip(gradients, expected, strict=True):
        if wanted is None:
            assert actual is None or torch.count_nonzero(actual) == 0
        else:
            torch.testing.assert_close(actual, wanted, rtol=1e-8, atol=1e-8)

    parameter = model.input[0].weight
    original = parameter[0, 0].item()
    step = 1e-5
    try:
        with torch.no_grad():
            parameter[0, 0] = original + step
        upper = objective()[0].item()
        with torch.no_grad():
            parameter[0, 0] = original - step
        lower = objective()[0].item()
    finally:
        with torch.no_grad():
            parameter[0, 0] = original
    assert gradients[0][0, 0].item() == pytest.approx(
        (upper - lower) / (2 * step), rel=1e-6, abs=1e-5,
    )


def test_exact_row_training_reduces_the_selected_error_without_changing_other_rows():
    weights = torch.nn.Parameter(torch.ones(4, 3, 2))

    def forward(vs):
        return torch.einsum("bl,mfl->bmf", vs, weights)

    vs = torch.ones(4, 2)
    labels = torch.ones(4, 4, 3, 2)
    labels[:, 3, 1, 0] = 1.2
    mask = torch.ones(4, 4, 3, dtype=torch.bool)
    bank, _ = mine_kernel_rows(forward, vs, labels, mask, torch.arange(4))
    models, modes, frequencies = bank.unbind(1)
    target = labels[models, modes, frequencies]
    _, before = exact_row_kernel_loss(forward, vs[models], target, modes, frequencies)
    optimizer = torch.optim.Adam([weights], lr=0.01)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = exact_row_kernel_loss(forward, vs[models], target, modes, frequencies)
        loss.backward()
        optimizer.step()
    _, after = exact_row_kernel_loss(forward, vs[models], target, modes, frequencies)
    assert torch.all(after < 0.05)
    assert torch.all(after < before)
    other_rows = torch.ones(4, 3, dtype=torch.bool)
    other_rows[3, 1] = False
    torch.testing.assert_close(weights.detach()[other_rows], torch.ones(11, 2))
    assert np.isfinite(float(loss.detach()))
