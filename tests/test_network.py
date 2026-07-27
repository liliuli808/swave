import pytest
import torch

from swave.network import FourHeadForwardModel, masked_smooth_l1


def test_network_output_shape() -> None:
    model = FourHeadForwardModel()
    output = model(torch.randn(7, 20))
    assert output.shape == (7, 4, 120)


def test_masked_loss_ignores_invalid_cells_and_balances_modes() -> None:
    prediction = torch.zeros(1, 4, 2)
    target = torch.ones(1, 4, 2)
    mask = torch.tensor(
        [[[1, 1], [1, 0], [0, 0], [1, 1]]], dtype=torch.bool
    )
    loss = masked_smooth_l1(prediction, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.5)


def test_empty_batch_mode_does_not_create_nan() -> None:
    prediction = torch.zeros(2, 4, 3)
    target = torch.zeros_like(prediction)
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[:, 0] = True
    assert torch.isfinite(masked_smooth_l1(prediction, target, mask))


def test_masked_loss_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        masked_smooth_l1(
            torch.zeros(2, 4, 120),
            torch.zeros(2, 4, 119),
            torch.zeros(2, 4, 120, dtype=torch.bool),
        )
