import pytest
import torch

from swave.network import (
    PROFILE_FEATURE_COUNT,
    FourHeadForwardModel,
    masked_smooth_l1,
    model_from_checkpoint,
    profile_feature_expansion,
)


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


def test_profile_features_localize_anomalies() -> None:
    vs = torch.linspace(1.0, 2.0, steps=20).unsqueeze(0)
    normal = profile_feature_expansion(vs)
    assert normal.shape == (1, PROFILE_FEATURE_COUNT)
    assert torch.all(normal[:, :40] == 0)  # monotone profile: no deficit/excess

    low = vs.clone()
    low[0, 8:10] *= 0.8  # thin low-velocity zone
    features = profile_feature_expansion(low)
    deficit = features[0, :20]
    assert deficit[8:10].min() > 0.05
    assert deficit.sum() > 0
    assert torch.argmax(deficit).item() in (8, 9)

    high = vs.clone()
    high[0, 12] *= 1.2  # thin high-velocity layer
    features = profile_feature_expansion(high)
    excess = features[0, 20:40]
    assert excess[12] > 0.05
    assert excess.sum() > 0


def test_profile_feature_model_keeps_vs_interface() -> None:
    model = FourHeadForwardModel(width=32, blocks=1, profile_features=True)
    vs = torch.randn(5, 20)
    output = model(vs)
    assert output.shape == (5, 4, 120)
    # Jacobians flow through the expansion (needed for kernel/inversion use);
    # mirror the inversion wrapper, which unsqueezes a single profile.
    jacobian = torch.func.jacfwd(lambda v: model(v.unsqueeze(0)).squeeze(0))(vs[0])
    assert jacobian.shape == (4, 120, 20)
    assert torch.isfinite(jacobian).all()


def test_model_from_checkpoint_profile_features_roundtrip() -> None:
    model = FourHeadForwardModel(width=32, blocks=1, profile_features=True)
    restored = model_from_checkpoint(
        {"architecture": {"width": 32, "blocks": 1, "profile_features": True},
         "model": model.state_dict()}
    )
    assert restored.profile_features
    vs = torch.randn(3, 20)
    assert torch.allclose(restored(vs), model(vs))
    # Old checkpoints without the flag still load with the default off.
    plain = FourHeadForwardModel(width=32, blocks=1)
    legacy = model_from_checkpoint(
        {"architecture": {"width": 32, "blocks": 1}, "model": plain.state_dict()}
    )
    assert not legacy.profile_features


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_layer_norm_preserves_native_outputs_and_input_jacobians(dtype) -> None:
    from swave.network import HigherOrderLayerNorm

    torch.manual_seed(8)
    native = torch.nn.LayerNorm(16).to(dtype=dtype)
    replacement = HigherOrderLayerNorm(16).to(dtype=dtype)
    replacement.load_state_dict(native.state_dict(), strict=True)
    values = torch.randn(3, 16, dtype=dtype)
    tangent = torch.randn_like(values)
    expected = torch.func.jvp(native, (values,), (tangent,))
    actual = torch.func.jvp(replacement, (values,), (tangent,))
    torch.testing.assert_close(actual, expected)
