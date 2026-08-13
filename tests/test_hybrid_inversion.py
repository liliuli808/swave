from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from swave.hybrid_inversion import (
    HybridSurrogateObjective,
    inverse_sensitivity_weights,
    mean_dimensionless_sensitivity,
)
from swave.inversion import DifferentiableSurrogate, SurrogateObjective


class LinearForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        coefficients = (
            np.arange(1, 12 * 20 + 1, dtype=np.float64).reshape(12, 20) / 500.0
        )
        bias = np.linspace(0.8, 1.3, 12, dtype=np.float64)
        self.register_buffer("coefficients", torch.from_numpy(coefficients))
        self.register_buffer("bias", torch.from_numpy(bias))

    def forward(self, values: Tensor) -> Tensor:
        prediction = values @ self.coefficients.T + self.bias
        return prediction.reshape(values.shape[0], 4, 3)


class InactiveZeroForward(LinearForward):
    def forward(self, values: Tensor) -> Tensor:
        prediction = super().forward(values).clone()
        prediction[:, 3, 2] = 0.0
        return prediction


def _linear_surrogate() -> DifferentiableSurrogate:
    return DifferentiableSurrogate(
        model=LinearForward(),
        input_mean=torch.zeros(20, dtype=torch.float64),
        input_std=torch.ones(20, dtype=torch.float64),
        target_mean=torch.zeros((4, 1), dtype=torch.float64),
        target_std=torch.ones((4, 1), dtype=torch.float64),
        device=torch.device("cpu"),
    )


@pytest.mark.filterwarnings(
    "ignore:.*torch.jit.script.*deprecated.*:DeprecationWarning"
)
def test_mean_dimensionless_sensitivity_matches_linear_jacobian_and_mask() -> None:
    surrogate = _linear_surrogate()
    prior = np.linspace(0.5, 1.5, 20, dtype=np.float64)
    mask = np.ones((4, 3), dtype=np.bool_)
    mask[1, 2] = False
    mode_weights = (4.0, 2.0, 1.0, 0.0)

    actual = mean_dimensionless_sensitivity(
        surrogate,
        prior,
        mask,
        mode_weights,
        phase_floor=1e-12,
    )

    coefficients = np.arange(1, 241, dtype=np.float64).reshape(12, 20) / 500.0
    bias = np.linspace(0.8, 1.3, 12, dtype=np.float64)
    prediction = coefficients @ prior + bias
    dimensionless = np.abs(coefficients * prior[None, :] / prediction[:, None])
    active = mask.reshape(-1)
    cell_weights = np.repeat(np.asarray(mode_weights), 3) * active
    expected = (dimensionless * cell_weights[:, None]).sum(axis=0) / cell_weights.sum()

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_inactive_zero_prediction_does_not_enter_sensitivity_division() -> None:
    surrogate = _linear_surrogate()
    surrogate.model = InactiveZeroForward().to(dtype=torch.float64)
    mask = np.ones((4, 3), dtype=np.bool_)
    mask[3, 2] = False

    sensitivity = mean_dimensionless_sensitivity(
        surrogate,
        np.linspace(0.5, 1.5, 20),
        mask,
        (4.0, 2.0, 1.0, 1.0),
    )

    assert np.all(np.isfinite(sensitivity))
    assert np.all(sensitivity > 0)


def test_inverse_sensitivity_weights_are_bounded_normalized_and_reversed() -> None:
    sensitivity = np.geomspace(0.01, 10.0, 20)

    weights = inverse_sensitivity_weights(
        sensitivity,
        epsilon_fraction=1e-2,
        minimum=0.25,
        maximum=4.0,
    )

    assert np.all(weights >= 0.25)
    assert np.all(weights <= 4.0)
    assert weights.mean() == pytest.approx(1.0, abs=1e-12)
    assert weights[0] > weights[-1]
    assert np.all(np.diff(weights) <= 0)


@pytest.mark.parametrize(
    ("sensitivity", "kwargs", "message"),
    [
        (np.zeros(20), {}, "positive"),
        (np.full(20, np.nan), {}, "finite"),
        (np.ones(19), {}, "20"),
        (np.ones(20), {"minimum": 1.1}, "bounds"),
        (np.ones(20), {"maximum": 0.9}, "bounds"),
    ],
)
def test_inverse_sensitivity_weights_reject_invalid_inputs(
    sensitivity: np.ndarray,
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        inverse_sensitivity_weights(sensitivity, **kwargs)


def _objective_pair(
    prior_lambda: float,
) -> tuple[SurrogateObjective, HybridSurrogateObjective]:
    surrogate = _linear_surrogate()
    frequencies = np.array([1.0, 2.0, 3.0])
    observed = np.full((4, 3), 3.0)
    mask = np.ones((4, 3), dtype=np.bool_)
    reference = np.linspace(0.7, 1.4, 20)
    regularization = np.zeros((19, 20), dtype=np.float64)
    common = {
        "surrogate": surrogate,
        "frequencies": frequencies,
        "observed": observed,
        "valid_mask": mask,
        "mode_weights": (4.0, 1.0, 1.0, 1.0),
        "reference": reference,
        "regularization": regularization,
        "regularization_lambda": 0.01,
    }
    base = SurrogateObjective(**common)
    hybrid = HybridSurrogateObjective(
        **common,
        learning_prior=np.linspace(0.6, 1.6, 20),
        prior_weights=np.linspace(0.5, 1.5, 20),
        prior_lambda=prior_lambda,
    )
    return base, hybrid


def test_hybrid_objective_reports_prior_term_and_same_pass_gradient() -> None:
    _, objective = _objective_pair(prior_lambda=0.7)
    values = np.linspace(0.8, 1.8, 20)

    details = objective.detailed_terms(values)
    total, gradient = objective.value_and_grad(values)

    prior = np.linspace(0.6, 1.6, 20)
    weights = np.linspace(0.5, 1.5, 20)
    expected_prior = 0.7 / 20.0 * np.sum(weights * np.square(values - prior))
    assert details.learning_prior_regularization == pytest.approx(expected_prior)
    assert details.smoothness_regularization == pytest.approx(0.0)
    assert details.total == pytest.approx(
        details.data_misfit
        + details.smoothness_regularization
        + details.learning_prior_regularization
    )
    assert total == pytest.approx(details.total)

    finite_difference = np.empty(20)
    step = 1e-6
    for layer in range(20):
        upper = values.copy()
        lower = values.copy()
        upper[layer] += step
        lower[layer] -= step
        finite_difference[layer] = (
            objective.terms(upper).total - objective.terms(lower).total
        ) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-5, atol=2e-6)


def test_zero_learning_lambda_matches_global_bound_control_objective() -> None:
    base, hybrid = _objective_pair(prior_lambda=0.0)
    values = np.linspace(0.8, 1.8, 20)

    base_value, base_gradient = base.value_and_grad(values)
    hybrid_value, hybrid_gradient = hybrid.value_and_grad(values)

    assert hybrid_value == pytest.approx(base_value)
    np.testing.assert_allclose(hybrid_gradient, base_gradient)
    assert hybrid.detailed_terms(values).learning_prior_regularization == 0.0
