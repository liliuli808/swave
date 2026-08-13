"""Sensitivity-weighted supervised priors for differentiable inversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from .inversion import DifferentiableSurrogate, SurrogateObjective


def _layer_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (20,):
        raise ValueError(f"{name} must contain 20 values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.copy()


def mean_dimensionless_sensitivity(
    surrogate: DifferentiableSurrogate,
    model: ArrayLike,
    valid_mask: ArrayLike,
    mode_weights: tuple[float, float, float, float],
    *,
    phase_floor: float = 1e-12,
) -> NDArray[np.float64]:
    """Average absolute dimensionless dispersion sensitivity for every layer."""
    values = _layer_vector(model, "sensitivity model")
    mask = np.asarray(valid_mask, dtype=np.bool_)
    weights = np.asarray(mode_weights, dtype=np.float64)
    if weights.shape != (4,) or not np.all(np.isfinite(weights)):
        raise ValueError("mode_weights must contain four finite values")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("mode_weights must be nonnegative with one positive value")
    if not np.isfinite(phase_floor) or phase_floor <= 0:
        raise ValueError("phase_floor must be finite and positive")

    tensor = torch.as_tensor(values, device=surrogate.device, dtype=torch.float64)
    prediction = surrogate.predict_tensor(tensor)
    if prediction.ndim != 2 or prediction.shape[0] != 4:
        raise ValueError("surrogate prediction must have shape (4, frequency_count)")
    if mask.shape != tuple(prediction.shape):
        raise ValueError("valid_mask must match the surrogate prediction shape")
    jacobian = torch.func.jacfwd(surrogate.predict_tensor)(tensor)
    expected_shape = (*prediction.shape, 20)
    if tuple(jacobian.shape) != expected_shape:
        raise ValueError("surrogate Jacobian has an invalid shape")
    if not bool(torch.isfinite(prediction).all()) or not bool(
        torch.isfinite(jacobian).all()
    ):
        raise ArithmeticError("surrogate sensitivity calculation is non-finite")

    prediction_values = np.asarray(
        prediction.detach().cpu().numpy(), dtype=np.float64
    )
    jacobian_values = np.asarray(
        jacobian.detach().cpu().numpy(), dtype=np.float64
    )
    active = mask & (weights[:, None] > 0)
    if not np.any(active):
        raise ValueError("sensitivity requires at least one weighted valid cell")
    if np.any(np.abs(prediction_values[active]) <= phase_floor):
        raise ArithmeticError("active predicted phase velocity is too close to zero")

    dimensionless = np.abs(
        jacobian_values
        * values[None, None, :]
        / prediction_values[:, :, None]
    )
    cell_weights = weights[:, None] * active
    denominator = float(cell_weights.sum())
    sensitivity = (
        dimensionless * cell_weights[:, :, None]
    ).sum(axis=(0, 1)) / denominator
    if not np.all(np.isfinite(sensitivity)) or np.any(sensitivity < 0):
        raise ArithmeticError("layer sensitivity is invalid")
    if not np.any(sensitivity > 0):
        raise ArithmeticError("layer sensitivity must contain a positive value")
    return np.asarray(sensitivity, dtype=np.float64)


def inverse_sensitivity_weights(
    sensitivity: ArrayLike,
    *,
    epsilon_fraction: float = 1e-2,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> NDArray[np.float64]:
    """Return inverse-sensitivity weights with exact unit mean and hard bounds."""
    values = np.asarray(sensitivity, dtype=np.float64)
    if values.shape != (20,):
        raise ValueError("sensitivity must contain 20 values")
    if not np.all(np.isfinite(values)):
        raise ValueError("sensitivity must be finite")
    if np.any(values < 0) or not np.any(values > 0):
        raise ValueError("sensitivity must be nonnegative with a positive value")
    if not np.isfinite(epsilon_fraction) or epsilon_fraction <= 0:
        raise ValueError("epsilon_fraction must be finite and positive")
    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or not 0 < minimum <= 1.0 <= maximum
    ):
        raise ValueError("prior weight bounds must contain one and be positive")
    if minimum == maximum:
        return np.ones(20, dtype=np.float64)

    epsilon = epsilon_fraction * float(values.mean())
    reciprocal = 1.0 / (values + epsilon)
    lower_scale = 0.0
    upper_scale = maximum / float(reciprocal.min()) * 2.0
    for _ in range(200):
        scale = (lower_scale + upper_scale) / 2.0
        total = float(np.clip(scale * reciprocal, minimum, maximum).sum())
        if total < 20.0:
            lower_scale = scale
        else:
            upper_scale = scale
    weights = np.clip(
        (lower_scale + upper_scale) / 2.0 * reciprocal,
        minimum,
        maximum,
    )
    if not np.isclose(weights.mean(), 1.0, rtol=0.0, atol=1e-12):
        raise ArithmeticError("prior weights could not be normalized")
    return np.asarray(weights, dtype=np.float64)


@dataclass(frozen=True)
class LearningPrior:
    """One supervised target and its fixed sensitivity-derived layer weights."""

    vs: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    weights: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "vs", _layer_vector(self.vs, "learning prior"))
        object.__setattr__(
            self, "sensitivity", _layer_vector(self.sensitivity, "sensitivity")
        )
        object.__setattr__(
            self, "weights", _layer_vector(self.weights, "prior weights")
        )
        if np.any(self.sensitivity < 0) or not np.any(self.sensitivity > 0):
            raise ValueError("sensitivity must be nonnegative with a positive value")
        if np.any(self.weights <= 0):
            raise ValueError("prior weights must be positive")


@dataclass(frozen=True)
class HybridObjectiveTerms:
    """Separately auditable components of the hybrid objective."""

    total: float
    data_misfit: float
    smoothness_regularization: float
    learning_prior_regularization: float


@dataclass
class HybridSurrogateObjective(SurrogateObjective):
    """Base surrogate objective augmented by a fixed weighted learning prior."""

    learning_prior: ArrayLike
    prior_weights: ArrayLike
    prior_lambda: float

    def __post_init__(self) -> None:
        super().__post_init__()
        prior = _layer_vector(self.learning_prior, "learning_prior")
        weights = _layer_vector(self.prior_weights, "prior_weights")
        if np.any(weights <= 0):
            raise ValueError("prior_weights must be positive")
        if not np.isfinite(self.prior_lambda) or self.prior_lambda < 0:
            raise ValueError("prior_lambda must be finite and nonnegative")
        self.learning_prior = prior
        self.prior_weights = weights
        self._learning_prior_tensor = torch.as_tensor(
            prior, device=self.surrogate.device, dtype=torch.float64
        )
        self._prior_weights_tensor = torch.as_tensor(
            weights, device=self.surrogate.device, dtype=torch.float64
        )

    def _components(
        self, vs: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        base_total, data_misfit, smoothness, prediction = super()._calculate(vs)
        difference = vs - self._learning_prior_tensor
        learning_prior = self.prior_lambda / 20.0 * torch.sum(
            self._prior_weights_tensor * torch.square(difference)
        )
        total = base_total + learning_prior
        if not bool(torch.isfinite(total)):
            raise ArithmeticError("hybrid objective is non-finite")
        return total, data_misfit, smoothness, learning_prior, prediction

    def _calculate(self, vs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        total, data_misfit, smoothness, learning_prior, prediction = self._components(
            vs
        )
        return total, data_misfit, smoothness + learning_prior, prediction

    def detailed_terms(self, vs: ArrayLike) -> HybridObjectiveTerms:
        """Evaluate the four scalar terms without constructing gradients."""
        vs_tensor = self._vs_tensor(vs, requires_grad=False)
        with torch.no_grad():
            total, data_misfit, smoothness, learning_prior, _ = self._components(
                vs_tensor
            )
        return HybridObjectiveTerms(
            total=float(total.item()),
            data_misfit=float(data_misfit.item()),
            smoothness_regularization=float(smoothness.item()),
            learning_prior_regularization=float(learning_prior.item()),
        )
