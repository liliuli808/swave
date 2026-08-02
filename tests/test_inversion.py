from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

import swave.inversion as inversion_module
from swave.inference import resolve_device
from swave.inversion import (
    DifferentiableSurrogate,
    EnsembleResult,
    InversionRun,
    ObjectiveTerms,
    SurrogateObjective,
    apply_observation_noise,
    build_reference_model,
    generate_initial_models,
    invert_ensemble,
    invert_one,
    iqr_inlier_mask,
    regularization_matrix,
)


class ToyForward(nn.Module):
    """A fixed linear forward map with four three-frequency heads."""

    def __init__(self) -> None:
        super().__init__()
        coefficients = torch.arange(1, 241, dtype=torch.float64).reshape(12, 20)
        self.register_buffer("coefficients", coefficients / 1000.0)
        self.register_buffer("bias", torch.linspace(-0.2, 0.2, 12))

    def forward(self, values: Tensor) -> Tensor:
        prediction = values @ self.coefficients.T + self.bias
        return prediction.reshape(values.shape[0], 4, 3)


def _toy_surrogate() -> DifferentiableSurrogate:
    return DifferentiableSurrogate(
        model=ToyForward(),
        input_mean=torch.zeros(20, dtype=torch.float64),
        input_std=torch.ones(20, dtype=torch.float64),
        target_mean=torch.zeros((4, 1), dtype=torch.float64),
        target_std=torch.ones((4, 1), dtype=torch.float64),
        device=torch.device("cpu"),
    )


class QuadraticObjective:
    """Small optimizer fixture with an independently known bounded solution."""

    frequencies = np.array([1.0, 2.0, 3.0])

    def __init__(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=np.float64)

    def value_and_grad(self, values):
        difference = np.asarray(values, dtype=np.float64) - self.target
        return float(np.sum(np.square(difference))), 2.0 * difference

    def terms(self, values) -> ObjectiveTerms:
        difference = np.asarray(values, dtype=np.float64) - self.target
        total = float(np.sum(np.square(difference)))
        return ObjectiveTerms(total=total, data_misfit=total, regularization=0.0)

    def predict(self, values):
        return np.full((4, 3), np.asarray(values, dtype=np.float64).mean())


def test_reference_uses_only_fundamental_observation() -> None:
    frequencies = np.arange(0.5, 60.0 + 0.25, 0.5)
    observed = np.full((4, 120), 99.0)
    mask = np.ones((4, 120), dtype=bool)
    observed[0] = 1.0

    reference = build_reference_model(
        frequencies, observed, mask, vs_min=0.3, vs_max=2.6, vs_width=0.7
    )

    np.testing.assert_allclose(reference.vs, 1.1)
    np.testing.assert_allclose(reference.lower, 0.75)
    np.testing.assert_allclose(reference.upper, 1.45)
    changed_higher_modes = observed.copy()
    changed_higher_modes[1:] = -1234.0
    changed = build_reference_model(
        frequencies,
        changed_higher_modes,
        mask,
        vs_min=0.3,
        vs_max=2.6,
        vs_width=0.7,
    )
    np.testing.assert_array_equal(changed.vs, reference.vs)


def test_reference_requires_two_finite_valid_fundamental_cells() -> None:
    frequencies = np.array([1.0, 2.0, 3.0])
    observed = np.full((4, 3), np.nan)
    mask = np.zeros((4, 3), dtype=bool)
    observed[0, 0] = 1.0
    mask[0, 0] = True
    observed[1:] = 1.5
    mask[1:] = True

    with pytest.raises(ValueError, match="two finite.*fundamental"):
        build_reference_model(
            frequencies,
            observed,
            mask,
            vs_min=0.3,
            vs_max=2.6,
            vs_width=0.7,
        )


def test_one_percent_noise_is_reproducible_and_preserves_mask() -> None:
    observed = np.ones((4, 120), dtype=np.float64)
    mask = np.ones_like(observed, dtype=bool)
    mask[3, :4] = False

    first = apply_observation_noise(observed, mask, "noise_1pct", 7, 90)
    second = apply_observation_noise(observed, mask, "noise_1pct", 7, 90)
    clean = apply_observation_noise(observed, mask, "clean", 7, 90)

    np.testing.assert_array_equal(first, second)
    assert np.all(np.isnan(first[~mask]))
    assert np.all(first[mask] != observed[mask])
    np.testing.assert_array_equal(clean[mask], observed[mask])
    assert np.all(np.isnan(clean[~mask]))


def test_regularization_matrix_uses_approved_adaptive_weights() -> None:
    reference = np.array([0.0, 1.0, 3.0] + [3.0] * 17)
    adaptive = regularization_matrix(reference, "adaptive")
    expected_q = np.array([1.0 / 6.0, 1.0 / 11.0, 1.0] + [1.0] * 16)

    assert adaptive.shape == (19, 20)
    np.testing.assert_allclose(np.diag(adaptive[:, :-1]), expected_q)
    np.testing.assert_allclose(np.diag(adaptive[:, 1:]), -expected_q)
    assert np.count_nonzero(adaptive) == 38
    first_order = regularization_matrix(reference, "first_order")
    np.testing.assert_array_equal(np.diag(first_order[:, :-1]), np.ones(19))
    np.testing.assert_array_equal(np.diag(first_order[:, 1:]), -np.ones(19))
    constant = regularization_matrix(np.ones(20), "adaptive")
    np.testing.assert_array_equal(constant, first_order)


def test_initial_models_are_bounded_and_deterministic_by_scenario() -> None:
    frequencies = np.array([1.0, 2.0])
    observed = np.ones((4, 2))
    mask = np.ones((4, 2), dtype=bool)
    reference = build_reference_model(
        frequencies, observed, mask, vs_min=0.3, vs_max=2.6, vs_width=0.7
    )

    first = generate_initial_models(reference, 5, 17, 90, "noise_1pct")
    second = generate_initial_models(reference, 5, 17, 90, "noise_1pct")
    clean = generate_initial_models(reference, 5, 17, 90, "clean")

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], reference.vs)
    assert np.all(first >= reference.lower)
    assert np.all(first <= reference.upper)
    assert not np.array_equal(first[1:], clean[1:])


def test_objective_terms_respect_masks_weights_and_regularization() -> None:
    surrogate = _toy_surrogate()
    values = np.linspace(0.4, 1.6, 20)
    prediction = (
        values @ surrogate.model.coefficients.numpy().T + surrogate.model.bias.numpy()
    ).reshape(4, 3)
    observed = prediction.copy()
    observed[0] += np.array([1.0, 2.0, 1000.0])
    observed[1] += 1000.0
    observed[2] += np.array([3.0, 0.0, 0.0])
    observed[3] = np.nan
    mask = np.array(
        [
            [True, True, False],
            [True, True, True],
            [True, False, False],
            [False, False, False],
        ]
    )
    reference = np.full(20, 1.0)
    regularization = regularization_matrix(reference, "first_order")
    objective = SurrogateObjective(
        surrogate=surrogate,
        frequencies=np.array([1.0, 2.0, 3.0]),
        observed=observed,
        valid_mask=mask,
        mode_weights=(4.0, 0.0, 2.0, 1.0),
        reference=reference,
        regularization=regularization,
        regularization_lambda=0.5,
    )

    terms = objective.terms(values)
    expected_data = 4.0 * (1.0**2 + 2.0**2) / 2.0 + 2.0 * 3.0**2
    difference = values - reference
    expected_regularization = (
        0.5 / 20.0 * np.sum(np.square(regularization @ difference))
    )

    assert terms.data_misfit == pytest.approx(expected_data)
    assert terms.regularization == pytest.approx(expected_regularization)
    assert terms.total == pytest.approx(expected_data + expected_regularization)
    np.testing.assert_allclose(objective.predict(values), prediction)


def test_autograd_gradient_matches_central_finite_differences() -> None:
    surrogate = _toy_surrogate()
    values = np.linspace(0.6, 1.5, 20)
    observed = np.linspace(0.7, 2.0, 12).reshape(4, 3)
    reference = np.linspace(0.5, 1.4, 20)
    objective = SurrogateObjective(
        surrogate=surrogate,
        frequencies=np.array([1.0, 2.0, 3.0]),
        observed=observed,
        valid_mask=np.ones((4, 3), dtype=bool),
        mode_weights=(4.0, 1.0, 0.5, 2.0),
        reference=reference,
        regularization=regularization_matrix(reference, "adaptive"),
        regularization_lambda=0.2,
    )

    value, gradient = objective.value_and_grad(values)
    h = 1e-6
    finite_difference = np.empty(20)
    for index in range(20):
        forward = values.copy()
        backward = values.copy()
        forward[index] += h
        backward[index] -= h
        finite_difference[index] = (
            objective.terms(forward).total - objective.terms(backward).total
        ) / (2.0 * h)

    assert value == pytest.approx(objective.terms(values).total)
    assert gradient.dtype == np.float64
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-7)


def test_objective_requires_a_positive_weight_observed_cell() -> None:
    with pytest.raises(ValueError, match="used modal cell"):
        SurrogateObjective(
            surrogate=_toy_surrogate(),
            frequencies=np.array([1.0, 2.0, 3.0]),
            observed=np.ones((4, 3)),
            valid_mask=np.zeros((4, 3), dtype=bool),
            mode_weights=(4.0, 1.0, 1.0, 1.0),
            reference=np.ones(20),
            regularization=np.zeros((19, 20)),
            regularization_lambda=0.1,
        )


def test_differentiable_surrogate_loads_double_checkpoint(
    tiny_checkpoint: Path,
) -> None:
    surrogate = DifferentiableSurrogate.load(tiny_checkpoint, "cpu")

    assert next(surrogate.model.parameters()).dtype == torch.float64
    assert surrogate.input_mean.dtype == torch.float64
    prediction = surrogate.predict_tensor(torch.ones(20, dtype=torch.float64))
    assert prediction.shape == (4, 120)
    assert prediction.requires_grad


def test_inversion_auto_uses_cpu_when_only_mps_is_available(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("mps")
    assert inversion_module.resolve_inversion_device("auto") == torch.device("cpu")


def test_inversion_rejects_mps_before_loading_checkpoint(monkeypatch) -> None:
    def fail_checkpoint_load(*args, **kwargs):
        raise AssertionError("checkpoint loading must not begin for MPS")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch, "load", fail_checkpoint_load)

    with pytest.raises(ValueError, match="MPS.*float64"):
        DifferentiableSurrogate.load(Path("unused.pt"), "mps")


def test_lbfgsb_recovers_bounded_quadratic_solution() -> None:
    objective = QuadraticObjective(target=np.linspace(0.5, 2.4, 20))
    initial = np.full(20, 1.0)
    lower = np.full(20, 0.3)
    upper = np.full(20, 2.0)

    result = invert_one(
        objective,
        initial,
        lower,
        upper,
        max_iterations=100,
        relative_tolerance=1e-9,
    )

    np.testing.assert_allclose(result.vs, np.minimum(objective.target, 2.0), atol=1e-6)
    assert result.success
    assert np.all((result.vs >= lower) & (result.vs <= upper))
    assert result.initial_objective == pytest.approx(objective.terms(initial).total)
    assert result.terms.total == pytest.approx(objective.terms(result.vs).total)
    assert result.predicted_phase_velocity.shape == (4, 3)


@pytest.mark.parametrize(
    ("callback_result", "message"),
    [
        ((np.nan, np.zeros(20)), "objective value"),
        ((np.array([1.0]), np.zeros(20)), "objective value"),
        ((1.0, np.full(20, np.nan)), "objective gradient"),
        ((1.0, np.zeros(19)), "objective gradient"),
    ],
)
def test_lbfgsb_rejects_invalid_objective_callback_results(
    callback_result, message
) -> None:
    objective = QuadraticObjective(target=np.ones(20))
    objective.value_and_grad = lambda values: callback_result

    with pytest.raises((ArithmeticError, ValueError), match=message):
        invert_one(
            objective,
            np.ones(20),
            np.full(20, 0.3),
            np.full(20, 2.0),
            max_iterations=10,
            relative_tolerance=1e-5,
        )


def test_lbfgsb_validates_models_and_bounds_before_evaluation() -> None:
    objective = QuadraticObjective(target=np.ones(20))

    with pytest.raises(ValueError, match="initial.*20 finite"):
        invert_one(
            objective,
            np.ones(19),
            np.full(20, 0.3),
            np.full(20, 2.0),
            max_iterations=10,
            relative_tolerance=1e-5,
        )
    with pytest.raises(ValueError, match="lower.*upper"):
        invert_one(
            objective,
            np.ones(20),
            np.full(20, 2.0),
            np.full(20, 0.3),
            max_iterations=10,
            relative_tolerance=1e-5,
        )


@pytest.mark.parametrize(
    "recovered",
    [np.full(20, 2.1), np.full(20, np.nan)],
)
def test_lbfgsb_rejects_invalid_optimizer_results(monkeypatch, recovered) -> None:
    def invalid_minimize(*args, **kwargs):
        return inversion_module.scipy.optimize.OptimizeResult(
            x=recovered,
            success=True,
            status=0,
            message="invalid fixture",
            nit=1,
            nfev=1,
        )

    monkeypatch.setattr(inversion_module.scipy.optimize, "minimize", invalid_minimize)

    with pytest.raises(ArithmeticError, match="invalid bounded model"):
        invert_one(
            QuadraticObjective(target=np.ones(20)),
            np.ones(20),
            np.full(20, 0.3),
            np.full(20, 2.0),
            max_iterations=10,
            relative_tolerance=1e-5,
        )


def test_iqr_inliers_ignore_nonfinite_values_and_apply_zero_iqr_fences() -> None:
    keep = iqr_inlier_mask(np.array([1.0] * 9 + [100.0, np.nan, np.inf]))

    assert keep.tolist() == [True] * 9 + [False, False, False]


def test_ensemble_uses_only_successful_finite_inliers_for_percentiles(
    monkeypatch,
) -> None:
    profiles = np.array([0.8, 1.0, 1.2, 1.4, 2.5, 0.0])
    objectives = np.array([1.0, 2.0, 2.0, 3.0, 100.0, 0.0])
    successes = np.array([True, True, True, True, True, False])

    def fake_invert_one(objective, initial, lower, upper, **options):
        index = int(initial[0])
        value = profiles[index]
        return InversionRun(
            vs=np.full(20, value),
            predicted_phase_velocity=np.full((4, 3), value),
            success=bool(successes[index]),
            status=index,
            message=f"run-{index}",
            iterations=index,
            evaluations=index + 1,
            initial_objective=float(objectives[index] + 10.0),
            terms=ObjectiveTerms(
                total=float(objectives[index]),
                data_misfit=float(objectives[index]),
                regularization=0.0,
            ),
        )

    monkeypatch.setattr(inversion_module, "invert_one", fake_invert_one)
    objective = QuadraticObjective(target=np.zeros(20))
    starts = np.repeat(np.arange(6, dtype=np.float64)[:, None], 20, axis=1)

    result = invert_ensemble(
        objective,
        starts,
        np.full(20, -1.0),
        np.full(20, 6.0),
        max_iterations=10,
        relative_tolerance=1e-5,
        minimum_valid_solutions=4,
    )

    assert isinstance(result, EnsembleResult)
    assert [run.message for run in result.runs] == [
        f"run-{index}" for index in range(6)
    ]
    assert result.inlier_mask.tolist() == [True, True, True, True, False, False]
    assert result.sufficient
    np.testing.assert_allclose(result.median_vs, 1.1)
    np.testing.assert_allclose(result.p10_vs, 0.86)
    np.testing.assert_allclose(result.p90_vs, 1.34)
    assert result.representative_terms.total == pytest.approx(20 * 1.1**2)
    np.testing.assert_allclose(result.representative_prediction, 1.1)


def test_ensemble_contains_per_start_failures_and_continues_in_order() -> None:
    class SelectiveFailureObjective(QuadraticObjective):
        def value_and_grad(self, values):
            value = float(np.asarray(values)[0])
            if value == pytest.approx(0.9):
                raise RuntimeError("deliberate optimizer failure")
            return value, np.zeros(20)

        def terms(self, values) -> ObjectiveTerms:
            value = float(np.asarray(values)[0])
            return ObjectiveTerms(value, value, 0.0)

    objective = SelectiveFailureObjective(target=np.ones(20))
    starts = np.repeat(np.array([0.7, 0.9, 1.1])[:, None], 20, axis=1)

    result = invert_ensemble(
        objective,
        starts,
        np.full(20, 0.3),
        np.full(20, 2.0),
        max_iterations=10,
        relative_tolerance=1e-5,
        minimum_valid_solutions=2,
    )

    assert [run.success for run in result.runs] == [True, False, True]
    assert result.runs[1].status == -1
    assert "RuntimeError" in result.runs[1].message
    assert np.all(np.isnan(result.runs[1].vs))
    assert np.all(np.isnan(result.runs[1].predicted_phase_velocity))
    assert np.isnan(result.runs[1].terms.total)
    assert result.inlier_mask.tolist() == [True, False, True]
    assert result.sufficient


@pytest.mark.parametrize("invalid_value", [np.nan, 2.1])
def test_ensemble_contains_invalid_initial_rows_and_runs_later_starts(
    invalid_value,
) -> None:
    objective = QuadraticObjective(target=np.ones(20))
    starts = np.repeat(np.array([0.8, invalid_value, 1.2])[:, None], 20, axis=1)

    result = invert_ensemble(
        objective,
        starts,
        np.full(20, 0.3),
        np.full(20, 2.0),
        max_iterations=10,
        relative_tolerance=1e-5,
        minimum_valid_solutions=2,
    )

    assert [run.success for run in result.runs] == [True, False, True]
    assert result.runs[1].status == -1
    assert "ValueError" in result.runs[1].message
    assert result.inlier_mask.tolist() == [True, False, True]
    assert result.sufficient


def test_early_failed_run_uses_later_successful_prediction_shape() -> None:
    class ObjectiveWithoutFrequencies:
        def value_and_grad(self, values):
            value = float(np.asarray(values)[0])
            if value == pytest.approx(0.8):
                raise RuntimeError("first start fails")
            return value, np.zeros(20)

        def terms(self, values) -> ObjectiveTerms:
            value = float(np.asarray(values)[0])
            return ObjectiveTerms(value, value, 0.0)

        def predict(self, values):
            return np.zeros((4, 3), dtype=np.float64)

    starts = np.repeat(np.array([0.8, 1.2])[:, None], 20, axis=1)
    result = invert_ensemble(
        ObjectiveWithoutFrequencies(),
        starts,
        np.full(20, 0.3),
        np.full(20, 2.0),
        max_iterations=10,
        relative_tolerance=1e-5,
        minimum_valid_solutions=1,
    )

    assert not result.runs[0].success
    assert result.runs[0].predicted_phase_velocity.shape == (4, 3)
    assert np.all(np.isnan(result.runs[0].predicted_phase_velocity))
    assert result.runs[1].predicted_phase_velocity.shape == (4, 3)


def test_insufficient_ensemble_preserves_runs_but_returns_nan_statistics(
    monkeypatch,
) -> None:
    def failed_invert_one(objective, initial, lower, upper, **options):
        return InversionRun(
            vs=np.full(20, float(initial[0])),
            predicted_phase_velocity=np.zeros((4, 3)),
            success=False,
            status=2,
            message="not converged",
            iterations=10,
            evaluations=11,
            initial_objective=2.0,
            terms=ObjectiveTerms(1.0, 1.0, 0.0),
        )

    monkeypatch.setattr(inversion_module, "invert_one", failed_invert_one)
    result = invert_ensemble(
        QuadraticObjective(target=np.ones(20)),
        np.ones((3, 20)),
        np.full(20, 0.3),
        np.full(20, 2.0),
        max_iterations=10,
        relative_tolerance=1e-5,
        minimum_valid_solutions=2,
    )

    assert len(result.runs) == 3
    assert not result.sufficient
    assert not np.any(result.inlier_mask)
    assert np.all(np.isnan(result.median_vs))
    assert np.all(np.isnan(result.p10_vs))
    assert np.all(np.isnan(result.p90_vs))
    assert np.isnan(result.representative_terms.total)
    assert np.all(np.isnan(result.representative_prediction))
