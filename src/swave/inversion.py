"""Leakage-free preprocessing and differentiable inversion objectives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import scipy.optimize
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from .config import NoiseScenario
from .inference import resolve_device
from .network import FourHeadForwardModel
from .splits import validate_checkpoint_split_policy


def resolve_inversion_device(requested: str) -> torch.device:
    """Resolve a device that supports the inversion's required float64 math."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        raise ValueError("MPS cannot be used for inversion because float64 is required")
    return resolve_device(requested)


@dataclass(frozen=True)
class ReferenceModel:
    """Observation-derived reference model and its local optimization bounds."""

    vs: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


@dataclass(frozen=True)
class ObjectiveTerms:
    """Scalar components of the surrogate inversion objective."""

    total: float
    data_misfit: float
    regularization: float


class InversionObjective(Protocol):
    """Optimizer-facing objective contract independent of dataset truth."""

    def value_and_grad(self, vs: ArrayLike) -> tuple[float, NDArray[np.float64]]: ...

    def terms(self, vs: ArrayLike) -> ObjectiveTerms: ...

    def predict(self, vs: ArrayLike) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class InversionRun:
    """Diagnostics and recovered model from one bounded optimization run."""

    vs: NDArray[np.float64]
    predicted_phase_velocity: NDArray[np.float64]
    success: bool
    status: int
    message: str
    iterations: int
    evaluations: int
    initial_objective: float
    terms: ObjectiveTerms


@dataclass(frozen=True)
class EnsembleResult:
    """Ordered run diagnostics and robust statistics for a multi-start inversion."""

    runs: tuple[InversionRun, ...]
    inlier_mask: NDArray[np.bool_]
    median_vs: NDArray[np.float64]
    p10_vs: NDArray[np.float64]
    p90_vs: NDArray[np.float64]
    representative_terms: ObjectiveTerms
    representative_prediction: NDArray[np.float64]
    sufficient: bool


def _smooth(values: NDArray[np.float64]) -> NDArray[np.float64]:
    current = values.copy()
    kernel = np.array([0.25, 0.5, 0.25])
    for _ in range(2):
        current = np.convolve(np.pad(current, 1, mode="reflect"), kernel, mode="valid")
    return np.asarray(current, dtype=np.float64)


def _dispersion_arrays(
    observed: ArrayLike, valid_mask: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    values = np.asarray(observed, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if values.ndim != 2 or values.shape[0] != 4:
        raise ValueError("observed must have shape (4, frequency_count)")
    if mask.shape != values.shape:
        raise ValueError("valid_mask must have the same shape as observed")
    if not np.all(np.isfinite(values[mask])):
        raise ValueError("valid observed cells must be finite")
    return values, mask


def build_reference_model(
    frequencies: ArrayLike,
    observed: ArrayLike,
    valid_mask: ArrayLike,
    *,
    vs_min: float,
    vs_max: float,
    vs_width: float,
) -> ReferenceModel:
    """Build a 20-layer reference using only valid fundamental observations."""
    frequency_values = np.asarray(frequencies, dtype=np.float64)
    phase_velocity, mask = _dispersion_arrays(observed, valid_mask)
    if frequency_values.ndim != 1 or frequency_values.shape[0] != mask.shape[1]:
        raise ValueError("frequencies must match the observation frequency axis")
    if not np.all(np.isfinite(frequency_values)) or np.any(frequency_values <= 0):
        raise ValueError("frequencies must be finite and positive")
    if (
        not np.isfinite([vs_min, vs_max, vs_width]).all()
        or not 0 < vs_min < vs_max
        or not 0 < vs_width <= vs_max - vs_min
    ):
        raise ValueError("Vs bounds and vs_width are invalid")

    fundamental = mask[0] & np.isfinite(phase_velocity[0])
    if np.count_nonzero(fundamental) < 2:
        raise ValueError("at least two finite valid fundamental cells are required")
    velocity = phase_velocity[0, fundamental]
    depth = (velocity / frequency_values[fundamental]) / 3.0
    layer_vs = 1.1 * velocity
    if not np.all(np.isfinite(depth)) or not np.all(np.isfinite(layer_vs)):
        raise ValueError("fundamental observations produced non-finite reference data")

    order = np.argsort(depth, kind="stable")
    sorted_depth = depth[order]
    sorted_vs = layer_vs[order]
    unique_depth, inverse = np.unique(sorted_depth, return_inverse=True)
    sums = np.bincount(inverse, weights=sorted_vs)
    counts = np.bincount(inverse)
    averaged_vs = sums / counts
    depth_grid = np.arange(20, dtype=np.float64) * 0.1
    interpolated = np.interp(depth_grid, unique_depth, averaged_vs)
    reference = np.clip(_smooth(interpolated), vs_min, vs_max)
    half_width = 0.5 * vs_width
    lower = np.maximum(vs_min, reference - half_width)
    upper = np.minimum(vs_max, reference + half_width)
    return ReferenceModel(
        vs=np.asarray(reference, dtype=np.float64),
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
    )


def apply_observation_noise(
    observed: ArrayLike,
    valid_mask: ArrayLike,
    scenario: NoiseScenario,
    seed: int,
    sample_id: int,
) -> NDArray[np.float64]:
    """Copy clean observations or apply deterministic 1% relative noise."""
    values, mask = _dispersion_arrays(observed, valid_mask)
    if scenario not in {"clean", "noise_1pct"}:
        raise ValueError("scenario must be clean or noise_1pct")
    if seed < 0 or sample_id < 0:
        raise ValueError("seed and sample_id must be nonnegative")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[mask] = values[mask]
    if scenario == "noise_1pct":
        rng = np.random.default_rng(np.random.SeedSequence([seed, sample_id, 1]))
        result[mask] *= 1.0 + 0.01 * rng.normal(size=int(mask.sum()))
    return result


def regularization_matrix(
    reference: ArrayLike, regularization_type: str
) -> NDArray[np.float64]:
    """Return the ordinary or QEDispInv-style adaptive first difference."""
    values = np.asarray(reference, dtype=np.float64)
    if values.shape != (20,) or not np.all(np.isfinite(values)):
        raise ValueError("reference must contain 20 finite values")
    if regularization_type not in {"adaptive", "first_order"}:
        raise ValueError("regularization_type must be adaptive or first_order")

    differences = np.abs(np.diff(values))
    if regularization_type == "first_order" or np.max(differences) == 0.0:
        weights = np.ones(19, dtype=np.float64)
    else:
        scale = 0.1 * np.max(differences)
        weights = scale / (scale + differences)
    matrix = np.zeros((19, 20), dtype=np.float64)
    rows = np.arange(19)
    matrix[rows, rows] = weights
    matrix[rows, rows + 1] = -weights
    return matrix


def generate_initial_models(
    reference: ReferenceModel,
    count: int,
    seed: int,
    sample_id: int,
    scenario: NoiseScenario,
) -> NDArray[np.float64]:
    """Generate deterministic, bounded smooth starts around the reference."""
    if count <= 0:
        raise ValueError("count must be positive")
    if seed < 0 or sample_id < 0:
        raise ValueError("seed and sample_id must be nonnegative")
    if scenario not in {"clean", "noise_1pct"}:
        raise ValueError("scenario must be clean or noise_1pct")
    values = np.asarray(reference.vs, dtype=np.float64)
    lower = np.asarray(reference.lower, dtype=np.float64)
    upper = np.asarray(reference.upper, dtype=np.float64)
    if any(array.shape != (20,) for array in (values, lower, upper)):
        raise ValueError("reference and bounds must contain 20 values")
    if not all(np.all(np.isfinite(array)) for array in (values, lower, upper)):
        raise ValueError("reference and bounds must be finite")
    if np.any(lower > values) or np.any(values > upper):
        raise ValueError("reference must lie within its bounds")

    starts = np.empty((count, 20), dtype=np.float64)
    starts[0] = values
    scenario_code = 0 if scenario == "clean" else 1
    scale = 0.5 * (upper - lower)
    for start_index in range(1, count):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, sample_id, scenario_code, start_index])
        )
        perturbation = _smooth(rng.normal(size=20)) * scale
        starts[start_index] = np.clip(values + perturbation, lower, upper)
    return starts


@dataclass
class DifferentiableSurrogate:
    """Double-precision surrogate and normalizers that retain autograd."""

    model: nn.Module
    input_mean: Tensor
    input_std: Tensor
    target_mean: Tensor
    target_std: Tensor
    device: torch.device

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.device.type == "mps":
            raise ValueError(
                "MPS cannot be used for inversion because float64 is required"
            )
        self.model.to(device=self.device, dtype=torch.float64)
        self.model.eval()
        self.input_mean = self._normalizer(self.input_mean, (20,), "input_mean")
        self.input_std = self._normalizer(self.input_std, (20,), "input_std")
        self.target_mean = self._normalizer(self.target_mean, (4, 1), "target_mean")
        self.target_std = self._normalizer(self.target_std, (4, 1), "target_std")
        if torch.any(self.input_std <= 0) or torch.any(self.target_std <= 0):
            raise ValueError("normalization standard deviations must be positive")

    def _normalizer(
        self, values: ArrayLike | Tensor, shape: tuple[int, ...], name: str
    ) -> Tensor:
        tensor = (
            torch.as_tensor(values)
            .detach()
            .clone()
            .to(device=self.device, dtype=torch.float64)
        )
        if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} has an invalid shape or non-finite values")
        return tensor

    @classmethod
    def load(
        cls, checkpoint: Path | str, device: str = "auto"
    ) -> DifferentiableSurrogate:
        """Load a split-compatible four-head checkpoint in double precision."""
        selected_device = resolve_inversion_device(device)
        payload = torch.load(
            Path(checkpoint),
            map_location=selected_device,
            weights_only=False,
        )
        validate_checkpoint_split_policy(payload)
        model = FourHeadForwardModel()
        model.load_state_dict(payload["model"])
        return cls(
            model=model,
            input_mean=np.array(payload["input_mean"], dtype=np.float64, copy=True),
            input_std=np.array(payload["input_std"], dtype=np.float64, copy=True),
            target_mean=np.array(payload["target_mean"], dtype=np.float64, copy=True),
            target_std=np.array(payload["target_std"], dtype=np.float64, copy=True),
            device=selected_device,
        )

    def predict_tensor(self, vs: Tensor) -> Tensor:
        """Predict physical phase velocities without disabling autograd."""
        if vs.ndim not in {1, 2}:
            raise ValueError("vs must have shape (20,) or (batch, 20)")
        single = vs.ndim == 1
        values = vs.unsqueeze(0) if single else vs
        if values.shape[1] != 20:
            raise ValueError("vs must have shape (20,) or (batch, 20)")
        values = values.to(device=self.device, dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("vs values must be finite")
        normalized = (values - self.input_mean) / self.input_std
        prediction = self.model(normalized)
        if prediction.ndim != 3 or prediction.shape[1] != 4:
            raise ValueError("surrogate output must have shape (batch, 4, frequency)")
        physical = prediction * self.target_std + self.target_mean
        return physical[0] if single else physical


@dataclass
class SurrogateObjective:
    """Masked multimode data misfit plus vertical smoothness regularization."""

    surrogate: DifferentiableSurrogate
    frequencies: ArrayLike
    observed: ArrayLike
    valid_mask: ArrayLike
    mode_weights: tuple[float, float, float, float]
    reference: ArrayLike
    regularization: ArrayLike
    regularization_lambda: float

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies, dtype=np.float64)
        observed, mask = _dispersion_arrays(self.observed, self.valid_mask)
        weights = np.asarray(self.mode_weights, dtype=np.float64)
        reference = np.asarray(self.reference, dtype=np.float64)
        regularization = np.asarray(self.regularization, dtype=np.float64)
        if frequencies.shape != (observed.shape[1],):
            raise ValueError("frequencies must match the observation frequency axis")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("frequencies must be finite and positive")
        if weights.shape != (4,) or not np.all(np.isfinite(weights)):
            raise ValueError("mode_weights must contain four finite values")
        if np.any(weights < 0):
            raise ValueError("mode_weights must be nonnegative")
        if reference.shape != (20,) or not np.all(np.isfinite(reference)):
            raise ValueError("reference must contain 20 finite values")
        if regularization.shape != (19, 20) or not np.all(np.isfinite(regularization)):
            raise ValueError("regularization must be a finite 19-by-20 matrix")
        if (
            not np.isfinite(self.regularization_lambda)
            or self.regularization_lambda < 0
        ):
            raise ValueError("regularization_lambda must be finite and nonnegative")
        used = [mode for mode in range(4) if weights[mode] > 0 and np.any(mask[mode])]
        if not used:
            raise ValueError("at least one used modal cell is required")

        self.frequencies = frequencies.copy()
        self.observed = observed.copy()
        self.valid_mask = mask.copy()
        self.mode_weights = tuple(float(value) for value in weights)
        self.reference = reference.copy()
        self.regularization = regularization.copy()
        self._used_modes = tuple(used)
        device = self.surrogate.device
        self._observed_tensor = torch.as_tensor(
            self.observed, device=device, dtype=torch.float64
        )
        self._mask_tensor = torch.as_tensor(
            self.valid_mask, device=device, dtype=torch.bool
        )
        self._reference_tensor = torch.as_tensor(
            self.reference, device=device, dtype=torch.float64
        )
        self._regularization_tensor = torch.as_tensor(
            self.regularization, device=device, dtype=torch.float64
        )

    def _vs_tensor(self, values: ArrayLike, *, requires_grad: bool) -> Tensor:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (20,) or not np.all(np.isfinite(array)):
            raise ValueError("vs must contain 20 finite values")
        return torch.tensor(
            array,
            device=self.surrogate.device,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    def _calculate(self, vs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        prediction = self.surrogate.predict_tensor(vs)
        expected_shape = (4, self.frequencies.size)
        if tuple(prediction.shape) != expected_shape:
            raise ValueError(
                "surrogate prediction shape does not match the observations"
            )
        if not bool(torch.isfinite(prediction).all()):
            raise ArithmeticError("surrogate prediction is non-finite")

        data_misfit = prediction.sum() * 0.0
        for mode in self._used_modes:
            mode_mask = self._mask_tensor[mode]
            residual = (
                prediction[mode, mode_mask] - self._observed_tensor[mode, mode_mask]
            )
            data_misfit = data_misfit + self.mode_weights[mode] * torch.mean(
                torch.square(residual)
            )
        difference = vs - self._reference_tensor
        regularized = self._regularization_tensor @ difference
        regularization = (
            self.regularization_lambda / 20.0 * torch.sum(torch.square(regularized))
        )
        total = data_misfit + regularization
        if not bool(torch.isfinite(total)):
            raise ArithmeticError("objective is non-finite")
        return total, data_misfit, regularization, prediction

    def value_and_grad(self, vs: ArrayLike) -> tuple[float, NDArray[np.float64]]:
        """Return one objective value and its same-pass autograd gradient."""
        vs_tensor = self._vs_tensor(vs, requires_grad=True)
        total, _, _, _ = self._calculate(vs_tensor)
        (gradient_tensor,) = torch.autograd.grad(total, vs_tensor)
        if not bool(torch.isfinite(gradient_tensor).all()):
            raise ArithmeticError("objective gradient is non-finite")
        gradient = np.asarray(
            gradient_tensor.detach().cpu().numpy(), dtype=np.float64
        ).copy()
        return float(total.detach().item()), gradient

    def terms(self, vs: ArrayLike) -> ObjectiveTerms:
        """Evaluate scalar objective components without constructing gradients."""
        vs_tensor = self._vs_tensor(vs, requires_grad=False)
        with torch.no_grad():
            total, data_misfit, regularization, _ = self._calculate(vs_tensor)
        return ObjectiveTerms(
            total=float(total.item()),
            data_misfit=float(data_misfit.item()),
            regularization=float(regularization.item()),
        )

    def predict(self, vs: ArrayLike) -> NDArray[np.float64]:
        """Return a physical four-mode float64 prediction."""
        vs_tensor = self._vs_tensor(vs, requires_grad=False)
        with torch.no_grad():
            prediction = self.surrogate.predict_tensor(vs_tensor)
        expected_shape = (4, self.frequencies.size)
        if tuple(prediction.shape) != expected_shape:
            raise ValueError(
                "surrogate prediction shape does not match the observations"
            )
        if not bool(torch.isfinite(prediction).all()):
            raise ArithmeticError("surrogate prediction is non-finite")
        return np.asarray(prediction.detach().cpu().numpy(), dtype=np.float64).copy()


def _model_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (20,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain 20 finite values")
    return array.copy()


def _optimizer_inputs(
    initial: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    max_iterations: int,
    relative_tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    initial_values = _model_vector(initial, "initial model")
    lower_values = _model_vector(lower, "lower bounds")
    upper_values = _model_vector(upper, "upper bounds")
    if np.any(lower_values > upper_values):
        raise ValueError("lower bounds must not exceed upper bounds")
    if np.any(initial_values < lower_values) or np.any(initial_values > upper_values):
        raise ValueError("initial model must lie within lower and upper bounds")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be a positive integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must be finite and in (0, 1)")
    return initial_values, lower_values, upper_values


def _finite_terms(terms: ObjectiveTerms, context: str) -> ObjectiveTerms:
    values = np.array(
        [terms.total, terms.data_misfit, terms.regularization], dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ArithmeticError(f"{context} objective terms are non-finite")
    return ObjectiveTerms(*(float(value) for value in values))


def _finite_prediction(
    objective: InversionObjective, values: ArrayLike
) -> NDArray[np.float64]:
    prediction = np.asarray(objective.predict(values), dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape[0] != 4:
        raise ValueError("objective prediction must have shape (4, frequency_count)")
    if not np.all(np.isfinite(prediction)):
        raise ArithmeticError("objective prediction is non-finite")
    return prediction.copy()


def invert_one(
    objective: InversionObjective,
    initial: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    max_iterations: int,
    relative_tolerance: float,
) -> InversionRun:
    """Run one strict bounded L-BFGS-B inversion with a validated callback."""
    initial_values, lower_values, upper_values = _optimizer_inputs(
        initial, lower, upper, max_iterations, relative_tolerance
    )
    initial_terms = _finite_terms(objective.terms(initial_values), "initial")

    def value_and_grad(
        values: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        value, gradient = objective.value_and_grad(values)
        value_array = np.asarray(value, dtype=np.float64)
        if value_array.shape != ():
            raise ValueError("objective value must be scalar")
        objective_value = float(value_array)
        if not np.isfinite(objective_value):
            raise ArithmeticError("objective value is non-finite")
        gradient_values = np.asarray(gradient, dtype=np.float64)
        if gradient_values.shape != (20,):
            raise ValueError("objective gradient must contain 20 values")
        if not np.all(np.isfinite(gradient_values)):
            raise ArithmeticError("objective gradient is non-finite")
        return objective_value, gradient_values

    result = scipy.optimize.minimize(
        value_and_grad,
        initial_values,
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(lower_values, upper_values, strict=True)),
        options={"maxiter": max_iterations, "ftol": relative_tolerance},
    )
    recovered = np.asarray(result.x, dtype=np.float64)
    if recovered.shape != (20,) or not np.all(np.isfinite(recovered)):
        raise ArithmeticError("L-BFGS-B returned an invalid bounded model")
    if np.any(recovered < lower_values) or np.any(recovered > upper_values):
        raise ArithmeticError("L-BFGS-B returned an invalid bounded model")
    terms = _finite_terms(objective.terms(recovered), "final")
    prediction = _finite_prediction(objective, recovered)
    return InversionRun(
        vs=recovered.copy(),
        predicted_phase_velocity=prediction,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        initial_objective=initial_terms.total,
        terms=terms,
    )


def iqr_inlier_mask(objectives: ArrayLike) -> NDArray[np.bool_]:
    """Return finite objective values within the inclusive 1.5-IQR fences."""
    values = np.asarray(objectives, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("objectives must be one-dimensional")
    finite = np.isfinite(values)
    keep = np.zeros(values.shape, dtype=np.bool_)
    if not np.any(finite):
        return keep
    q1, q3 = np.percentile(values[finite], [25.0, 75.0])
    iqr = q3 - q1
    if iqr == 0.0:
        keep[finite] = True
        return keep
    keep[finite] = (values[finite] >= q1 - 1.5 * iqr) & (
        values[finite] <= q3 + 1.5 * iqr
    )
    return keep


def _prediction_shape(
    objective: InversionObjective, runs: list[InversionRun]
) -> tuple[int, int]:
    for run in runs:
        if run.predicted_phase_velocity.ndim == 2:
            shape = run.predicted_phase_velocity.shape
            if shape[0] == 4 and shape[1] > 0:
                return shape
    frequencies = np.asarray(getattr(objective, "frequencies", ()))
    if frequencies.ndim == 1 and frequencies.size > 0:
        return (4, int(frequencies.size))
    return (4, 0)


def _complete_prediction_shape(
    objective: InversionObjective,
    runs: list[InversionRun],
    starts: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> tuple[int, int]:
    shape = _prediction_shape(objective, runs)
    if shape[1] > 0:
        return shape
    for initial in starts:
        if (
            not np.all(np.isfinite(initial))
            or np.any(initial < lower)
            or np.any(initial > upper)
        ):
            continue
        try:
            return _finite_prediction(objective, initial).shape
        except (ArithmeticError, RuntimeError, ValueError):
            continue
    return shape


def _failed_run(
    objective: InversionObjective,
    initial: NDArray[np.float64],
    error: ArithmeticError | RuntimeError | ValueError,
    prediction_shape: tuple[int, int],
) -> InversionRun:
    try:
        initial_objective = _finite_terms(objective.terms(initial), "initial").total
    except (ArithmeticError, RuntimeError, ValueError):
        initial_objective = float("nan")
    nan_terms = ObjectiveTerms(float("nan"), float("nan"), float("nan"))
    return InversionRun(
        vs=np.full(20, np.nan, dtype=np.float64),
        predicted_phase_velocity=np.full(prediction_shape, np.nan, dtype=np.float64),
        success=False,
        status=-1,
        message=f"{type(error).__name__}: {error}",
        iterations=0,
        evaluations=0,
        initial_objective=initial_objective,
        terms=nan_terms,
    )


def invert_ensemble(
    objective: InversionObjective,
    initial_models: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    max_iterations: int,
    relative_tolerance: float,
    minimum_valid_solutions: int,
) -> EnsembleResult:
    """Run ordered starts independently and summarize successful finite inliers."""
    starts = np.asarray(initial_models, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1] != 20 or starts.shape[0] == 0:
        raise ValueError("initial_models must have shape (start_count, 20)")
    lower_values = _model_vector(lower, "lower bounds")
    upper_values = _model_vector(upper, "upper bounds")
    if np.any(lower_values > upper_values):
        raise ValueError("lower bounds must not exceed upper bounds")
    if (
        isinstance(minimum_valid_solutions, bool)
        or not isinstance(minimum_valid_solutions, int)
        or not 1 <= minimum_valid_solutions <= starts.shape[0]
    ):
        raise ValueError(
            "minimum_valid_solutions must be between one and the start count"
        )

    runs: list[InversionRun] = []
    provisional_prediction_shape = _prediction_shape(objective, runs)
    for initial in starts:
        try:
            run = invert_one(
                objective,
                initial,
                lower_values,
                upper_values,
                max_iterations=max_iterations,
                relative_tolerance=relative_tolerance,
            )
        except (ArithmeticError, RuntimeError, ValueError) as error:
            run = _failed_run(objective, initial, error, provisional_prediction_shape)
        runs.append(run)
        provisional_prediction_shape = _prediction_shape(objective, runs)

    prediction_shape = _complete_prediction_shape(
        objective, runs, starts, lower_values, upper_values
    )
    runs = [
        replace(
            run,
            predicted_phase_velocity=np.full(
                prediction_shape, np.nan, dtype=np.float64
            ),
        )
        if run.status == -1 and run.predicted_phase_velocity.shape != prediction_shape
        else run
        for run in runs
    ]

    objectives = np.full(len(runs), np.nan, dtype=np.float64)
    successful = np.zeros(len(runs), dtype=np.bool_)
    for index, run in enumerate(runs):
        finite_solution = run.vs.shape == (20,) and np.all(np.isfinite(run.vs))
        finite_objective = np.isfinite(run.terms.total)
        if run.success and finite_solution and finite_objective:
            successful[index] = True
            objectives[index] = run.terms.total
    inlier_mask = successful & iqr_inlier_mask(objectives)

    nan_model = np.full(20, np.nan, dtype=np.float64)
    nan_terms = ObjectiveTerms(float("nan"), float("nan"), float("nan"))
    if np.count_nonzero(inlier_mask) < minimum_valid_solutions:
        return EnsembleResult(
            runs=tuple(runs),
            inlier_mask=inlier_mask,
            median_vs=nan_model.copy(),
            p10_vs=nan_model.copy(),
            p90_vs=nan_model.copy(),
            representative_terms=nan_terms,
            representative_prediction=np.full(
                prediction_shape, np.nan, dtype=np.float64
            ),
            sufficient=False,
        )

    solutions = np.stack([run.vs for run in runs], axis=0)[inlier_mask]
    p10, median, p90 = np.percentile(solutions, [10.0, 50.0, 90.0], axis=0)
    representative_terms = _finite_terms(objective.terms(median), "representative")
    representative_prediction = _finite_prediction(objective, median)
    return EnsembleResult(
        runs=tuple(runs),
        inlier_mask=inlier_mask,
        median_vs=np.asarray(median, dtype=np.float64),
        p10_vs=np.asarray(p10, dtype=np.float64),
        p90_vs=np.asarray(p90, dtype=np.float64),
        representative_terms=representative_terms,
        representative_prediction=representative_prediction,
        sufficient=True,
    )
