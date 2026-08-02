# Inversion Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leakage-free 80/5/5/10 dataset protocol and a resumable, paper-style L-BFGS-B surface-wave inversion experiment using the trained four-head forward surrogate.

**Architecture:** Centralize the deterministic split policy, keep optimization-only samples free of true `Vs`, and separate numerical inversion, result persistence, orchestration, and reporting into focused modules. Run a single-start inversion over the complete 10% holdout and a 100-start stratified deep experiment, with clean and 1% noise scenarios and physical Dunkin revalidation for deep results.

**Tech Stack:** Python 3.11+, NumPy, SciPy L-BFGS-B, PyTorch autograd, HDF5/HighFive-compatible files through h5py, Matplotlib, pytest, ruff.

## Global Constraints

- Preserve exactly 20 `Vs` values in `[0.3, 2.6]` km/s, 19 finite 0.1 km layers, and a half-space beginning at 1.9 km.
- Preserve four Rayleigh modes and the 0.5–60.0 Hz grid at 0.5 Hz spacing.
- Use `sample_id % 100`: train 0–79, validation 80–84, test 85–89, inversion 90–99.
- Record `split_policy = "mod100-v2-80-5-5-10"` in every new training checkpoint and reject older checkpoints for inversion.
- Never expose true `Vs` to reference-model construction, bound construction, objective evaluation, gradient evaluation, or optimization.
- Use strict SciPy `L-BFGS-B`, default mode weights `[4, 1, 1, 1]`, `lambda = 1e-2`, `vs_width = 0.7`, 100 maximum iterations, and relative tolerance `1e-5`.
- Generate `clean` and deterministic `noise_1pct` observations; never fill or interpolate invalid modal cells.
- Full experiment: all inversion rows, one reference start. Deep experiment: 100 stable rows per model family, 100 starts per row, minimum 20 valid solutions.
- Persist result shards atomically, bind them to dataset/checkpoint/config hashes, and support deterministic single-machine and cluster partitioning.
- Use the physical Python Dunkin solver with the quadratic strategy to validate deep median models.
- The local machine runs tests and tiny end-to-end fixtures only; formal production metrics require the complete external-machine run.

---

## File Structure

- Create `src/swave/splits.py`: the sole `sample_id`-to-split policy.
- Modify `src/swave/training.py`: consume the shared split policy and stamp checkpoints.
- Modify `src/swave/config.py`: add validated inversion configuration and loader.
- Create `configs/inversion.toml`: production inversion defaults.
- Create `src/swave/inversion_data.py`: optimization-safe HDF5 sample metadata and deterministic deep selection.
- Create `src/swave/inversion.py`: reference construction, noise, bounds, regularization, differentiable objective, L-BFGS-B, and ensemble statistics.
- Create `src/swave/inversion_results.py`: result schema, run identity, atomic HDF5/JSON writes, checksums, and resume validation.
- Create `src/swave/inversion_runner.py`: full/deep jobs, multiprocessing, cluster assignment, and physical deep validation.
- Create `src/swave/inversion_report.py`: truth join, metrics, JSON summary, and figures.
- Modify `src/swave/cli.py`: `invert` and `inversion-report` commands.
- Modify `scripts/plot_five_panel_comparison.py`: use the shared test split.
- Modify `README.md`: document the four splits and production workflow.
- Create `tests/conftest.py`: shared complete tiny HDF5 dataset/checkpoint fixtures for inversion tests.
- Create `tests/test_splits.py`, `tests/test_inversion_data.py`, `tests/test_inversion.py`, `tests/test_inversion_results.py`, `tests/test_inversion_runner.py`, and `tests/test_inversion_report.py`.
- Modify `tests/test_training.py`, `tests/test_cli.py`, and `tests/test_plot_five_panel_comparison.py`.

---

### Task 1: Centralize the Four-Way Split and Version Checkpoints

**Files:**
- Create: `src/swave/splits.py`
- Create: `tests/test_splits.py`
- Modify: `src/swave/training.py:24-57,194-219,382-400`
- Modify: `tests/test_training.py:20-120`
- Modify: `scripts/plot_five_panel_comparison.py:153-181`
- Modify: `tests/test_plot_five_panel_comparison.py:63-90`

**Interfaces:**
- Produces: `Split`, `SPLIT_POLICY`, `split_for_sample_id(sample_id)`, and `mask_for_split(sample_ids, split)`.
- Consumed by: training, plotting, inversion data access, checkpoint validation, and reporting.

- [ ] **Step 1: Write failing split-policy tests**

```python
# tests/test_splits.py
import numpy as np

from swave.splits import SPLIT_POLICY, mask_for_split, split_for_sample_id


def test_mod100_policy_has_exact_80_5_5_10_partition() -> None:
    ids = np.arange(100, dtype=np.uint64)
    counts = {
        split: int(mask_for_split(ids, split).sum())
        for split in ("train", "validation", "test", "inversion")
    }
    assert counts == {"train": 80, "validation": 5, "test": 5, "inversion": 10}
    assert SPLIT_POLICY == "mod100-v2-80-5-5-10"


def test_split_boundaries_are_stable() -> None:
    expected = {
        79: "train",
        80: "validation",
        84: "validation",
        85: "test",
        89: "test",
        90: "inversion",
        99: "inversion",
        190: "inversion",
    }
    assert {value: split_for_sample_id(value) for value in expected} == expected
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `.venv/bin/python -m pytest tests/test_splits.py -q`  
Expected: FAIL with `ModuleNotFoundError: No module named 'swave.splits'`.

- [ ] **Step 3: Implement the shared split module**

```python
# src/swave/splits.py
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Split = Literal["train", "validation", "test", "inversion"]
SPLIT_POLICY = "mod100-v2-80-5-5-10"


def split_for_sample_id(sample_id: int) -> Split:
    if sample_id < 0:
        raise ValueError("sample_id must be nonnegative")
    remainder = sample_id % 100
    if remainder < 80:
        return "train"
    if remainder < 85:
        return "validation"
    if remainder < 90:
        return "test"
    return "inversion"


def mask_for_split(sample_ids: ArrayLike, split: Split) -> NDArray[np.bool_]:
    if split not in {"train", "validation", "test", "inversion"}:
        raise ValueError("split must be train, validation, test, or inversion")
    values = np.asarray(sample_ids)
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("sample_ids must be a nonnegative one-dimensional array")
    remainder = values % 100
    bounds = {
        "train": (0, 80),
        "validation": (80, 85),
        "test": (85, 90),
        "inversion": (90, 100),
    }
    lower, upper = bounds[split]
    return np.asarray((remainder >= lower) & (remainder < upper), dtype=np.bool_)
```

- [ ] **Step 4: Update training and plotting to use the policy**

In `training.py`, import `Split`, `SPLIT_POLICY`, and `mask_for_split`; allow all four split names; replace the hard-coded remainder branches with `selected = mask_for_split(sample_ids, split)`; add `"split_policy": SPLIT_POLICY` to `_checkpoint_payload`; and make `evaluate()` reject a payload whose `split_policy` differs.

Update the tiny fixture IDs to `[0, 1, 2, 3, 80, 81, 85, 86, 90, 91]`, assert lengths `(4, 2, 2, 2)`, and assert the checkpoint contains `SPLIT_POLICY`. In the comparison script, select automatic rows with `mask_for_split(sample_ids, "test")` instead of `remainder >= 95`; change its test IDs to `[84, 85, 86]` and continue expecting automatic coupled test sample 86.

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_splits.py tests/test_training.py tests/test_plot_five_panel_comparison.py -q`  
Expected: all tests PASS.

- [ ] **Step 6: Commit the split protocol**

```bash
git add src/swave/splits.py src/swave/training.py scripts/plot_five_panel_comparison.py tests/test_splits.py tests/test_training.py tests/test_plot_five_panel_comparison.py
git commit -m "feat: reserve deterministic inversion split"
```

---

### Task 2: Add Strict Inversion Configuration

**Files:**
- Modify: `src/swave/config.py:152-210`
- Modify: `tests/test_config.py`
- Create: `configs/inversion.toml`

**Interfaces:**
- Produces: `NoiseScenario`, `InversionConfig`, `load_inversion_config(path)`, and `inversion_identity_hash(config)`.
- Consumed by: objective construction, result identity, the runner, CLI, and report generation.

- [ ] **Step 1: Write failing configuration tests**

```python
# append to tests/test_config.py
from swave.config import InversionConfig, load_inversion_config


def test_inversion_defaults_match_approved_experiment() -> None:
    config = InversionConfig()
    assert config.mode_weights == (4.0, 1.0, 1.0, 1.0)
    assert config.noise_scenarios == ("clean", "noise_1pct")
    assert config.initial_models == 100
    assert config.samples_per_kind == 100
    assert config.minimum_valid_solutions == 20


def test_inversion_config_rejects_invalid_cluster_and_unknown_key(tmp_path) -> None:
    with pytest.raises(ValueError, match="task_index"):
        InversionConfig(task_index=2, task_count=2)
    path = tmp_path / "bad.toml"
    path.write_text("[inversion]\nunknown = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown inversion keys"):
        load_inversion_config(path)
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `.venv/bin/python -m pytest tests/test_config.py -q`  
Expected: FAIL because `InversionConfig` does not exist.

- [ ] **Step 3: Implement the frozen configuration and strict loader**

Add a frozen dataclass with these fields and validation:

```python
NoiseScenario = Literal["clean", "noise_1pct"]


@dataclass(frozen=True)
class InversionConfig:
    dataset_config: Path = Path("configs/dataset.toml")
    dataset_dir: Path = Path("data/production")
    checkpoint: Path = Path("runs/production-48g/best.pt")
    output_dir: Path = Path("results/inversion")
    mode_weights: tuple[float, float, float, float] = (4.0, 1.0, 1.0, 1.0)
    regularization_lambda: float = 1e-2
    regularization_type: str = "adaptive"
    vs_min: float = 0.3
    vs_max: float = 2.6
    vs_width: float = 0.7
    max_iterations: int = 100
    relative_tolerance: float = 1e-5
    initial_models: int = 100
    minimum_valid_solutions: int = 20
    samples_per_kind: int = 100
    noise_scenarios: tuple[NoiseScenario, ...] = ("clean", "noise_1pct")
    seed: int = 20_260_727
    device: str = "auto"
    workers: int = 0
    task_index: int = 0
    task_count: int = 1

    def __post_init__(self) -> None:
        for name in ("dataset_config", "dataset_dir", "checkpoint", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if len(self.mode_weights) != 4 or any(value < 0 for value in self.mode_weights):
            raise ValueError("mode_weights must contain four nonnegative values")
        if not any(self.mode_weights):
            raise ValueError("at least one mode weight must be positive")
        if self.regularization_lambda < 0:
            raise ValueError("regularization_lambda must be nonnegative")
        if self.regularization_type not in {"adaptive", "first_order"}:
            raise ValueError("regularization_type must be adaptive or first_order")
        if not 0 < self.vs_min < self.vs_max or not 0 < self.vs_width <= self.vs_max - self.vs_min:
            raise ValueError("Vs bounds and vs_width are invalid")
        if self.max_iterations <= 0 or not 0 < self.relative_tolerance < 1:
            raise ValueError("optimizer limits are invalid")
        if self.initial_models <= 0 or not 1 <= self.minimum_valid_solutions <= self.initial_models:
            raise ValueError("ensemble solution counts are invalid")
        if self.samples_per_kind <= 0:
            raise ValueError("samples_per_kind must be positive")
        if not self.noise_scenarios or any(value not in {"clean", "noise_1pct"} for value in self.noise_scenarios):
            raise ValueError("noise_scenarios are invalid")
        if self.device not in {"auto", "cpu", "cuda", "mps"} or self.workers < 0:
            raise ValueError("device or workers is invalid")
        if self.task_count <= 0 or not 0 <= self.task_index < self.task_count:
            raise ValueError("task_index must be in [0, task_count)")

    def to_dict(self) -> dict[str, Any]:
        return _plain(dataclasses.asdict(self))
```

`from_mapping()` must compare TOML keys against dataclass field names before construction and convert list values for `mode_weights` and `noise_scenarios` to tuples. Extend `canonical_hash()` to accept `InversionConfig`. `load_inversion_config()` reads `[inversion]` and rejects unknown keys. `inversion_identity_hash()` hashes the scientific/path fields after removing `device`, `workers`, `task_index`, and `task_count`; those execution controls may differ across cluster tasks without creating a different experiment identity.

- [ ] **Step 4: Add production `configs/inversion.toml`**

```toml
[inversion]
dataset_config = "configs/dataset.toml"
dataset_dir = "data/production"
checkpoint = "runs/production-48g/best.pt"
output_dir = "results/inversion"
mode_weights = [4.0, 1.0, 1.0, 1.0]
regularization_lambda = 0.01
regularization_type = "adaptive"
vs_min = 0.3
vs_max = 2.6
vs_width = 0.7
max_iterations = 100
relative_tolerance = 0.00001
initial_models = 100
minimum_valid_solutions = 20
samples_per_kind = 100
noise_scenarios = ["clean", "noise_1pct"]
seed = 20260727
device = "auto"
workers = 0
task_index = 0
task_count = 1
```

- [ ] **Step 5: Run configuration tests and lint**

Run: `.venv/bin/python -m pytest tests/test_config.py -q && .venv/bin/ruff check src/swave/config.py tests/test_config.py`  
Expected: PASS with no lint output.

- [ ] **Step 6: Commit configuration support**

```bash
git add src/swave/config.py tests/test_config.py configs/inversion.toml
git commit -m "feat: configure paper-style inversion runs"
```

---

### Task 3: Add Optimization-Safe Inversion Data Access

**Files:**
- Create: `src/swave/inversion_data.py`
- Create: `tests/conftest.py`
- Create: `tests/test_inversion_data.py`

**Interfaces:**
- Consumes: `mask_for_split()`, `validate_dataset_files()`, and `ModelKind`.
- Produces: `InversionSample`, `iter_inversion_samples(dataset_dir)`, `samples_by_source_shard(dataset_dir)`, and `select_deep_samples(dataset_dir, per_kind)`.

- [ ] **Step 1: Write a fixture and failing data-access tests**

```python
# tests/test_inversion_data.py
def test_optimizer_samples_exclude_true_vs(tiny_complete_dataset) -> None:
    rows = list(iter_inversion_samples(tiny_complete_dataset))
    assert [row.sample_id for row in rows] == [90, 91, 92, 93]
    assert all(not hasattr(row, "vs") and not hasattr(row, "true_vs") for row in rows)
    assert all(row.phase_velocity.shape == (4, 120) for row in rows)


def test_deep_selection_is_smallest_id_per_family(tiny_complete_dataset) -> None:
    selected = select_deep_samples(tiny_complete_dataset, per_kind=1)
    assert [(row.model_kind, row.sample_id) for row in selected] == [
        (0, 90),
        (1, 91),
        (2, 92),
        (3, 93),
    ]
```

The shared `tests/conftest.py` `tiny_complete_dataset` fixture must write all datasets required by `validate_dataset_files()`, use inversion IDs 90–93 with model kinds 0–3, and recompute the shard checksum after writing. It must also expose `tiny_checkpoint`, whose payload contains a valid `FourHeadForwardModel`, normalization arrays, the fixture dataset hash, and `SPLIT_POLICY`.

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `.venv/bin/python -m pytest tests/test_inversion_data.py -q`  
Expected: FAIL with `ModuleNotFoundError: No module named 'swave.inversion_data'`.

- [ ] **Step 3: Implement the safe sample interface**

```python
@dataclass(frozen=True)
class InversionSample:
    sample_id: int
    model_kind: int
    phase_velocity: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    source_path: Path
    source_shard_id: int
    source_row: int


def iter_inversion_samples(dataset_dir: Path | str) -> Iterator[InversionSample]:
    directory = Path(dataset_dir)
    validate_dataset_files(directory)
    for path in sorted(directory.glob("shard-*.h5")):
        with h5py.File(path, "r") as handle:
            sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
            rows = np.flatnonzero(mask_for_split(sample_ids, "inversion"))
            shard_id = int(handle.attrs["shard_id"])
            for row in rows:
                yield InversionSample(
                    sample_id=int(sample_ids[row]),
                    model_kind=int(handle["model_kind"][row]),
                    phase_velocity=np.asarray(handle["phase_velocity"][row], dtype=np.float32),
                    valid_mask=np.asarray(handle["valid_mask"][row], dtype=np.bool_),
                    source_path=path,
                    source_shard_id=shard_id,
                    source_row=int(row),
                )
```

`samples_by_source_shard()` returns a dictionary keyed by source shard ID with rows sorted by `sample_id`. `select_deep_samples()` scans only inversion rows, retains the first `per_kind` rows for each `ModelKind`, returns them grouped in enum order and sorted within each family, and raises `ValueError` listing each deficient family.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_inversion_data.py tests/test_dataset.py -q`  
Expected: all tests PASS.

- [ ] **Step 5: Commit safe inversion data access**

```bash
git add src/swave/inversion_data.py tests/conftest.py tests/test_inversion_data.py
git commit -m "feat: expose isolated inversion samples"
```

---

### Task 4: Implement Reference Models, Noise, Bounds, and Differentiable Objective

**Files:**
- Create: `src/swave/inversion.py`
- Create: `tests/test_inversion.py`

**Interfaces:**
- Consumes: `InversionConfig`, `NoiseScenario`, `FourHeadForwardModel`, checkpoint normalization arrays, frequencies, observed phase velocity, and valid mask.
- Produces: `ReferenceModel`, `ObjectiveTerms`, `DifferentiableSurrogate`, `SurrogateObjective`, `build_reference_model()`, `apply_observation_noise()`, `regularization_matrix()`, and `generate_initial_models()`.

- [ ] **Step 1: Write failing reference, noise, and mask tests**

```python
def test_reference_uses_only_fundamental_observation() -> None:
    frequencies = np.arange(0.5, 60.0 + 0.25, 0.5)
    observed = np.full((4, 120), np.nan)
    mask = np.zeros((4, 120), dtype=bool)
    observed[0] = 0.8 + 0.1 / frequencies
    mask[0] = True
    reference = build_reference_model(
        frequencies, observed, mask, vs_min=0.3, vs_max=2.6, vs_width=0.7
    )
    assert reference.vs.shape == (20,)
    assert np.all(reference.lower <= reference.vs)
    assert np.all(reference.vs <= reference.upper)


def test_one_percent_noise_is_reproducible_and_preserves_mask() -> None:
    observed = np.ones((4, 120), dtype=np.float64)
    mask = np.ones_like(observed, dtype=bool)
    mask[3, :4] = False
    first = apply_observation_noise(observed, mask, "noise_1pct", 7, 90)
    second = apply_observation_noise(observed, mask, "noise_1pct", 7, 90)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isnan(first[~mask]))
    assert np.all(first[mask] != observed[mask])
```

- [ ] **Step 2: Write a failing finite-difference gradient test**

Create a deterministic `ToyForward` module mapping 20 inputs to `(batch, 4, 3)` through a fixed linear layer. Construct `SurrogateObjective` with identity input normalization, zero target mean, unit target standard deviation, a three-frequency observation, and an all-true mask. Compare each autograd component with central differences using `h = 1e-6`, `rtol = 1e-5`, and `atol = 1e-7`.

- [ ] **Step 3: Run the tests and verify the missing module failure**

Run: `.venv/bin/python -m pytest tests/test_inversion.py -q`  
Expected: FAIL because the inversion functions do not exist.

- [ ] **Step 4: Implement deterministic preprocessing**

Implement:

```python
@dataclass(frozen=True)
class ReferenceModel:
    vs: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


@dataclass(frozen=True)
class ObjectiveTerms:
    total: float
    data_misfit: float
    regularization: float


def _smooth(values: NDArray[np.float64]) -> NDArray[np.float64]:
    current = values.copy()
    kernel = np.array([0.25, 0.5, 0.25])
    for _ in range(2):
        current = np.convolve(np.pad(current, 1, mode="reflect"), kernel, mode="valid")
    return current
```

`build_reference_model()` must require at least two finite, valid fundamental cells; compute `depth = (c/f)/3` and `Vs = 1.1*c`; sort ascending depths; average duplicate depths; interpolate to `np.arange(20)*0.1` with constant endpoint extrapolation; smooth twice; clip globally; and build local bounds with half `vs_width`.

`apply_observation_noise()` must first create an all-NaN output, copy only valid cells for `clean`, and for `noise_1pct` multiply valid cells by `1 + 0.01*rng.normal(size=int(mask.sum()))`, where `rng = np.random.default_rng(np.random.SeedSequence([seed, sample_id, 1]))`.

`regularization_matrix()` returns the 19-by-20 `[q, -q]` matrix from the approved adaptive formula, with all `q=1` for ordinary first order or a constant reference.

`generate_initial_models()` returns shape `(count, 20)`, uses the reference as row zero, seeds each additional row with `SeedSequence([seed, sample_id, scenario_code, start_index])`, smooths the Gaussian vector twice, scales it by `0.5*(upper-lower)`, adds it to the reference, and clips to bounds.

- [ ] **Step 5: Implement the differentiable surrogate objective**

`DifferentiableSurrogate.load(checkpoint, device)` must load `FourHeadForwardModel`, require `payload["split_policy"] == SPLIT_POLICY`, copy all normalization arrays, move the model and normalizers to the selected device in `torch.float64`, and expose `predict_tensor(vs)` without inference mode.

`SurrogateObjective.value_and_grad(vs)` must:

1. create a double tensor with `requires_grad=True`;
2. normalize `Vs`, call the model, and restore physical phase velocities;
3. compute one mean squared residual for every nonempty positive-weight mode over mask-true cells;
4. add `lambda/20 * ||L(vs-reference)||²`;
5. call `torch.autograd.grad(total, vs_tensor)`;
6. return finite Python `float` and `float64` NumPy gradient.

`terms(vs)` repeats the same calculation without gradient and returns `ObjectiveTerms`; `predict(vs)` returns a physical `(4, frequency_count)` float64 array. Shapes, finite values, and at least one used modal cell must be validated before evaluation.

- [ ] **Step 6: Run objective tests and lint**

Run: `.venv/bin/python -m pytest tests/test_inversion.py -q && .venv/bin/ruff check src/swave/inversion.py tests/test_inversion.py`  
Expected: all tests PASS and no lint output.

- [ ] **Step 7: Commit the numerical inversion core**

```bash
git add src/swave/inversion.py tests/test_inversion.py
git commit -m "feat: add differentiable inversion objective"
```

---

### Task 5: Add L-BFGS-B Runs and Multi-Start Statistics

**Files:**
- Modify: `src/swave/inversion.py`
- Modify: `tests/test_inversion.py`

**Interfaces:**
- Consumes: `SurrogateObjective`, `ReferenceModel`, `InversionConfig`, and deterministic initial models.
- Produces: `InversionRun`, `EnsembleResult`, `invert_one()`, `invert_ensemble()`, and `iqr_inlier_mask()`.

- [ ] **Step 1: Write failing bounded optimizer tests**

```python
def test_lbfgsb_recovers_bounded_quadratic_solution() -> None:
    objective = QuadraticObjective(target=np.linspace(0.5, 2.4, 20))
    reference = np.full(20, 1.0)
    lower = np.full(20, 0.3)
    upper = np.full(20, 2.0)
    result = invert_one(objective, reference, lower, upper, max_iterations=100, relative_tolerance=1e-9)
    np.testing.assert_allclose(result.vs, np.minimum(objective.target, 2.0), atol=1e-6)
    assert result.success
    assert np.all((result.vs >= lower) & (result.vs <= upper))


def test_ensemble_rejects_objective_outlier_and_reports_percentiles() -> None:
    solutions = np.vstack([np.full((9, 20), 1.0), np.full((1, 20), 2.5)])
    objectives = np.array([1.0] * 9 + [100.0])
    keep = iqr_inlier_mask(objectives)
    assert keep.tolist() == [True] * 9 + [False]
```

`QuadraticObjective.value_and_grad(x)` returns `sum((x-target)**2)` and `2*(x-target)`; `terms(x)` returns the same total as data misfit with zero regularization; `predict(x)` returns a deterministic `(4, 3)` array.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_inversion.py -q`  
Expected: FAIL because `invert_one` and ensemble types do not exist.

- [ ] **Step 3: Implement result dataclasses and strict L-BFGS-B**

```python
@dataclass(frozen=True)
class InversionRun:
    vs: NDArray[np.float64]
    predicted_phase_velocity: NDArray[np.float64]
    success: bool
    status: int
    message: str
    iterations: int
    evaluations: int
    initial_objective: float
    terms: ObjectiveTerms


def invert_one(objective, initial, lower, upper, *, max_iterations, relative_tolerance) -> InversionRun:
    initial_value = objective.terms(initial).total
    result = scipy.optimize.minimize(
        objective.value_and_grad,
        np.asarray(initial, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(lower, upper, strict=True)),
        options={"maxiter": max_iterations, "ftol": relative_tolerance},
    )
    recovered = np.asarray(result.x, dtype=np.float64)
    if np.any(recovered < lower) or np.any(recovered > upper) or not np.all(np.isfinite(recovered)):
        raise ArithmeticError("L-BFGS-B returned an invalid bounded model")
    terms = objective.terms(recovered)
    return InversionRun(
        vs=recovered,
        predicted_phase_velocity=objective.predict(recovered),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        initial_objective=float(initial_value),
        terms=terms,
    )
```

Catch arithmetic/runtime/value failures at the per-start wrapper boundary and store a failed run with NaN arrays and a stable failure code rather than terminating the sample.

- [ ] **Step 4: Implement ensemble aggregation**

`iqr_inlier_mask()` operates only on finite objectives and always applies the inclusive `[Q1-1.5*IQR, Q3+1.5*IQR]` fences, including when IQR is zero. Thus `[1, ..., 1, 100]` rejects 100 rather than invoking a keep-all exception. `invert_ensemble()` runs every initial row, collects solution/objective/status diagnostics, filters successful finite solutions and IQR outliers, enforces `minimum_valid_solutions`, and returns:

```python
@dataclass(frozen=True)
class EnsembleResult:
    runs: tuple[InversionRun, ...]
    inlier_mask: NDArray[np.bool_]
    median_vs: NDArray[np.float64]
    p10_vs: NDArray[np.float64]
    p90_vs: NDArray[np.float64]
    representative_terms: ObjectiveTerms
    representative_prediction: NDArray[np.float64]
    sufficient: bool
```

For insufficient ensembles, percentile arrays are NaN but all run diagnostics remain available. For sufficient ensembles, evaluate the objective and prediction at the componentwise median model.

- [ ] **Step 5: Run inversion tests**

Run: `.venv/bin/python -m pytest tests/test_inversion.py -q`  
Expected: all tests PASS.

- [ ] **Step 6: Commit optimization and ensemble support**

```bash
git add src/swave/inversion.py tests/test_inversion.py
git commit -m "feat: optimize bounded multi-start inversions"
```

---

### Task 6: Add Atomic Result Shards and Run Identity

**Files:**
- Create: `src/swave/inversion_results.py`
- Create: `tests/test_inversion_results.py`

**Interfaces:**
- Consumes: dataset manifest hash, checkpoint path, `SPLIT_POLICY`, inversion config hash, experiment name, and batches produced by the runner.
- Produces: `ResultManifest`, `ResultBatch`, `checkpoint_sha256()`, `initialize_result_manifest()`, `write_result_shard()`, `validate_result_shard()`, `mark_job_complete()`, and `validate_complete_results()`.

- [ ] **Step 1: Write failing identity and corruption tests**

Define `result_manifest` by calling `initialize_result_manifest()` with one expected job, and define `result_batch` with two ordered inversion IDs, bounded 20-layer profiles, `(2,4,120)` observed/surrogate arrays, all-valid masks, and no deep optional fields.

```python
def test_result_identity_binds_checkpoint_and_configuration(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = initialize_result_manifest(
        tmp_path / "results",
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        inversion_config_hash="b" * 64,
        experiment="full",
        expected_jobs=("full-clean-shard-00000",),
    )
    assert manifest.checkpoint_sha256 == hashlib.sha256(b"checkpoint").hexdigest()
    assert manifest.split_policy == SPLIT_POLICY


def test_corrupt_completed_result_is_rejected(tmp_path, result_batch, result_manifest) -> None:
    path = write_result_shard(
        tmp_path, "full-clean-shard-00000", result_batch, result_manifest
    )
    with h5py.File(path, "r+") as handle:
        handle["sample_id"][0] = 999
    with pytest.raises(ValueError, match="checksum|sample_id"):
        validate_result_shard(path, expected_sample_ids=result_batch.sample_id)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `.venv/bin/python -m pytest tests/test_inversion_results.py -q`  
Expected: FAIL with `ModuleNotFoundError: No module named 'swave.inversion_results'`.

- [ ] **Step 3: Implement manifest and batch schemas**

Implement these exact public schemas:

```python
@dataclass(frozen=True)
class ResultManifest:
    schema_version: int
    dataset_config_hash: str
    checkpoint_sha256: str
    split_policy: str
    inversion_config_hash: str
    experiment: str
    expected_jobs: tuple[str, ...]
    completed_jobs: tuple[str, ...]
    job_sha256: dict[str, str]
    package_version: str
    created_at: str
    complete: bool


@dataclass(frozen=True)
class ResultBatch:
    sample_id: NDArray[np.uint64]
    model_kind: NDArray[np.uint8]
    success: NDArray[np.bool_]
    status: NDArray[np.int32]
    iterations: NDArray[np.int32]
    evaluations: NDArray[np.int32]
    initial_objective: NDArray[np.float64]
    final_objective: NDArray[np.float64]
    data_misfit: NDArray[np.float64]
    regularization: NDArray[np.float64]
    reference_vs: NDArray[np.float32]
    inverted_vs: NDArray[np.float32]
    observed_phase_velocity: NDArray[np.float32]
    surrogate_phase_velocity: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    failure_code: NDArray[np.bytes_]
    ensemble_vs: NDArray[np.float32] | None = None
    ensemble_success: NDArray[np.bool_] | None = None
    ensemble_status: NDArray[np.int32] | None = None
    ensemble_objective: NDArray[np.float64] | None = None
    ensemble_inlier_mask: NDArray[np.bool_] | None = None
    median_vs: NDArray[np.float32] | None = None
    p10_vs: NDArray[np.float32] | None = None
    p90_vs: NDArray[np.float32] | None = None
    physical_phase_velocity: NDArray[np.float32] | None = None
    physical_valid_mask: NDArray[np.bool_] | None = None
```

Full fields have fixed leading-row shapes; deep optional fields must either all be present with consistent row/start dimensions or all be absent.

Validate leading row counts, `Vs` width 20, curve shape `(N, 4, 120)`, unique ordered sample IDs, and finite bounded values where success is true. A caught exception stores NaN numerical outputs; a finite but nonconverged optimizer result may retain its bounded final model and diagnostics with `success=False` and a nonempty failure code.

- [ ] **Step 4: Implement atomic writes, checksums, and locked manifest updates**

`write_result_shard()` writes `<job>.h5.tmp-<pid>`, stores job name and result identity attributes, flushes and closes, validates the temporary file, computes SHA-256, then `replace()` publishes `<job>.h5`.

`mark_job_complete()` acquires an exclusive `fcntl.flock()` on `manifest.lock`, reloads and revalidates `manifest.json`, adds exactly one job/checksum, writes JSON through `manifest.json.tmp-<pid>`, and atomically replaces the manifest. Duplicate jobs are accepted only when their checksum matches. The manifest uses `inversion_identity_hash(config)`, so different worker/device/task-assignment controls share one scientific run. `validate_complete_results()` requires every expected job, no unexpected HDF5 jobs, matching checksums, and `complete=True`.

- [ ] **Step 5: Test resume and identity conflicts**

Add tests that a second identical initialization returns the existing manifest, a changed checkpoint/config hash raises `ValueError`, a matching duplicate job is idempotent, and a missing job makes complete validation fail.

- [ ] **Step 6: Run result tests and lint**

Run: `.venv/bin/python -m pytest tests/test_inversion_results.py -q && .venv/bin/ruff check src/swave/inversion_results.py tests/test_inversion_results.py`  
Expected: all tests PASS and no lint output.

- [ ] **Step 7: Commit resumable result storage**

```bash
git add src/swave/inversion_results.py tests/test_inversion_results.py
git commit -m "feat: persist resumable inversion shards"
```

---

### Task 7: Orchestrate Full and Deep Experiments

**Files:**
- Create: `src/swave/inversion_runner.py`
- Create: `tests/test_inversion_runner.py`

**Interfaces:**
- Consumes: `InversionConfig`, dataset config/manifest, safe inversion samples, differentiable objective, optimizer/ensemble functions, result persistence, `LayeredModel`, and `DispersionSolver`.
- Produces: `Experiment = Literal["full", "deep", "both"]`, `InversionJob`, `build_jobs()`, `assigned_jobs()`, `run_inversion_job()`, and `run_inversion_experiment()`.

- [ ] **Step 1: Write failing deterministic job-assignment tests**

```python
def test_cluster_assignment_is_disjoint_and_complete() -> None:
    jobs = tuple(InversionJob(name=f"job-{index}", experiment="full", noise="clean", samples=()) for index in range(17))
    assigned = [assigned_jobs(jobs, task_index=index, task_count=4) for index in range(4)]
    names = [{job.name for job in group} for group in assigned]
    assert not any(names[left] & names[right] for left in range(4) for right in range(left + 1, 4))
    assert set().union(*names) == {job.name for job in jobs}


def test_job_names_are_stable_by_experiment_noise_and_source_shard(tiny_complete_dataset) -> None:
    jobs = build_jobs(tiny_complete_dataset, "full", ("clean", "noise_1pct"), samples_per_kind=1)
    assert [job.name for job in jobs] == [
        "full-clean-shard-00000",
        "full-noise_1pct-shard-00000",
    ]
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `.venv/bin/python -m pytest tests/test_inversion_runner.py -q`  
Expected: FAIL with `ModuleNotFoundError: No module named 'swave.inversion_runner'`.

- [ ] **Step 3: Implement validation and stable jobs**

Before job construction, load `dataset_config`, require `canonical_hash(dataset_config) == validate_dataset_files(dataset_dir).config_hash`, load the checkpoint on CPU, and require matching dataset hash and `SPLIT_POLICY`. Require `workers == 1` when the resolved device is CUDA or MPS.

```python
Experiment = Literal["full", "deep", "both"]


@dataclass(frozen=True)
class InversionJob:
    name: str
    experiment: Literal["full", "deep"]
    noise: NoiseScenario
    samples: tuple[InversionSample, ...]
```

`build_jobs()` groups full rows by source shard. It sorts the selected deep population by sample ID and partitions it into stable bounded chunks (production default: 10 rows), with first/last sample IDs and the ordered-ID digest in each deep job name. `assigned_jobs()` selects jobs whose stable index modulo `task_count` equals `task_index`.

- [ ] **Step 4: Implement one inversion job**

For each worker/job, load one `DifferentiableSurrogate`. For each sample:

1. apply the job noise scenario;
2. build the reference and bounds from that observation;
3. create `SurrogateObjective` using the sample mask;
4. for `full`, call `invert_one()` with the reference;
5. for `deep`, generate configured starts and call `invert_ensemble()`;
6. store stable failure codes for insufficient fundamental data, nonfinite objective, optimizer failure, or insufficient valid solutions;
7. for a sufficient deep ensemble, build `model = LayeredModel.from_vs(median_vs, dataset_config.geology.empirical_method, dataset_config.geology.thickness_km)` and call `DispersionSolver(model, dataset_config.physics).solve_grid(frequencies=dataset_config.physics.frequencies, strategy="quadratic")`;
8. assemble a `ResultBatch`, write atomically, and mark the job complete.

The worker must never open the source `vs` dataset.

- [ ] **Step 5: Implement single-machine execution and recovery**

`run_inversion_experiment()` initializes the exact expected-job manifest, validates and skips completed shards, filters cluster-assigned jobs, and runs pending jobs sequentially for one worker or through `ProcessPoolExecutor` for CPU workers. Each process receives immutable config and job data and loads its own checkpoint once per submitted job. Return the refreshed manifest; it is complete only when all cluster tasks have collectively published all expected jobs.

- [ ] **Step 6: Add a tiny end-to-end runner test**

Create a tiny valid dataset, a checkpoint stamped with its config hash and `SPLIT_POLICY`, and monkeypatch `invert_one` plus physical `solve_grid` with deterministic bounded results. Run `full` and `deep`, verify clean/noise job files, verify sample IDs are inversion-only, rerun to confirm no file modification times change, and change a config field to confirm resume rejection.

- [ ] **Step 7: Run runner, result, and inversion tests**

Run: `.venv/bin/python -m pytest tests/test_inversion_runner.py tests/test_inversion_results.py tests/test_inversion.py -q`  
Expected: all tests PASS.

- [ ] **Step 8: Commit experiment orchestration**

```bash
git add src/swave/inversion_runner.py tests/test_inversion_runner.py
git commit -m "feat: run full and deep inversion experiments"
```

---

### Task 8: Build Metrics, Reports, and Scientific Figures

**Files:**
- Create: `src/swave/inversion_report.py`
- Create: `tests/test_inversion_report.py`

**Interfaces:**
- Consumes: complete result manifest/shards and the original dataset directory.
- Produces: `compute_vs_metrics()`, `compute_inversion_metrics()`, `build_inversion_report()`, `summary.json`, and PNG figures.

- [ ] **Step 1: Write failing metric and truth-isolation tests**

```python
def test_metrics_are_grouped_by_noise_and_model_kind() -> None:
    truth = np.array([[1.0] * 20, [2.0] * 20])
    recovered = np.array([[1.1] * 20, [1.8] * 20])
    metrics = compute_vs_metrics(truth, recovered)
    assert metrics["mae_km_s"] == pytest.approx(0.15)
    assert metrics["rmse_km_s"] == pytest.approx(np.sqrt(0.025))
    assert metrics["mean_relative_percent"] == pytest.approx(10.0)


def test_report_reads_truth_only_after_complete_result_validation(monkeypatch, report_fixture) -> None:
    events = []
    monkeypatch.setattr(module, "validate_complete_results", lambda path: events.append("validated") or report_fixture.manifest)
    monkeypatch.setattr(module, "_load_true_vs", lambda path, ids: events.append("truth") or report_fixture.truth)
    build_inversion_report(report_fixture.results, report_fixture.dataset, report_fixture.output)
    assert events[:2] == ["validated", "truth"]
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `.venv/bin/python -m pytest tests/test_inversion_report.py -q`  
Expected: FAIL with `ModuleNotFoundError: No module named 'swave.inversion_report'`.

- [ ] **Step 3: Implement truth join and scalar metrics**

`_load_true_vs()` scans source shards only after complete result validation, builds a requested-ID dictionary, rejects duplicates/missing IDs, and returns `(N,20)` truth in result order. Implement `compute_vs_metrics()` with MAE, RMSE, mean relative percent, P95 absolute error, per-layer MAE/bias, and row count. Implement masked per-mode frequency metrics for surrogate and physical curves, including physical missing fraction.

Group outputs under `overall`, `noise.<scenario>`, `model_kind.<name>`, and `noise_model_kind.<scenario>.<name>`. Compute convergence/failure distributions and clean-to-noisy differences. Deep results add P10/P90 truth coverage and mean interval width.

- [ ] **Step 4: Implement deterministic figures**

Use Matplotlib `Agg` and fixed file names:

- `vs-error-by-depth.png`: layer depth versus MAE and bias, split by noise;
- `vs-error-by-kind-and-noise.png`: four-family clean/noisy boxplots;
- `optimization-diagnostics.png`: initial/final objective, convergence, and iterations;
- `representative-<kind>-<noise>.png`: true/reference/median/P10/P90 profile plus four observed/surrogate/physical modal panels for the smallest successful deep sample in each group.

Every plotting helper must reject empty groups instead of drawing zeros. Use km/s and km internally; include m/s only in plot labels where explicitly converted.

- [ ] **Step 5: Write summary atomically and test artifacts**

`build_inversion_report()` validates results, loads all result rows, joins truth, computes the nested dictionary, writes `summary.json.tmp-<pid>` then atomically replaces `summary.json`, writes every required PNG, and returns the summary dictionary. Tests assert exact metric values, expected group keys, nonempty PNG files, and an error for incomplete manifests or missing truth rows.

- [ ] **Step 6: Run report tests and lint**

Run: `.venv/bin/python -m pytest tests/test_inversion_report.py -q && .venv/bin/ruff check src/swave/inversion_report.py tests/test_inversion_report.py`  
Expected: all tests PASS and no lint output.

- [ ] **Step 7: Commit reporting**

```bash
git add src/swave/inversion_report.py tests/test_inversion_report.py
git commit -m "feat: report inversion accuracy and uncertainty"
```

---

### Task 9: Wire CLI, Documentation, and Final Verification

**Files:**
- Modify: `src/swave/cli.py:14-25,158-250`
- Modify: `tests/test_cli.py`
- Modify: `README.md:82-170,219-238`

**Interfaces:**
- Consumes: `load_inversion_config()`, `run_inversion_experiment()`, and `build_inversion_report()`.
- Produces: public `swave invert` and `swave inversion-report` workflows plus external-machine run instructions.

- [ ] **Step 1: Write failing CLI help tests**

```python
def test_help_lists_inversion_workflows(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "invert" in output
    assert "inversion-report" in output


def test_invert_help_exposes_cluster_overrides(capsys) -> None:
    assert main(["invert", "--help"]) == 0
    output = capsys.readouterr().out
    for option in ("--experiment", "--workers", "--task-index", "--task-count"):
        assert option in output
```

- [ ] **Step 2: Run the tests and verify CLI failure**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`  
Expected: FAIL because the new subcommands are absent.

- [ ] **Step 3: Add lazy CLI handlers and validated overrides**

`_invert()` loads `InversionConfig`, applies non-None path/device/worker/task overrides through `dataclasses.replace()`, imports `run_inversion_experiment` inside the handler, runs the selected `full|deep|both` experiment, and prints the manifest as JSON.

`_inversion_report()` imports `build_inversion_report` inside the handler, passes result/dataset/output directories, and prints returned summary JSON. Add parser options matching the design and ensure all errors still flow through the existing clean `error:` prefix path.

- [ ] **Step 4: Document production and cluster commands**

Update README with the 80/5/5/10 table, explain checkpoint incompatibility with the previous split, document the objective and two noise scenarios, and include exact commands:

```bash
swave generate --config configs/dataset.toml
swave train --config configs/training-48g.toml
swave invert --config configs/inversion.toml --experiment both
swave inversion-report \
  --results-dir results/inversion \
  --dataset-dir data/production \
  --output-dir results/inversion-report
```

For a four-task cluster, document four invocations differing only in `--task-index 0` through `3` with `--task-count 4`, followed by one report command after all tasks finish.

- [ ] **Step 5: Run the complete verification suite**

Run: `.venv/bin/python -m pytest -q`  
Expected: all tests PASS.

Run: `.venv/bin/ruff check src tests scripts`  
Expected: no lint output.

Run: `.venv/bin/python -m build`  
Expected: wheel and source distribution build successfully.

Run: `git diff --check`  
Expected: no whitespace errors.

- [ ] **Step 6: Commit CLI and documentation**

```bash
git add src/swave/cli.py tests/test_cli.py README.md
git commit -m "docs: expose inversion experiment workflow"
```

- [ ] **Step 7: Record the final local verification evidence**

Run: `git status --short && git log -10 --oneline --decorate`  
Expected: clean status and the inversion task commits above the approved design commit `4905341`.
