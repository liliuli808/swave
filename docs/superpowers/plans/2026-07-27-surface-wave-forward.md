# Pure-Python Surface-Wave Forward Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a standalone Python package that generates one million deterministic 20-parameter geological models, calculates Rayleigh modes 0–3 with mode-kissing recovery, and trains/serves a four-head neural surrogate.

**Architecture:** A `src/swave` package separates empirical material laws, geological sampling, secular-function physics, adaptive root search, curve quality control, HDF5 sharding, and PyTorch training. The physics solver is NumPy/SciPy-only at runtime; dataset workers write independent atomic shards; the neural model consumes 20 `Vs` values and emits four 120-frequency curves.

**Tech Stack:** Python 3.11+, NumPy, SciPy, h5py, Matplotlib, PyTorch, pytest, Ruff, TOML via Python `tomllib`.

## Global Constraints

- Runtime must not invoke, compile, link, or shell out to `mode-kissing` or `QEDispInv`.
- Public units are kilometers, kilometers per second, hertz, and grams per cubic centimeter.
- Input has exactly 20 `Vs` values: 19 finite 0.1 km layers plus a half-space beginning at 1.9 km.
- `Vs` is bounded to 0.30–2.60 km/s.
- Frequencies are exactly 0.5–60.0 Hz in 0.5 Hz increments, yielding 120 points.
- Outputs are Rayleigh modes 0, 1, 2, and 3 ordered by ascending phase velocity at each frequency.
- Default material conversion is Brocher (2005); Gardner and near-surface conversions remain selectable.
- Default model mixture is 25% normal, 15% low velocity, 10% high velocity, and 50% coupled high-then-low anomaly.
- Anomaly positions are restricted to user-facing layers 3–12.
- Root solving uses `float64`; persisted scientific arrays use `float32` only after validation.
- Dataset generation is deterministic from `(global_seed, sample_id, retry_count)`, independent of worker count.
- Production data consists of 1,000,000 accepted samples in 100 independently resumable HDF5 shards by default.
- No failed curve may be replaced with zeros or silently interpolated.
- Development uses red–green–refactor; every behavior change begins with a failing test.

---

## File Responsibility Map

**Project and configuration**

- `pyproject.toml`: package metadata, runtime/dev dependencies, CLI entry point, pytest and Ruff settings.
- `configs/dataset.toml`: production physics, geology, sharding, and seed values.
- `configs/training.toml`: streaming loader, architecture, optimizer, and checkpoint values.
- `src/swave/config.py`: immutable validated configurations, TOML readers, canonical hash.

**Physics and data**

- `src/swave/empirical.py`: three `Vs -> (Vp, density)` laws.
- `src/swave/geology.py`: deterministic 20-value background and anomaly generation.
- `src/swave/secular.py`: unnormalized Dunkin delta-matrix Rayleigh secular function.
- `src/swave/sampling.py`: root-count sampling and supplementary samples.
- `src/swave/solver.py`: bracketing, TOMS 748 refinement, grid solution, recovery.
- `src/swave/quality.py`: curve classification and retry decisions.
- `src/swave/dataset.py`: HDF5 schema, atomic shards, manifest, resume, multiprocessing.

**Surrogate and interfaces**

- `src/swave/network.py`: shared MLP backbone and four output heads.
- `src/swave/training.py`: streaming shards, masked loss, metrics, checkpoint lifecycle.
- `src/swave/inference.py`: checkpoint loading and NumPy prediction API.
- `src/swave/plotting.py`: noninteractive scientific figures.
- `src/swave/cli.py`: `generate`, `train`, `evaluate`, `predict`, and `plot` commands.
- `src/swave/__init__.py`: supported public API and package version.
- `README.md`: setup, smoke run, production run, outputs, and external inversion usage.

---

### Task 1: Package Scaffold and Validated Configuration

**Files:**

- Create: `pyproject.toml`
- Create: `configs/dataset.toml`
- Create: `configs/training.toml`
- Create: `src/swave/__init__.py`
- Create: `src/swave/config.py`
- Create: `tests/test_config.py`

**Interfaces:**

- Produces: `PhysicsConfig`, `GeologyConfig`, `DatasetConfig`, `TrainingConfig`
- Produces: `load_dataset_config(path) -> DatasetConfig`
- Produces: `load_training_config(path) -> TrainingConfig`
- Produces: `canonical_hash(config) -> str`

- [ ] **Step 1: Add packaging and default TOML files**

Create `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "swave-forward"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=2.0",
  "scipy>=1.14",
  "h5py>=3.11",
  "matplotlib>=3.9",
  "torch>=2.4",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-cov>=5.0", "ruff>=0.6"]

[project.scripts]
swave = "swave.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 88
target-version = "py311"
```

Create production defaults in `configs/dataset.toml`:

```toml
[physics]
fmin = 0.5
fmax = 60.0
fstep = 0.5
mode_count = 4
epsilon = 0.5
nfine = 2
quadratic_iterations = 3
root_tolerance = 1e-8
dedup_tolerance = 1e-7
strategy = "quadratic"

[geology]
layers = 20
thickness_km = 0.1
vs_min = 0.30
vs_max = 2.60
anomaly_first_layer = 3
anomaly_last_layer = 12
normal_fraction = 0.25
low_fraction = 0.15
high_fraction = 0.10
coupled_fraction = 0.50
empirical_method = "brocher05"

[dataset]
samples = 1000000
shard_size = 10000
seed = 20260727
workers = 0
output_dir = "data/production"
max_model_retries = 8
```

Create training defaults in `configs/training.toml`:

```toml
[training]
dataset_dir = "data/production"
output_dir = "runs/default"
batch_size = 512
epochs = 100
learning_rate = 0.001
weight_decay = 0.0001
num_workers = 4
seed = 20260727
device = "auto"
resume = true
```

- [ ] **Step 2: Write failing configuration tests**

```python
from pathlib import Path

import pytest

from swave.config import (
    DatasetConfig,
    PhysicsConfig,
    canonical_hash,
    load_dataset_config,
)


def test_default_frequency_grid_has_120_exact_points() -> None:
    cfg = PhysicsConfig()
    assert cfg.frequencies[0] == pytest.approx(0.5)
    assert cfg.frequencies[-1] == pytest.approx(60.0)
    assert len(cfg.frequencies) == 120


def test_invalid_model_mixture_is_rejected() -> None:
    with pytest.raises(ValueError, match="fractions must sum to 1"):
        DatasetConfig.from_mapping(
            {"geology": {"normal_fraction": 0.9, "coupled_fraction": 0.9}}
        )


def test_config_hash_is_independent_of_mapping_order(tmp_path: Path) -> None:
    left = DatasetConfig()
    right = DatasetConfig.from_mapping(DatasetConfig().to_dict())
    assert canonical_hash(left) == canonical_hash(right)


def test_loads_production_config() -> None:
    cfg = load_dataset_config(Path("configs/dataset.toml"))
    assert cfg.samples == 1_000_000
    assert cfg.physics.mode_count == 4
```

- [ ] **Step 3: Run tests and verify the import failure**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'swave'`.

- [ ] **Step 4: Implement frozen validated dataclasses**

Implement dataclasses with `__post_init__` validation and canonical JSON hashing:

```python
@dataclass(frozen=True)
class PhysicsConfig:
    fmin: float = 0.5
    fmax: float = 60.0
    fstep: float = 0.5
    mode_count: int = 4
    epsilon: float = 0.5
    nfine: int = 2
    quadratic_iterations: int = 3
    root_tolerance: float = 1e-8
    dedup_tolerance: float = 1e-7
    strategy: str = "quadratic"

    @property
    def frequencies(self) -> np.ndarray:
        count = round((self.fmax - self.fmin) / self.fstep) + 1
        return self.fmin + np.arange(count, dtype=np.float64) * self.fstep


@dataclass(frozen=True)
class GeologyConfig:
    layers: int = 20
    thickness_km: float = 0.1
    vs_min: float = 0.30
    vs_max: float = 2.60
    anomaly_first_layer: int = 3
    anomaly_last_layer: int = 12
    normal_fraction: float = 0.25
    low_fraction: float = 0.15
    high_fraction: float = 0.10
    coupled_fraction: float = 0.50
    empirical_method: str = "brocher05"


@dataclass(frozen=True)
class DatasetConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    geology: GeologyConfig = field(default_factory=GeologyConfig)
    samples: int = 1_000_000
    shard_size: int = 10_000
    seed: int = 20_260_727
    workers: int = 0
    output_dir: Path = Path("data/production")
    max_model_retries: int = 8
```

`DatasetConfig.from_mapping` must merge nested partial mappings with defaults. `canonical_hash` must use sorted compact JSON and SHA-256.

- [ ] **Step 5: Run the configuration tests**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: 4 passed.

- [ ] **Step 6: Commit the scaffold**

```bash
git add pyproject.toml configs src/swave/__init__.py src/swave/config.py tests/test_config.py
git commit -m "build: scaffold package and validated configuration"
```

---

### Task 2: Empirical Material Relations

**Files:**

- Create: `src/swave/empirical.py`
- Create: `tests/test_empirical.py`

**Interfaces:**

- Produces: `brocher05(vs) -> tuple[NDArray, NDArray]`
- Produces: `gardner(vs) -> tuple[NDArray, NDArray]`
- Produces: `near_surface(vs, vp_vs_ratio) -> tuple[NDArray, NDArray]`
- Produces: `material_properties(vs, method, vp_vs_ratio=1.7321)`

- [ ] **Step 1: Write scalar, vector, and failure tests**

```python
import numpy as np
import pytest

from swave.empirical import material_properties


def test_brocher05_matches_published_polynomials() -> None:
    vs = np.array([0.3, 1.0, 2.6])
    vp, rho = material_properties(vs, "brocher05")
    expected_vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4
    expected_rho = (
        1.6612 * expected_vp
        - 0.4721 * expected_vp**2
        + 0.0671 * expected_vp**3
        - 0.0043 * expected_vp**4
        + 0.000106 * expected_vp**5
    )
    np.testing.assert_allclose(vp, expected_vp)
    np.testing.assert_allclose(rho, expected_rho)


def test_gardner_is_positive_and_vp_exceeds_vs() -> None:
    vs = np.linspace(0.3, 2.6, 20)
    vp, rho = material_properties(vs, "gardner")
    assert np.all(vp > vs)
    assert np.all(rho > 0)


def test_nonfinite_vs_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        material_properties(np.array([0.5, np.nan]), "brocher05")
```

- [ ] **Step 2: Verify the tests fail because the module is missing**

Run:

```bash
python -m pytest tests/test_empirical.py -q
```

Expected: import failure for `swave.empirical`.

- [ ] **Step 3: Implement vectorized laws and validation**

Use exactly these formulas:

```python
def brocher05(vs: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    value = _validate_vs(vs)
    vp = (
        0.9409
        + 2.0947 * value
        - 0.8206 * value**2
        + 0.2683 * value**3
        - 0.0251 * value**4
    )
    rho = (
        1.6612 * vp
        - 0.4721 * vp**2
        + 0.0671 * vp**3
        - 0.0043 * vp**4
        + 0.000106 * vp**5
    )
    return _validate_output(value, vp, rho)


def gardner(vs: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    value = _validate_vs(vs)
    vp = 1.7321 * value
    rho = 0.31 * (1000.0 * vp) ** 0.25
    return _validate_output(value, vp, rho)


def near_surface(
    vs: ArrayLike, vp_vs_ratio: float = 1.7321
) -> tuple[np.ndarray, np.ndarray]:
    value = _validate_vs(vs)
    vp = vp_vs_ratio * value
    rho = -0.22374079 * value**2 + 1.32248261 * value + 1.54840433
    return _validate_output(value, vp, rho)
```

The dispatcher accepts only `brocher05`, `gardner`, and `near_surface`; its error lists all three names.

- [ ] **Step 4: Run the empirical tests**

Run:

```bash
python -m pytest tests/test_empirical.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit empirical relations**

```bash
git add src/swave/empirical.py tests/test_empirical.py
git commit -m "feat: add empirical material property relations"
```

---

### Task 3: Deterministic Geological Model Families

**Files:**

- Create: `src/swave/geology.py`
- Create: `tests/test_geology.py`

**Interfaces:**

- Produces: `ModelKind(IntEnum)`
- Produces: `GeneratedModel(sample_id, kind, vs, vp, density, retry_count)`
- Produces: `generate_model(sample_id, config, global_seed, retry_count=0) -> GeneratedModel`
- Consumes: `GeologyConfig`, `material_properties`

- [ ] **Step 1: Write tests for bounds, families, anomalies, and worker-independent determinism**

```python
import numpy as np

from swave.config import GeologyConfig
from swave.geology import ModelKind, generate_model


def test_same_identity_generates_identical_model() -> None:
    cfg = GeologyConfig()
    left = generate_model(1234, cfg, global_seed=99, retry_count=2)
    right = generate_model(1234, cfg, global_seed=99, retry_count=2)
    np.testing.assert_array_equal(left.vs, right.vs)
    assert left.kind == right.kind


def test_normal_model_is_nondecreasing_and_bounded() -> None:
    cfg = GeologyConfig(
        normal_fraction=1.0,
        low_fraction=0.0,
        high_fraction=0.0,
        coupled_fraction=0.0,
    )
    model = generate_model(7, cfg, global_seed=10)
    assert model.kind is ModelKind.NORMAL
    assert np.all(np.diff(model.vs) >= 0)
    assert np.all((model.vs >= 0.30) & (model.vs <= 2.60))


def test_coupled_model_has_high_then_two_or_three_low_layers_in_target_zone() -> None:
    cfg = GeologyConfig(
        normal_fraction=0.0,
        low_fraction=0.0,
        high_fraction=0.0,
        coupled_fraction=1.0,
    )
    model = generate_model(12, cfg, global_seed=33)
    assert model.kind is ModelKind.COUPLED
    delta = model.vs - model.background_vs
    positive = np.flatnonzero(delta > 0)
    negative = np.flatnonzero(delta < 0)
    assert 2 <= positive[0] <= 11
    assert negative[0] == positive[0] + 1
    assert len(negative) in (2, 3)
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```bash
python -m pytest tests/test_geology.py -q
```

Expected: import failure for `swave.geology`.

- [ ] **Step 3: Implement the generator**

Define family selection with cumulative fractions and a deterministic RNG:

```python
seed = np.random.SeedSequence([global_seed, sample_id, retry_count])
rng = np.random.default_rng(seed)
family_draw = rng.random()
```

Generate background increments using Gamma draws:

```python
surface = rng.uniform(0.30, 0.65)
halfspace = rng.uniform(max(1.60, surface + 0.6), 2.60)
increments = rng.gamma(shape=2.0, scale=1.0, size=19)
increments /= increments.sum()
background = surface + np.r_[0.0, np.cumsum(increments)] * (halfspace - surface)
background = np.convolve(
    np.pad(background, (1, 1), mode="edge"),
    np.array([0.2, 0.6, 0.2]),
    mode="valid",
)
background = np.maximum.accumulate(background)
```

For anomaly magnitudes, draw a relative contrast in `[0.08, 0.35]`. Select user-facing layers 3–12 by zero-based index `[2, 11]`, constraining start positions so the entire anomaly remains in the zone. Preserve `background_vs` in `GeneratedModel` for validation but do not persist it in production HDF5.

Reject a candidate unless:

- shape is `(20,)`;
- all values are finite and within bounds;
- requested perturbations retain at least 5% contrast from background;
- only intended anomaly indices change sign relative to background;
- `Vp > Vs` and density is positive.

- [ ] **Step 4: Run geology tests and a 10,000-model statistical check**

Run:

```bash
python -m pytest tests/test_geology.py -q
python -c "from collections import Counter; from swave.config import GeologyConfig; from swave.geology import generate_model; c=Counter(generate_model(i, GeologyConfig(), 20260727).kind.name for i in range(10000)); print(c)"
```

Expected: tests pass; every family appears and counts are within 2 percentage points of configured fractions.

- [ ] **Step 5: Commit the generator**

```bash
git add src/swave/geology.py tests/test_geology.py
git commit -m "feat: generate deterministic geological model families"
```

---

### Task 4: Unnormalized Dunkin Secular Function

**Files:**

- Create: `src/swave/secular.py`
- Create: `tests/test_secular.py`
- Create: `tests/fixtures/paper_model.txt`

**Interfaces:**

- Produces: `LayeredModel.from_vs(vs, empirical_method) -> LayeredModel`
- Produces: `RayleighSecular(model).evaluate(frequency, phase_velocity) -> float`
- Produces: `SecularNumericalError`
- Consumes: material arrays with 20 rows or explicit fixture rows.

- [ ] **Step 1: Add the paper model fixture**

`tests/fixtures/paper_model.txt`:

```text
1 0.00 1.78 0.18 1.50
2 0.01 1.85 0.35 1.70
3 0.02 1.80 0.25 1.60
4 0.04 1.93 0.60 1.90
```

- [ ] **Step 2: Write behavior tests before the implementation**

```python
import numpy as np
import pytest

from swave.secular import LayeredModel, RayleighSecular


def test_layered_model_rejects_nonincreasing_depth() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        LayeredModel(
            depth=np.array([0.0, 0.1, 0.1]),
            density=np.ones(3) * 2,
            vs=np.ones(3),
            vp=np.ones(3) * 2,
        )


def test_paper_model_secular_values_are_finite_and_change_sign_near_roots() -> None:
    raw = np.loadtxt("tests/fixtures/paper_model.txt")
    model = LayeredModel(raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4])
    secular = RayleighSecular(model)
    values = np.array([secular.evaluate(19.7, c) for c in np.linspace(0.16, 0.59, 4000)])
    assert np.all(np.isfinite(values))
    assert np.count_nonzero(values[:-1] * values[1:] < 0) >= 2


def test_invalid_frequency_and_velocity_are_rejected() -> None:
    raw = np.loadtxt("tests/fixtures/paper_model.txt")
    secular = RayleighSecular(LayeredModel(raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]))
    with pytest.raises(ValueError, match="frequency"):
        secular.evaluate(0.0, 0.3)
    with pytest.raises(ValueError, match="phase_velocity"):
        secular.evaluate(10.0, -0.3)
```

- [ ] **Step 3: Run the secular tests and verify failure**

Run:

```bash
python -m pytest tests/test_secular.py -q
```

Expected: import failure for `swave.secular`.

- [ ] **Step 4: Implement model validation and the delta matrix**

Implement the QEDispInv paper’s unnormalized Dunkin formulation in these private units:

```python
def _layer_variables(
    p: float,
    q: float,
    ra: float,
    rb: float,
    wavenumber: float,
    xka: float,
    xkb: float,
    thickness: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    if wavenumber < xka:
        sinp = np.sin(p)
        w = sinp / ra
        x = -ra * sinp
        cosp = np.cos(p)
        pex = 0.0
    elif wavenumber == xka:
        w = thickness
        x = 0.0
        cosp = 1.0
        pex = 0.0
    else:
        pex = p
        fac = np.exp(-2.0 * p) if p < 16.0 else 0.0
        cosp = 0.5 * (1.0 + fac)
        sinp = 0.5 * (1.0 - fac)
        w = sinp / ra
        x = ra * sinp

    if wavenumber < xkb:
        sinq = np.sin(q)
        y = sinq / rb
        z = -rb * sinq
        cosq = np.cos(q)
        sex = 0.0
    elif wavenumber == xkb:
        y = thickness
        z = 0.0
        cosq = 1.0
        sex = 0.0
    else:
        sex = q
        fac = np.exp(-2.0 * q) if q < 16.0 else 0.0
        cosq = 0.5 * (1.0 + fac)
        sinq = 0.5 * (1.0 - fac)
        y = sinq / rb
        z = rb * sinq

    exponent = pex + sex
    a0 = np.exp(-exponent) if exponent < 60.0 else 0.0
    return (
        w,
        cosp,
        a0,
        cosp * cosq,
        cosp * y,
        cosp * z,
        cosq * w,
        cosq * x,
        x * y,
        x * z,
        w * y,
        w * z,
    )
```

Implement `_dunkin_matrix` as:

```python
def _dunkin_matrix(
    wavenumber2: float,
    gam: float,
    gammk: float,
    rho: float,
    a0: float,
    cpcq: float,
    cpy: float,
    cpz: float,
    cqw: float,
    cqx: float,
    xy: float,
    xz: float,
    wy: float,
    wz: float,
) -> np.ndarray:
    ca = np.zeros((5, 5), dtype=np.float64)
    gamm1 = gam - 1.0
    twgm1 = gam + gamm1
    gmgmk = gam * gammk
    gmgm1 = gam * gamm1
    gm1sq = gamm1 * gamm1
    rho2 = rho * rho
    a0pq = a0 - cpcq
    t = -2.0 * wavenumber2

    ca[0, 0] = cpcq - 2.0 * gmgm1 * a0pq - gmgmk * xz - wavenumber2 * gm1sq * wy
    ca[0, 1] = (wavenumber2 * cpy - cqx) / rho
    ca[0, 2] = -(twgm1 * a0pq + gammk * xz + wavenumber2 * gamm1 * wy) / rho
    ca[0, 3] = (cpz - wavenumber2 * cqw) / rho
    ca[0, 4] = -(2.0 * wavenumber2 * a0pq + xz + wavenumber2**2 * wy) / rho2

    ca[1, 0] = (gmgmk * cpz - gm1sq * cqw) * rho
    ca[1, 1] = cpcq
    ca[1, 2] = gammk * cpz - gamm1 * cqw
    ca[1, 3] = -wz
    ca[1, 4] = ca[0, 3]

    ca[3, 0] = (gm1sq * cpy - gmgmk * cqx) * rho
    ca[3, 1] = -xy
    ca[3, 2] = gamm1 * cpy - gammk * cqx
    ca[3, 3] = ca[1, 1]
    ca[3, 4] = ca[0, 1]

    ca[4, 0] = -(
        2.0 * gmgmk * gm1sq * a0pq
        + gmgmk**2 * xz
        + gm1sq**2 * wy
    ) * rho2
    ca[4, 1] = ca[3, 0]
    ca[4, 2] = -(
        gammk * gamm1 * twgm1 * a0pq
        + gam * gammk**2 * xz
        + gamm1 * gm1sq * wy
    ) * rho
    ca[4, 3] = ca[1, 0]
    ca[4, 4] = ca[0, 0]

    ca[2, 0] = t * ca[4, 2]
    ca[2, 1] = t * ca[3, 2]
    ca[2, 2] = a0 + 2.0 * (cpcq - ca[0, 0])
    ca[2, 3] = t * ca[1, 2]
    ca[2, 4] = t * ca[0, 2]
    return ca
```

Implement bottom-half-space initialization and bottom-up multiplication:

```python
state = np.array(
    [
        rho**2 * (gamm1**2 - gam * gammk * ra * rb),
        -rho * ra,
        rho * (gamm1 - gammk * ra * rb),
        rho * rb,
        wavenumber**2 - ra * rb,
    ],
    dtype=np.float64,
)
for layer in range(model.layers - 2, -1, -1):
    (
        _w,
        _cosp,
        a0,
        cpcq,
        cpy,
        cpz,
        cqw,
        cqx,
        xy,
        xz,
        wy,
        wz,
    ) = _layer_variables(
        p, q, ra, rb, wavenumber, xka, xkb, model.thickness[layer]
    )
    state = _dunkin_matrix(
        wavenumber2,
        gam,
        gammk,
        model.density[layer],
        a0,
        cpcq,
        cpy,
        cpz,
        cqw,
        cqx,
        xy,
        xz,
        wy,
        wz,
    ).T @ state
return float(state[0])
```

Use `np.errstate(over="raise", invalid="raise", divide="raise")` and convert `FloatingPointError` to `SecularNumericalError(frequency, phase_velocity)`.

- [ ] **Step 5: Run secular tests and scan a broad valid grid**

Run:

```bash
python -m pytest tests/test_secular.py -q
python -c "import numpy as np; from swave.secular import LayeredModel, RayleighSecular; r=np.loadtxt('tests/fixtures/paper_model.txt'); s=RayleighSecular(LayeredModel(r[:,1],r[:,2],r[:,3],r[:,4])); a=np.array([s.evaluate(f,c) for f in np.linspace(.5,60,24) for c in np.linspace(.15,.59,80)]); print(a.size, np.isfinite(a).all())"
```

Expected: tests pass; output ends with `1920 True`.

- [ ] **Step 6: Commit the secular function**

```bash
git add src/swave/secular.py tests/test_secular.py tests/fixtures/paper_model.txt
git commit -m "feat: implement Rayleigh Dunkin secular function"
```

---

### Task 5: Adaptive Sampling and Three Root Strategies

**Files:**

- Create: `src/swave/sampling.py`
- Create: `src/swave/solver.py`
- Create: `tests/test_sampling.py`
- Create: `tests/test_solver.py`

**Interfaces:**

- Produces: `estimate_mode_count(model, frequency, c) -> float`
- Produces: `initial_phase_samples(model, frequency, config) -> NDArray`
- Produces: `quadratic_vertex(x, y) -> float | None`
- Produces: `DispersionSolver.solve_frequency(frequency, strategy) -> FrequencySolution`
- Produces: `DispersionSolver.solve_grid(frequencies, strategy=None) -> DispersionResult`

- [ ] **Step 1: Write analytic sampling tests**

```python
import numpy as np
import pytest

from swave.sampling import quadratic_vertex


def test_quadratic_vertex_recovers_known_extremum() -> None:
    x = np.array([1.0, 2.0, 4.0])
    y = (x - 2.5) ** 2 - 0.25
    assert quadratic_vertex(x, y) == pytest.approx(2.5)


def test_quadratic_vertex_rejects_linear_and_outside_fit() -> None:
    assert quadratic_vertex(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) is None
    x = np.array([1.0, 2.0, 3.0])
    assert quadratic_vertex(x, (x - 8.0) ** 2) is None
```

- [ ] **Step 2: Write solver tests for ordering, deduplication, and paper-model recovery**

```python
import numpy as np

from swave.config import PhysicsConfig
from swave.secular import LayeredModel
from swave.solver import DispersionSolver, deduplicate_roots


def test_root_deduplication_preserves_ascending_distinct_values() -> None:
    roots = deduplicate_roots([0.4, 0.3, 0.30000000001, 0.5], tolerance=1e-7)
    np.testing.assert_allclose(roots, [0.3, 0.4, 0.5])


def test_quadratic_strategy_finds_at_least_raw_roots_at_paper_kissing_frequency() -> None:
    raw = np.loadtxt("tests/fixtures/paper_model.txt")
    model = LayeredModel(raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4])
    solver = DispersionSolver(model, PhysicsConfig())
    baseline = solver.solve_frequency(19.7, strategy="raw")
    improved = solver.solve_frequency(19.7, strategy="quadratic")
    assert len(improved.roots) >= len(baseline.roots)
    assert np.all(np.diff(improved.roots) > 0)
```

- [ ] **Step 3: Run both tests and confirm missing-module failures**

Run:

```bash
python -m pytest tests/test_sampling.py tests/test_solver.py -q
```

Expected: imports fail for `swave.sampling` and `swave.solver`.

- [ ] **Step 4: Implement initial sampling**

`estimate_mode_count` computes:

```python
c2 = c**-2
s_term = np.sqrt(np.maximum(model.vs[:-1] ** -2 - c2, 0.0))
r_term = np.sqrt(np.maximum(model.vp[:-1] ** -2 - c2, 0.0))
return 2.0 * frequency * np.sum((s_term + r_term) * model.thickness)
```

Build monotonically increasing samples between `0.8 * min(Vs)` and `Vs_halfspace - 1e-5` so adjacent estimates differ by no more than `epsilon`; add the homogeneous Rayleigh velocity and both sides of `min(Vs)`; refine every interval `nfine` times; sort and deduplicate.

- [ ] **Step 5: Implement supplementary sampling**

For the quadratic strategy:

1. evaluate the initial samples;
2. find the vertex of each consecutive triplet;
3. keep a vertex only when strictly between the triplet endpoints and at least `dedup_tolerance` from an existing sample;
4. evaluate inserted points;
5. sort samples and repeat `quadratic_iterations`.

For the degraded strategy:

1. find every strict local `Vs` minimum at index `i`;
2. build truncated models ending immediately before each local minimum’s deeper continuation, then append the complete model;
3. solve truncated models in shallow-to-deep order;
4. insert `root - 1e-6`, `root`, `root + 1e-6`, and both neighboring midpoints into the next model’s samples;
5. discard supplementary samples at or above the full-model half-space `Vs`.

- [ ] **Step 6: Implement bracketing and TOMS 748 refinement**

For each adjacent finite pair with opposite signs, call:

```python
root = scipy.optimize.toms748(
    lambda c: secular.evaluate(frequency, c),
    left,
    right,
    xtol=config.root_tolerance,
    rtol=4 * np.finfo(float).eps,
    maxiter=100,
)
```

Exact-zero samples are roots. Sort and merge roots with `dedup_tolerance`; retain at most `mode_count`.

`solve_grid` allocates `(4, len(frequencies))` with `NaN`, fills roots by modal order, and returns matching masks and per-frequency status values.

- [ ] **Step 7: Run sampling and solver tests**

Run:

```bash
python -m pytest tests/test_sampling.py tests/test_solver.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit all root-search strategies**

```bash
git add src/swave/sampling.py src/swave/solver.py tests/test_sampling.py tests/test_solver.py
git commit -m "feat: add adaptive multimode root search"
```

---

### Task 6: Curve Quality and Deterministic Recovery

**Files:**

- Create: `src/swave/quality.py`
- Modify: `src/swave/solver.py`
- Create: `tests/test_quality.py`

**Interfaces:**

- Produces: `QualityFlag(IntFlag)`
- Produces: `assess_dispersion(result) -> QualityReport`
- Produces: `solve_with_recovery(model, physics) -> DispersionResult`

- [ ] **Step 1: Write tests distinguishing physical leading gaps from internal gaps**

```python
import numpy as np

from swave.quality import QualityFlag, assess_arrays


def test_leading_missing_higher_mode_is_allowed() -> None:
    values = np.ones((4, 8))
    mask = np.ones((4, 8), dtype=bool)
    mask[3, :3] = False
    report = assess_arrays(values, mask)
    assert not (report.flags & QualityFlag.INTERNAL_GAP)


def test_internal_missing_cell_requests_recovery() -> None:
    values = np.ones((4, 8))
    mask = np.ones((4, 8), dtype=bool)
    mask[2, 4] = False
    report = assess_arrays(values, mask)
    assert report.flags & QualityFlag.INTERNAL_GAP
    assert report.retry_required


def test_nonfinite_valid_value_is_hard_failure() -> None:
    values = np.ones((4, 8))
    mask = np.ones((4, 8), dtype=bool)
    values[0, 2] = np.nan
    report = assess_arrays(values, mask)
    assert report.flags & QualityFlag.NONFINITE_VALID
```

- [ ] **Step 2: Verify the quality tests fail**

Run:

```bash
python -m pytest tests/test_quality.py -q
```

Expected: import failure for `swave.quality`.

- [ ] **Step 3: Implement quality flags and the bounded recovery pass**

Flags are:

```python
class QualityFlag(IntFlag):
    OK = 0
    INTERNAL_GAP = 1
    NONFINITE_VALID = 2
    ROOT_ORDER = 4
    NUMERICAL_FAILURE = 8
    RECOVERED = 16
```

An internal gap exists when a false mask cell lies between the first and last true cell of one mode. Root order fails when any frequency has a nonpositive difference among valid modes.

Recovery clones `PhysicsConfig` with `nfine + 1` and `quadratic_iterations + 2`, reruns only failing frequencies, and replaces a frequency only if its quality improves. Persistent internal gaps remain marked and cause model regeneration at the dataset layer.

- [ ] **Step 4: Run quality and solver tests**

Run:

```bash
python -m pytest tests/test_quality.py tests/test_solver.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit quality control**

```bash
git add src/swave/quality.py src/swave/solver.py tests/test_quality.py tests/test_solver.py
git commit -m "feat: validate and recover incomplete dispersion curves"
```

---

### Task 7: Atomic HDF5 Shards and Resumable Generation

**Files:**

- Create: `src/swave/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**

- Produces: `generate_dataset(config) -> Manifest`
- Produces: `generate_shard(shard_id, config) -> ShardResult`
- Produces: `load_manifest(path) -> Manifest`
- Consumes: `generate_model`, `solve_with_recovery`, `assess_dispersion`

- [ ] **Step 1: Write schema and deterministic-resume tests**

```python
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from swave.config import DatasetConfig
from swave.dataset import generate_dataset


def test_smoke_dataset_has_declared_schema(tmp_path: Path) -> None:
    cfg = replace(DatasetConfig(), samples=4, shard_size=2, workers=1, output_dir=tmp_path)
    manifest = generate_dataset(cfg)
    assert manifest.complete
    files = sorted(tmp_path.glob("shard-*.h5"))
    assert len(files) == 2
    with h5py.File(files[0]) as handle:
        assert handle["vs"].shape == (2, 20)
        assert handle["phase_velocity"].shape == (2, 4, 120)
        assert handle["valid_mask"].shape == (2, 4, 120)


def test_resume_does_not_rewrite_complete_shard(tmp_path: Path) -> None:
    cfg = replace(DatasetConfig(), samples=2, shard_size=2, workers=1, output_dir=tmp_path)
    generate_dataset(cfg)
    shard = next(tmp_path.glob("shard-*.h5"))
    before = shard.read_bytes()
    generate_dataset(cfg)
    assert shard.read_bytes() == before


def test_conflicting_configuration_is_rejected(tmp_path: Path) -> None:
    first = replace(DatasetConfig(), samples=2, shard_size=2, workers=1, output_dir=tmp_path)
    generate_dataset(first)
    second = replace(first, seed=first.seed + 1)
    with pytest.raises(ValueError, match="configuration hash"):
        generate_dataset(second)
```

- [ ] **Step 2: Run the dataset tests and confirm module failure**

Run:

```bash
python -m pytest tests/test_dataset.py -q
```

Expected: import failure for `swave.dataset`.

- [ ] **Step 3: Implement HDF5 schema and atomic publication**

Create datasets with gzip compression, shuffle enabled, and sample-major chunks:

```python
handle.create_dataset("sample_id", data=sample_ids, dtype="u8")
handle.create_dataset("model_kind", data=kinds, dtype="u1")
handle.create_dataset("vs", data=vs, dtype="f4", chunks=(min(256, n), 20), compression="gzip", shuffle=True)
handle.create_dataset("vp", data=vp, dtype="f4", chunks=(min(256, n), 20), compression="gzip", shuffle=True)
handle.create_dataset("density", data=rho, dtype="f4", chunks=(min(256, n), 20), compression="gzip", shuffle=True)
handle.create_dataset("phase_velocity", data=phase, dtype="f4", chunks=(min(32, n), 4, 120), compression="gzip", shuffle=True)
handle.create_dataset("valid_mask", data=mask, dtype="?", chunks=(min(64, n), 4, 120), compression="gzip", shuffle=True)
handle.create_dataset("quality_flags", data=flags, dtype="u2")
handle.create_dataset("retry_count", data=retries, dtype="u1")
```

Write to `shard-00000.h5.tmp-<pid>`, flush and close, compute SHA-256, then `Path.replace()` to `shard-00000.h5`. Attributes include configuration hash, shard ID, first/last sample ID, accepted count, and checksum of uncompressed sample IDs.

- [ ] **Step 4: Implement manifest and worker coordination**

Manifest JSON contains exact keys:

```json
{
  "schema_version": 1,
  "config_hash": "0000000000000000000000000000000000000000000000000000000000000000",
  "global_seed": 20260727,
  "expected_shards": 100,
  "completed_shards": [],
  "accepted_by_kind": {},
  "rejected_by_reason": {},
  "recovered_models": 0,
  "complete": false
}
```

The coordinator uses `ProcessPoolExecutor`; `workers=0` resolves to `max(1, os.cpu_count() - 1)`. Only the coordinator updates the manifest via an atomic temporary JSON rename. A complete shard with matching attributes is skipped.

- [ ] **Step 5: Run dataset tests**

Run:

```bash
python -m pytest tests/test_dataset.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit dataset generation**

```bash
git add src/swave/dataset.py tests/test_dataset.py
git commit -m "feat: generate resumable HDF5 training shards"
```

---

### Task 8: Four-Head PyTorch Network

**Files:**

- Create: `src/swave/network.py`
- Create: `tests/test_network.py`

**Interfaces:**

- Produces: `FourHeadForwardModel(nn.Module)`
- Produces: `masked_smooth_l1(prediction, target, mask) -> Tensor`

- [ ] **Step 1: Write architecture and masked-loss tests**

```python
import torch

from swave.network import FourHeadForwardModel, masked_smooth_l1


def test_network_output_shape() -> None:
    model = FourHeadForwardModel()
    output = model(torch.randn(7, 20))
    assert output.shape == (7, 4, 120)


def test_masked_loss_ignores_invalid_cells_and_balances_modes() -> None:
    prediction = torch.zeros(1, 4, 2)
    target = torch.ones(1, 4, 2)
    mask = torch.tensor([[[1, 1], [1, 0], [0, 0], [1, 1]]], dtype=torch.bool)
    loss = masked_smooth_l1(prediction, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.5


def test_empty_batch_mode_does_not_create_nan() -> None:
    prediction = torch.zeros(2, 4, 3)
    target = torch.zeros_like(prediction)
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[:, 0] = True
    assert torch.isfinite(masked_smooth_l1(prediction, target, mask))
```

- [ ] **Step 2: Run network tests and verify module failure**

Run:

```bash
python -m pytest tests/test_network.py -q
```

Expected: import failure for `swave.network`.

- [ ] **Step 3: Implement residual backbone and four heads**

```python
class ResidualBlock(nn.Module):
    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, 512),
            nn.GELU(),
            nn.Linear(512, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, value: Tensor) -> Tensor:
        return self.norm(value + self.layers(value))


class FourHeadForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(20, 256), nn.GELU())
        self.backbone = nn.Sequential(*(ResidualBlock() for _ in range(4)))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 120))
            for _ in range(4)
        )

    def forward(self, vs: Tensor) -> Tensor:
        features = self.backbone(self.input(vs))
        return torch.stack([head(features) for head in self.heads], dim=1)
```

The masked loss calculates Smooth-L1 with `reduction="none"`, averages valid cells independently per nonempty mode, then averages the nonempty modal losses.

- [ ] **Step 4: Run network tests**

Run:

```bash
python -m pytest tests/test_network.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit the network**

```bash
git add src/swave/network.py tests/test_network.py
git commit -m "feat: add four-head forward surrogate network"
```

---

### Task 9: Streaming Training, Metrics, Checkpoints, and Inference

**Files:**

- Create: `src/swave/training.py`
- Create: `src/swave/inference.py`
- Create: `tests/test_training.py`
- Create: `tests/test_inference.py`

**Interfaces:**

- Produces: `HDF5ShardDataset`
- Produces: `compute_normalization(dataset) -> Normalization`
- Produces: `train(config) -> Path`
- Produces: `evaluate(checkpoint, dataset_dir) -> dict`
- Produces: `ForwardPredictor.load(checkpoint) -> ForwardPredictor`
- Produces: `ForwardPredictor.predict(vs) -> NDArray`

- [ ] **Step 1: Write a one-epoch CPU training and inference test**

```python
from dataclasses import replace
from pathlib import Path

import numpy as np

from swave.config import DatasetConfig, TrainingConfig
from swave.dataset import generate_dataset
from swave.inference import ForwardPredictor
from swave.training import train


def test_one_epoch_produces_loadable_checkpoint(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    generate_dataset(
        replace(DatasetConfig(), samples=16, shard_size=8, workers=1, output_dir=data_dir)
    )
    checkpoint = train(
        replace(
            TrainingConfig(),
            dataset_dir=data_dir,
            output_dir=run_dir,
            batch_size=4,
            epochs=1,
            num_workers=0,
            device="cpu",
        )
    )
    predictor = ForwardPredictor.load(checkpoint, device="cpu")
    output = predictor.predict(np.linspace(0.4, 2.0, 20))
    assert output.shape == (4, 120)
    assert np.all(np.isfinite(output))
```

- [ ] **Step 2: Run the training test and verify missing modules**

Run:

```bash
python -m pytest tests/test_training.py tests/test_inference.py -q
```

Expected: imports fail for training and inference modules.

- [ ] **Step 3: Implement deterministic split-aware streaming**

`HDF5ShardDataset` indexes `(shard_path, row)` pairs whose `sample_id % 100` is in:

- training: `0..89`;
- validation: `90..94`;
- testing: `95..99`.

Each process lazily opens its own read-only HDF5 handle. `__getitem__` returns `vs`, `phase_velocity`, and `valid_mask`; invalid `NaN` targets are replaced by zero only in the returned tensor while the false mask prevents loss contribution.

Compute per-layer input mean/std and per-mode target mean/std from training valid cells only. Clamp any standard deviation below `1e-8` to one.

- [ ] **Step 4: Implement training and evaluation**

Use AdamW and cosine annealing:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.epochs
)
```

Seed Python, NumPy, and PyTorch. Resolve device `auto` in CUDA, MPS, CPU order. Save `last.pt` after every epoch and replace `best.pt` when validation MAE improves.

Checkpoint keys are:

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
    "best_validation_mae": best_mae,
    "input_mean": normalization.input_mean,
    "input_std": normalization.input_std,
    "target_mean": normalization.target_mean,
    "target_std": normalization.target_std,
    "dataset_config_hash": manifest.config_hash,
    "training_config": config.to_dict(),
    "torch_rng_state": torch.get_rng_state(),
    "numpy_rng_state": np.random.get_state(),
}
```

Evaluation reports MAE, RMSE, P95 absolute error, and valid count for each mode in km/s.

- [ ] **Step 5: Implement NumPy inference**

`ForwardPredictor.predict` accepts `(20,)` or `(N, 20)`, validates finite bounds, standardizes inputs, performs `torch.inference_mode()`, de-normalizes output, and restores single-model shape. `predict_with_frequencies` pairs output with `np.arange(0.5, 60.0 + 0.25, 0.5)`.

- [ ] **Step 6: Run training and inference tests**

Run:

```bash
python -m pytest tests/test_training.py tests/test_inference.py -q
```

Expected: all tests pass with finite one-epoch loss.

- [ ] **Step 7: Commit training and inference**

```bash
git add src/swave/training.py src/swave/inference.py tests/test_training.py tests/test_inference.py
git commit -m "feat: train evaluate and serve forward surrogate"
```

---

### Task 10: Plotting, CLI, and User Documentation

**Files:**

- Create: `src/swave/plotting.py`
- Create: `src/swave/cli.py`
- Create: `tests/test_cli.py`
- Create: `README.md`

**Interfaces:**

- Produces CLI: `swave generate`
- Produces CLI: `swave train`
- Produces CLI: `swave evaluate`
- Produces CLI: `swave predict`
- Produces CLI: `swave plot-model`
- Produces CLI: `swave plot-dispersion`

- [ ] **Step 1: Write CLI smoke tests**

```python
from pathlib import Path

from swave.cli import main


def test_help_lists_all_workflows(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    for command in ("generate", "train", "evaluate", "predict", "plot-model", "plot-dispersion"):
        assert command in output


def test_plot_model_creates_nonempty_png(tmp_path: Path) -> None:
    output = tmp_path / "model.png"
    assert main(["plot-model", "--sample-id", "4", "--output", str(output)]) == 0
    assert output.stat().st_size > 1000
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: import failure for `swave.cli`.

- [ ] **Step 3: Implement noninteractive plots**

Force the `Agg` backend before importing `pyplot`. Plot:

- stepped `Vs`, `Vp`, and density against inverted depth;
- one line per valid dispersion mode;
- optional comparison lines for raw, degraded, and quadratic strategies;
- training and validation histories from checkpoint-adjacent JSON.

Every plotting function accepts an explicit output `Path`, creates parents, saves at 180 DPI, closes the figure, and returns the path.

- [ ] **Step 4: Implement argparse commands**

`main(argv: Sequence[str] | None = None) -> int` builds subparsers and maps each command to one handler. `generate` accepts `--config`, `--samples`, `--workers`, and `--output-dir`; `train` accepts `--config`, `--device`, and `--epochs`; `predict` accepts a checkpoint plus either 20 command-line `Vs` values or a text file.

Exceptions deriving from `ValueError`, `OSError`, and project scientific errors produce a single `error: <message>` line on stderr and return exit code 2.

- [ ] **Step 5: Write the README**

Document:

1. Python 3.11 virtual environment and `pip install -e ".[dev]"`;
2. 16-sample smoke generation;
3. full one-million-sample resumable generation;
4. one-epoch smoke training and production training;
5. evaluation, prediction, and plot commands;
6. HDF5 schema and units;
7. `ForwardPredictor` external inversion example;
8. expected benchmark report before production generation;
9. citations for both supplied papers and repositories;
10. runtime independence from C++ sources.

- [ ] **Step 6: Run CLI tests**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: all tests pass and generated PNG is nonempty.

- [ ] **Step 7: Commit interfaces and documentation**

```bash
git add src/swave/plotting.py src/swave/cli.py tests/test_cli.py README.md
git commit -m "feat: add command line workflows and plots"
```

---

### Task 11: Golden Reference Fixtures and Full Verification

**Files:**

- Create: `tests/fixtures/golden_dispersion.npz`
- Create: `tests/test_reference.py`
- Modify: `README.md`

**Interfaces:**

- Consumes: immutable roots generated once from the supplied QEDispInv reference.
- Verifies: Python roots, root completeness, package, CLI, dataset, network, and documentation.

- [ ] **Step 1: Generate and document golden fixtures**

Use the supplied reference repository only during development to calculate modes 0–3 for:

- the four-layer paper model;
- a monotonic 20-value model;
- a single-LVL 20-value model;
- a coupled high-plus-three-low 20-value model.

Store only model arrays, frequencies, reference phase velocities, and masks in `golden_dispersion.npz`. Add fixture metadata containing reference repository commit, command, units, and generation date. The installed package and tests must read only the fixture, never invoke the executable.

Because the paper fixture has four layers while the other fixtures have 20, store model arrays padded with `NaN` to shape `(4, 20)` and store `layer_count = [4, 20, 20, 20]`. Tests slice each padded array to its declared layer count.

- [ ] **Step 2: Write reference comparisons**

```python
import numpy as np

from swave.secular import LayeredModel
from swave.solver import DispersionSolver


def test_python_roots_match_reference_within_1e_minus_5_km_s() -> None:
    fixture = np.load("tests/fixtures/golden_dispersion.npz")
    for index, count in enumerate(fixture["layer_count"]):
        model = LayeredModel(
            fixture["depth"][index, :count],
            fixture["density"][index, :count],
            fixture["vs"][index, :count],
            fixture["vp"][index, :count],
        )
        result = DispersionSolver(model).solve_grid(fixture["frequencies"])
        common = result.valid_mask & fixture["valid_mask"][index]
        np.testing.assert_allclose(
            result.phase_velocity[common],
            fixture["phase_velocity"][index][common],
            atol=1e-5,
            rtol=0.0,
        )


def test_quadratic_never_has_more_internal_gaps_than_raw() -> None:
    fixture = np.load("tests/fixtures/golden_dispersion.npz")
    for index, count in enumerate(fixture["layer_count"]):
        model = LayeredModel(
            fixture["depth"][index, :count],
            fixture["density"][index, :count],
            fixture["vs"][index, :count],
            fixture["vp"][index, :count],
        )
        solver = DispersionSolver(model)
        raw = solver.solve_grid(fixture["frequencies"], strategy="raw")
        quadratic = solver.solve_grid(
            fixture["frequencies"], strategy="quadratic"
        )
        raw_gap_count = sum(
            np.count_nonzero(
                ~row[np.flatnonzero(row)[0] : np.flatnonzero(row)[-1] + 1]
            )
            for row in raw.valid_mask
            if np.any(row)
        )
        quadratic_gap_count = sum(
            np.count_nonzero(
                ~row[np.flatnonzero(row)[0] : np.flatnonzero(row)[-1] + 1]
            )
            for row in quadratic.valid_mask
            if np.any(row)
        )
        assert quadratic_gap_count <= raw_gap_count
```

- [ ] **Step 3: Run the reference tests**

Run:

```bash
python -m pytest tests/test_reference.py -q
```

Expected: all common roots agree within `1e-5 km/s`; quadratic has no additional internal gaps.

- [ ] **Step 4: Run complete automated verification**

Run:

```bash
python -m pytest -q
ruff check .
python -m build
```

Expected: zero test failures, zero Ruff errors, wheel and source archive created.

- [ ] **Step 5: Verify clean installation without a compiler**

Create a fresh virtual environment, install the built wheel, and run:

```bash
swave --help
swave generate --config configs/dataset.toml --samples 16 --workers 1 --output-dir /tmp/swave-acceptance-data
swave train --config configs/training.toml --dataset-dir /tmp/swave-acceptance-data --epochs 1 --device cpu --output-dir /tmp/swave-acceptance-run
swave evaluate --checkpoint /tmp/swave-acceptance-run/best.pt --dataset-dir /tmp/swave-acceptance-data
```

Expected: every command exits zero; two shards or the configured smoke shards are complete; a loadable checkpoint and per-mode metrics are emitted.

- [ ] **Step 6: Benchmark one shard and document production estimates**

Run a timed 100-model shard on the available CPU. Record:

- models per second;
- secular evaluations per model;
- recovery rate;
- bytes per accepted sample;
- projected one-million-model wall time at 1, configured, and detected worker counts;
- projected disk storage.

Insert the measured table into `README.md`; label projections as estimates.

- [ ] **Step 7: Re-run verification after documentation changes**

Run:

```bash
python -m pytest -q
ruff check .
python -m build
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 8: Commit fixtures and verified release state**

```bash
git add tests/fixtures/golden_dispersion.npz tests/test_reference.py README.md
git commit -m "test: validate solver and end-to-end forward workflow"
```
