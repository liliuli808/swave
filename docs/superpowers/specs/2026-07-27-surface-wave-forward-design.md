# Pure-Python Surface-Wave Forward Modeling and Surrogate Design

## 1. Objective

Build a standalone Python project that:

1. generates geologically plausible 20-parameter shear-wave velocity (`Vs`) models;
2. computes Rayleigh-wave phase velocities for modes 0–3 on the fixed frequency grid 0.5–60.0 Hz at 0.5 Hz spacing;
3. resolves closely spaced roots around low-velocity-layer mode-kissing zones using the two methods described by Pan and collaborators;
4. produces a deterministic, resumable, one-million-sample training dataset; and
5. trains and serves a shared-backbone, four-head neural surrogate for forward prediction.

The delivered runtime must not invoke, compile, link, or shell out to the `mode-kissing` or `QEDispInv` C++ executables. NumPy, SciPy, Numba, HDF5, Matplotlib, and PyTorch binary wheels are allowed; a C++ compiler is not a runtime prerequisite. Numba uses its bundled LLVM toolchain to cache-compile the scalar secular-function hot path on first use while retaining a pure-Python reference backend for parity tests.

Inversion is outside this project because the supplied requirements state that inversion code already exists. The surrogate will expose stable NumPy and PyTorch inference interfaces that an external inversion program can call.

## 2. Sources and Attribution

The mathematical and algorithmic design is based on:

- `../bssa-2025003.1.pdf`: Pan and Chen (2025), degraded-model referencing for mode-kissing root recovery.
- `../bssa-2025207.1.pdf`: Pan, Wang, and Chen (2026), quadratic-extrema interpolation for mode-kissing root recovery.
- `../mode-kissing`: reference implementation and examples for the 2025 method.
- `../QEDispInv`: reference implementation and examples for the 2026 method.

The implementation will be written as a Python-native implementation with attribution in the documentation. The project will not require either reference repository at runtime. Reference executable output may be used during development to create small, immutable golden test fixtures.

## 3. Units and Model Interpretation

All public APIs use:

- depth and thickness: kilometers;
- velocity: kilometers per second;
- frequency: hertz;
- density: grams per cubic centimeter.

The user-facing model contains exactly 20 `Vs` values, preserving a fixed 20-feature neural-network input. Model row `i` starts at depth `0.1 * i` km for `i = 0..19`.

- Rows 0–18 are 19 finite layers, each 0.1 km (100 m) thick.
- Row 19 begins at 1.9 km and is the elastic half-space required by the dispersion formulation.

The phrase “20 layers, all 100 m thick” is therefore represented by 20 regularly spaced material samples while treating the final sample as the half-space. No artificial lower boundary is introduced.

The fixed frequency vector is:

```text
[0.5, 1.0, 1.5, ..., 60.0] Hz
```

It contains 120 points. Mode numbering is ascending phase velocity at each frequency: fundamental mode 0 followed by modes 1, 2, and 3.

## 4. Package Architecture

The project will use a `src` layout:

```text
surface-wave-forward/
├── pyproject.toml
├── README.md
├── configs/
│   ├── dataset.toml
│   └── training.toml
├── src/swave/
│   ├── __init__.py
│   ├── config.py
│   ├── empirical.py
│   ├── geology.py
│   ├── secular.py
│   ├── sampling.py
│   ├── solver.py
│   ├── quality.py
│   ├── dataset.py
│   ├── network.py
│   ├── training.py
│   ├── inference.py
│   ├── plotting.py
│   └── cli.py
└── tests/
    ├── fixtures/
    ├── test_empirical.py
    ├── test_geology.py
    ├── test_secular.py
    ├── test_sampling.py
    ├── test_solver.py
    ├── test_quality.py
    ├── test_dataset.py
    ├── test_network.py
    └── test_cli.py
```

Each module has one primary responsibility:

- `config.py`: validated immutable configuration objects and TOML loading.
- `empirical.py`: vectorized `Vs -> Vp, density` conversions.
- `geology.py`: reproducible normal and anomalous 20-value `Vs` generation.
- `secular.py`: Python implementation of the unnormalized Dunkin delta-matrix secular function with a Numba-compiled runtime hot path and a reference backend.
- `sampling.py`: phase-velocity sample selection, degraded-model supplementation, and quadratic-extrema supplementation.
- `solver.py`: bracket detection, TOMS 748 refinement, modal ordering, and recovery retries.
- `quality.py`: model and curve validation with machine-readable failure codes.
- `dataset.py`: multiprocessing orchestration, HDF5 shard writing, manifests, and resume behavior.
- `network.py`: shared backbone and four mode-specific heads.
- `training.py`: streaming data loading, masked loss, checkpoints, metrics, and evaluation.
- `inference.py`: stable single-model and batched prediction APIs.
- `plotting.py`: velocity-model, dispersion-curve, coverage, and training plots.
- `cli.py`: user-facing commands without scientific logic.

## 5. Empirical Material Properties

Three named conversions will be available:

1. `brocher05` (default): the Brocher polynomial for `Vs -> Vp`, followed by the Brocher polynomial for `Vp -> density`.
2. `gardner`: `Vp = 1.7321 Vs` and `density = 0.31 * (1000 Vp)^0.25`.
3. `near_surface`: configurable `Vp/Vs` ratio and the quadratic density relation from the QEDispInv documentation.

`brocher05` is the dataset default because the requested `Vs` interval, 0.3–2.6 km/s, maps into its stated crustal `Vp` calibration interval and avoids an arbitrary property discontinuity at a chosen depth.

Every conversion must produce finite positive values and satisfy `Vp > Vs`. Invalid output rejects the model before forward computation.

## 6. Geological Model Generator

### 6.1 Baseline profiles

The generator first constructs a nondecreasing background profile using:

- a surface `Vs` sampled from 0.30–0.65 km/s;
- a half-space `Vs` sampled from 1.60–2.60 km/s;
- positive stochastic increments normalized to connect the endpoints; and
- a short, normalized smoothing kernel that prevents implausibly jagged layer-to-layer changes.

All values are clipped to the global range 0.30–2.60 km/s. Clipping is followed by validation, not by silent acceptance.

### 6.2 Model families

The default one-million-sample mixture is:

- 25% normal nondecreasing models;
- 15% single low-velocity-layer models;
- 10% single high-velocity-layer models;
- 50% coupled anomaly models.

These proportions are configurable, must sum to one, and deliberately emphasize the teacher-requested coupled anomalies.

Anomaly centers are selected only from user-facing layers 3–12, inclusive. In zero-based arrays this is indices 2–11.

- A single low-velocity anomaly reduces one to three consecutive layers relative to the local background.
- A single high-velocity anomaly increases one or two consecutive layers.
- A coupled anomaly introduces one high-velocity layer followed immediately by two or three low-velocity layers.

Perturbation magnitudes are sampled relative to the local background and neighboring contrast, with a minimum accepted contrast of 5%. The generator rejects candidates that violate the global velocity bounds, remove the requested sign of the anomaly, or create an unintended second anomaly outside the selected interval.

### 6.3 Determinism

Every sample is a pure function of `(global_seed, sample_id)`. NumPy `SeedSequence` creates the per-sample random state. Results must be invariant to worker count, shard order, interruption, and resume.

## 7. Dispersion Physics and Root Search

### 7.1 Secular function

`secular.py` implements the Rayleigh-wave P–SV unnormalized Dunkin delta-matrix formulation in `float64`. It evaluates `D(f, c)` for a model at frequency `f` and candidate phase velocity `c`.

The implementation must:

- reject nonpositive frequency or phase velocity;
- reject malformed models before entering numerical loops;
- preserve the secular function morphology needed by quadratic fitting;
- return a structured numerical-failure status instead of propagating `NaN` or overflow silently.

Love waves and sensitivity kernels are not part of this scope.

### 7.2 Initial samples

The initial velocity range is derived from the minimum solid-layer `Vs`, the half-space `Vs`, and the homogeneous-half-space Rayleigh reference velocity.

Sampling uses the estimated root count

```text
N(c) = 2 f Σ_i (s_i + r_i) h_i
```

with the definitions from the papers. Roots of `N(c) = j * epsilon` define coarse samples, and every interval is halved twice by default. Default constants match the papers and are configurable:

- `epsilon = 0.5`
- initial interval refinements `nfine = 2`
- phase-velocity root tolerance `1e-8 km/s`

### 7.3 Root strategies

Three strategies share the same secular function and final TOMS 748 refinement:

- `raw`: sign-change brackets from initial samples only.
- `degraded`: the 2025 algorithm, which finds each local `Vs` minimum, solves progressively truncated models, and reuses their roots as supplementary samples for the full model.
- `quadratic` (default): the 2026 algorithm, which fits a parabola to every consecutive sample triplet, inserts an in-range vertex, reevaluates the secular function, and repeats for three iterations.

Duplicate roots within `1e-7 km/s` are merged. The first four distinct ascending roots become modes 0–3.

### 7.4 Recovery and missing modes

A missing high mode can be physical at low frequency. The solver therefore returns:

```python
phase_velocity: float64[4, 120]
valid_mask: bool[4, 120]
status: uint8[120]
```

`NaN` is used only in the stored phase-velocity cells for which `valid_mask` is false.

Quality analysis distinguishes:

- a physically not-yet-existing higher mode at the low-frequency edge;
- an isolated internal gap suggesting a missed root;
- a numerical failure;
- a root-ordering or duplicate-root failure.

Isolated internal gaps trigger one bounded recovery pass with one additional initial refinement and two additional quadratic iterations. A model that still contains an internal gap is marked failed and regenerated using the same sample ID plus a deterministic retry counter. At most eight model retries are allowed; exhaustion is recorded as a hard shard failure.

## 8. Dataset Format and Generation

The default dataset contains 1,000,000 accepted models in 100 HDF5 shards of 10,000 samples each. Shard size is configurable.

Each shard contains:

```text
sample_id        uint64   [N]
model_kind       uint8    [N]
vs               float32  [N, 20]
vp               float32  [N, 20]
density          float32  [N, 20]
phase_velocity   float32  [N, 4, 120]
valid_mask       bool     [N, 4, 120]
quality_flags    uint16   [N]
retry_count      uint8    [N]
```

Root solving remains `float64`; conversion to `float32` occurs only after quality checks.

A JSON manifest stores:

- configuration hash;
- global seed;
- expected and completed shard IDs;
- accepted/rejected counts by model family;
- root-recovery counts;
- package version;
- creation time and completion state.

Each worker writes a unique temporary shard and atomically renames it only after all datasets, attributes, and checksums are complete. Existing complete shards with the same configuration hash are skipped. A conflicting configuration stops immediately rather than mixing datasets.

The CLI command is:

```bash
swave generate --config configs/dataset.toml
```

A `--samples` override permits smoke tests without modifying the production configuration.

## 9. Four-Head Neural Surrogate

### 9.1 Inputs and outputs

Input is a standardized tensor of shape `[batch, 20]` containing `Vs`. Material properties are deterministic functions of `Vs`, so duplicating `Vp` and density in the input is unnecessary.

The network output shape is `[batch, 4, 120]`. One head corresponds to each modal order.

### 9.2 Architecture

The shared backbone is:

- linear `20 -> 256`;
- GELU;
- four residual multilayer-perceptron blocks, each `256 -> 512 -> 256`;
- LayerNorm after each residual addition.

Each of four independent heads is:

- linear `256 -> 256`;
- GELU;
- linear `256 -> 120`.

The final output is converted from normalized targets back to km/s by the inference wrapper. No monotonic-frequency constraint is imposed because low-velocity layers can create valid nonmonotonic dispersion.

### 9.3 Splits, normalization, and loss

Dataset splits are assigned by `sample_id % 100`:

- 0–89: training;
- 90–94: validation;
- 95–99: testing.

Normalization statistics are calculated from the training split only and stored in the checkpoint.

The loss is the mean Smooth-L1 error over cells whose `valid_mask` is true. Each mode is normalized by its count of valid cells before the four modal losses are averaged, preventing the fundamental mode from dominating sparse higher modes.

Reported metrics for every mode are MAE, RMSE, 95th-percentile absolute error, and valid-cell count in km/s.

### 9.4 Training and inference

Training supports CPU, CUDA, and Apple MPS with automatic selection and an explicit override. Checkpoints include:

- model weights;
- optimizer and scheduler state;
- epoch and best validation metric;
- architecture and dataset configuration hashes;
- normalization statistics;
- random-number-generator states.

The training command is:

```bash
swave train --config configs/training.toml
```

Inference exposes:

```python
predict(vs: numpy.ndarray) -> numpy.ndarray
predict_with_frequencies(vs: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
```

The first output has shape `[4, 120]` for one model or `[N, 4, 120]` for a batch. The second method also returns the fixed frequency vector.

## 10. Visualization

The project will generate:

- stepped `Vs`, `Vp`, and density profiles versus depth;
- four modal dispersion curves with missing cells omitted;
- comparison plots for raw, degraded, and quadratic root strategies;
- model-family distribution and anomaly-depth histograms;
- per-mode valid-data coverage heatmaps;
- training/validation loss and per-mode error plots;
- surrogate-versus-physics scatter and dispersion overlays.

Plots are saved noninteractively by default. Displaying a GUI is an explicit CLI option.

## 11. Error Handling

Configuration errors stop before work begins and include the invalid field and accepted range.

Scientific failures use typed exceptions internally and stable integer flags in HDF5. No worker may replace a failed curve with zero, interpolate through an unresolved gap, or silently drop a sample.

Worker exceptions are returned to the coordinator with sample and shard IDs. The coordinator stops on a hard failure, leaves completed shards intact, and leaves incomplete temporary files distinguishable from valid output.

Training refuses:

- incomplete manifests unless explicitly allowed for exploratory runs;
- mismatched dataset/checkpoint configuration hashes;
- shards with missing required datasets;
- empty modal masks;
- nonfinite valid targets.

## 12. Verification and Acceptance

Development follows red–green–refactor cycles. Acceptance requires fresh execution of all checks below.

### 12.1 Unit tests

- Every empirical relation matches hand-calculated scalar examples.
- Generated models satisfy bounds, family proportions within statistical tolerance, anomaly location rules, and deterministic reproducibility.
- Secular-function evaluations are finite on representative valid grids.
- Quadratic vertex calculations match analytic parabolas and reject degenerate triplets.
- Root deduplication, ordering, masks, and retry classification cover edge cases.

### 12.2 Reference tests

Golden fixtures will cover:

- the four-layer model from both papers;
- a monotonic 20-parameter model;
- a single low-velocity anomaly;
- the requested high-velocity plus multi-layer low-velocity coupled anomaly.

For roots present in both implementations, the Python result must match the reference result within `1e-5 km/s`. The quadratic strategy must not have more isolated internal modal gaps than the raw strategy on any golden fixture.

### 12.3 Pipeline tests

- Generate a deterministic 16-sample, two-shard dataset.
- Interrupt/resume simulation produces byte-equivalent scientific arrays.
- A one-epoch CPU training smoke test produces finite loss and a loadable checkpoint.
- Batched inference has the declared shape and finite values.
- Plot commands create nonempty image files.
- Package installation and all CLI help commands succeed in a fresh virtual environment without a C++ compiler.

### 12.4 Production readiness

The million-sample production command is not required to finish during the short acceptance run. Readiness is demonstrated by the same generation path at smoke scale, deterministic sharding, restart tests, and a documented command for the full run. Estimated storage and elapsed-time measurements from a benchmark shard will be printed before production confirmation.

## 13. Explicit Non-Goals

- Reimplementing the existing inversion code.
- Love-wave dispersion.
- Sensitivity kernels.
- Variable layer counts or trainable layer thicknesses.
- A web interface.
- Silently smoothing or interpolating physics outputs to make curves appear complete.
