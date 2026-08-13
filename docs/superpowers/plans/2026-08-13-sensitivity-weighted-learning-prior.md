# Sensitivity-Weighted Learning Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a split-safe supervised-ensemble prior to bounded L-BFGS-B, weight its twenty layer penalties by inverse mean dimensionless dispersion sensitivity, and document it as report sections 2.7 and 7.

**Architecture:** A public `SupervisedEnsemblePredictor` turns one observed four-mode dispersion record into a validated three-seed Vs prior. A focused `hybrid_inversion.py` module computes the forward-surrogate Jacobian, bounded inverse-sensitivity weights, and the augmented objective; `hybrid_runner.py` owns validation-only prior-strength selection and resumable HDF5 jobs without exposing target Vs to the optimizer. Hybrid artifacts live outside the existing schema-v3 inversion directory and a compact reporter joins truth only after result validation.

**Tech Stack:** Python 3.11+, NumPy, PyTorch 2.4+, SciPy L-BFGS-B, h5py, TOML, pytest, Ruff, XeLaTeX.

## Global Constraints

- Training IDs 0--79 and validation IDs 80--84 retain their existing roles; test IDs 85--89 and inversion IDs 90--99 never select checkpoints or `lambda_p`.
- `v_sup` enters only the quadratic learning-prior term: it is not an optimizer initial value, hard bound, or stopping criterion.
- The global-bound control and hybrid run both start from the observation-derived reference and use `[0.3, 2.6]` km/s bounds.
- Sensitivity is the valid-cell, mode-weighted mean of `abs((v_i / F_mf) * dF_mf/dv_i)` at `v_sup`.
- Prior weights are inverse sensitivities, have exact mean 1, and stay within `[0.25, 4.0]`.
- Existing `results/inversion` schema-v3 files and reported numbers are immutable.
- Missing supervised best checkpoints must produce an explicit error; no historical metrics may stand in for hybrid results.

---

### Task 1: Public, Identity-Checked Supervised Ensemble Prediction

**Files:**
- Modify: `src/swave/supervised_inversion.py`
- Test: `tests/test_supervised_inversion.py`

**Interfaces:**
- Consumes: `runs/supervised-inversion-v2/run-identity.json` and ordered `seed-<N>-best.pt` files.
- Produces: `SupervisedEnsemblePredictor.load(output_dir: Path | str, device: str = "auto") -> SupervisedEnsemblePredictor` and `predict(observed: ArrayLike, valid_mask: ArrayLike) -> NDArray[np.float64]`.

- [ ] **Step 1: Write failing predictor tests**

Add tests that create two tiny `InverseNet` checkpoints with a shared run identity and known output-head biases, then assert that `predict()` drops the first frequency, fills invalid cells from checkpoint train statistics, applies z-score normalization, averages physical predictions, and returns shape `(20,)`. Add separate failures for a missing best checkpoint, a checkpoint seed mismatch, unequal normalization arrays, non-finite valid observations, and an identity mismatch.

```python
predictor = SupervisedEnsemblePredictor.load(output_dir, device="cpu")
prediction = predictor.predict(observed, valid_mask)
assert prediction.shape == (20,)
np.testing.assert_allclose(prediction, expected_ensemble)
```

- [ ] **Step 2: Verify the tests fail for the missing interface**

Run: `pytest tests/test_supervised_inversion.py -k 'ensemble_predictor' -v`

Expected: collection or test failure because `SupervisedEnsemblePredictor` does not exist.

- [ ] **Step 3: Implement the predictor**

Add a frozen normalization loader and a predictor that:

```python
@dataclass
class SupervisedEnsemblePredictor:
    models: tuple[InverseNet, ...]
    normalization: SupervisedNormalization
    seeds: tuple[int, ...]
    device: torch.device
    checkpoint_sha256: tuple[str, ...]

    @classmethod
    def load(cls, output_dir: Path | str, device: str = "auto") -> "SupervisedEnsemblePredictor": ...

    def predict(self, observed: ArrayLike, valid_mask: ArrayLike) -> NDArray[np.float64]: ...
```

Load `run-identity.json` with duplicate-key rejection, require the exact ordered `seed_ensemble`, verify every checkpoint through `_validate_checkpoint_identity`, compare all five normalization fields exactly, hash every checkpoint, and use `torch.no_grad()` for inference. Validate `(4, 120)` shapes, require all masked-in cells finite, fill only masked-out cells after dropping column zero, and reject a non-finite physical prediction.

- [ ] **Step 4: Run supervised tests**

Run: `pytest tests/test_supervised_inversion.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit the predictor**

```bash
git add src/swave/supervised_inversion.py tests/test_supervised_inversion.py
git commit -m "feat: load supervised ensemble for hybrid inversion"
```

### Task 2: Sensitivity Weights and Hybrid Objective

**Files:**
- Create: `src/swave/hybrid_inversion.py`
- Create: `tests/test_hybrid_inversion.py`

**Interfaces:**
- Consumes: `DifferentiableSurrogate`, `SurrogateObjective`, and `SupervisedEnsemblePredictor`.
- Produces: `mean_dimensionless_sensitivity(...)`, `inverse_sensitivity_weights(...)`, `LearningPrior`, `HybridObjectiveTerms`, and `HybridSurrogateObjective`.

- [ ] **Step 1: Write analytic sensitivity and weighting tests**

Use a fixed linear four-mode forward map. Compare the returned layer sensitivity with an independently calculated Jacobian, then mask one cell and zero one mode weight to prove only active cells contribute. Assert lower sensitivity gives higher prior weight, all weights lie in `[0.25, 4.0]`, and `weights.mean() == 1` within `1e-12`. Add failures for all-zero sensitivity, non-finite arrays, invalid bounds, and a zero predicted phase velocity in an active cell.

```python
sensitivity = mean_dimensionless_sensitivity(
    surrogate, prior, mask, (4.0, 1.0, 1.0, 1.0), phase_floor=1e-12
)
weights = inverse_sensitivity_weights(
    sensitivity, epsilon_fraction=1e-2, minimum=0.25, maximum=4.0
)
assert weights[np.argmin(sensitivity)] > weights[np.argmax(sensitivity)]
np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-12)
```

- [ ] **Step 2: Run the core tests and confirm failure**

Run: `pytest tests/test_hybrid_inversion.py -v`

Expected: collection fails because `swave.hybrid_inversion` is absent.

- [ ] **Step 3: Implement sensitivity and exact bounded normalization**

Use `torch.func.jacfwd(surrogate.predict_tensor)` so the cost scales with twenty model inputs instead of 480 output cells. Form the dimensionless absolute Jacobian, apply the observation mask and modal weights, and average over active modal-frequency cells. Compute `epsilon_s = epsilon_fraction * sensitivity.mean()`.

Find the scale `c` by bisection on

```python
def total(scale: float) -> float:
    return np.clip(scale / (sensitivity + epsilon_s), minimum, maximum).sum()
```

until the sum is `20.0`; reject bounds unless `0 < minimum <= 1 <= maximum`.

- [ ] **Step 4: Write failing objective-value and gradient tests**

Construct a `HybridSurrogateObjective` with known prior and weights. Verify its detailed terms against a direct NumPy calculation and compare `value_and_grad` with central finite differences. Assert `prior_lambda=0` matches the base global-bound control objective exactly.

```python
details = objective.detailed_terms(values)
expected_prior = prior_lambda / 20.0 * np.sum(weights * (values - prior) ** 2)
assert details.learning_prior_regularization == pytest.approx(expected_prior)
assert details.total == pytest.approx(
    details.data_misfit
    + details.smoothness_regularization
    + details.learning_prior_regularization
)
```

- [ ] **Step 5: Implement the augmented objective**

Add validated dataclasses:

```python
@dataclass(frozen=True)
class LearningPrior:
    vs: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    weights: NDArray[np.float64]

@dataclass(frozen=True)
class HybridObjectiveTerms:
    total: float
    data_misfit: float
    smoothness_regularization: float
    learning_prior_regularization: float

@dataclass
class HybridSurrogateObjective(SurrogateObjective):
    learning_prior: ArrayLike
    prior_weights: ArrayLike
    prior_lambda: float
```

Override `_calculate()` to add the quadratic prior while returning the combined regularization through the existing `ObjectiveTerms` interface used by `invert_one()`. Add `detailed_terms()` for separate persistence.

- [ ] **Step 6: Run the core tests**

Run: `pytest tests/test_hybrid_inversion.py tests/test_inversion.py -v`

Expected: all tests pass and existing inversion behavior is unchanged.

- [ ] **Step 7: Commit the scientific core**

```bash
git add src/swave/hybrid_inversion.py tests/test_hybrid_inversion.py
git commit -m "feat: add sensitivity-weighted learning prior objective"
```

### Task 3: Hybrid Configuration and Split-Safe Sample Access

**Files:**
- Modify: `src/swave/config.py`
- Modify: `src/swave/inversion_data.py`
- Create: `configs/hybrid-inversion.toml`
- Modify: `tests/test_config.py`
- Modify: `tests/test_inversion_data.py`

**Interfaces:**
- Consumes: dataset, forward checkpoint, supervised output directory, sensitivity controls, optimizer controls, and cluster execution controls.
- Produces: `HybridInversionConfig`, `load_hybrid_inversion_config()`, `hybrid_inversion_identity_hash()`, `iter_observation_samples(dataset_dir, split)`, and `select_observation_samples_by_kind(dataset_dir, split, per_kind)`.

- [ ] **Step 1: Write failing configuration tests**

Test TOML tuple conversion and validation for ordered positive `prior_lambda_candidates`, `epsilon_fraction > 0`, `0 < prior_weight_min <= 1 <= prior_weight_max`, global Vs bounds exactly inside `[0.3, 2.6]`, positive validation sample count, two unique supported noise scenarios, device/worker/task controls, and unknown keys. Verify execution-only fields are excluded from `hybrid_inversion_identity_hash()` while all scientific fields and checkpoint paths affect it.

- [ ] **Step 2: Run configuration tests and confirm failure**

Run: `pytest tests/test_config.py -k 'hybrid' -v`

Expected: failure because `HybridInversionConfig` is missing.

- [ ] **Step 3: Implement the configuration**

Add defaults matching the approved design:

```python
prior_lambda_candidates = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
sensitivity_epsilon_fraction = 1e-2
prior_weight_min = 0.25
prior_weight_max = 4.0
validation_samples_per_kind = 100
mode_weights = (4.0, 1.0, 1.0, 1.0)
smoothness_lambda = 1e-2
vs_min = 0.3
vs_max = 2.6
```

Keep `reference_width=0.7` only for construction of `v_ref`; hybrid/control optimization bounds are the global `vs_min` and `vs_max` vectors.

- [ ] **Step 4: Generalize observation-only split access under tests**

Rename the internal iterator boundary without breaking existing callers:

```python
def iter_observation_samples(dataset_dir: Path | str, split: Split) -> Iterator[InversionSample]: ...
def iter_inversion_samples(dataset_dir: Path | str) -> Iterator[InversionSample]:
    return iter_observation_samples(dataset_dir, "inversion")
def select_observation_samples_by_kind(dataset_dir, split, per_kind) -> list[InversionSample]: ...
```

Tests must prove validation, test, and inversion IDs are disjoint and that returned objects do not expose `vs`.

- [ ] **Step 5: Run configuration and data-access tests**

Run: `pytest tests/test_config.py tests/test_inversion_data.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit configuration and access boundaries**

```bash
git add src/swave/config.py src/swave/inversion_data.py configs/hybrid-inversion.toml tests/test_config.py tests/test_inversion_data.py
git commit -m "feat: configure split-safe hybrid inversion"
```

### Task 4: Validation-Only Lambda Selection and Resumable Hybrid Jobs

**Files:**
- Create: `src/swave/hybrid_results.py`
- Create: `src/swave/hybrid_runner.py`
- Create: `tests/test_hybrid_results.py`
- Create: `tests/test_hybrid_runner.py`

**Interfaces:**
- Consumes: `HybridInversionConfig`, observation-only samples, the two predictors, and Task 2 objective functions.
- Produces: `select_prior_lambda(config) -> Path`, `run_hybrid_experiment(config, split: Literal["test", "inversion"]) -> HybridManifest`, atomic result shards, and strict validation functions.

- [ ] **Step 1: Write failing tuning-isolation tests**

Monkeypatch optimizer results for a deterministic four-family validation sample set. Assert every candidate runs both noise scenarios, the score is the mean Vs MAE over samples/layers/scenarios, a tie chooses the smaller lambda, and no test/inversion row is opened. Assert `tuning.json` binds validation sample IDs, candidate metrics, selected lambda, data/checkpoint/software identities, and refuses incompatible reuse.

- [ ] **Step 2: Write failing result-schema tests**

Define a fixture with control and hybrid rows. Require arrays for sample identity, diagnostics, reference/prior/control/hybrid profiles, sensitivity, weights, observations, both reconstructions, masks, iterations/evaluations, and separate objective components. Assert successful rows are finite and globally bounded, failed rows publish canonical all-NaN scientific outputs, weight mean/bounds hold, and

```python
hybrid_total == data_misfit + smoothness_regularization + learning_prior_regularization
control_total == control_data_misfit + control_smoothness_regularization
```

Reject content checksum, sample-ID digest, manifest identity, source-code digest, or checkpoint digest mismatches.

- [ ] **Step 3: Implement manifest and HDF5 validation**

Create schema version 1 with an immutable manifest and atomic shard publication. Use one job per source shard and noise scenario, stable names `hybrid-<split>-<noise>-shard-NNNNN`, fixed-width failure codes, SHA-256 for each shard, and the existing lock/fsync pattern. Keep this schema in `results/hybrid-inversion`; never import or mutate `ResultBatch` schema-v3 files.

- [ ] **Step 4: Implement validation-only lambda selection**

For each candidate and sample: construct noisy observations, `v_ref`, `v_sup`, sensitivity, weights, and `HybridSurrogateObjective`; call `invert_one()` from `v_ref` with twenty-element global lower/upper vectors. Only after the recovered model is complete, read the validation target using the sample's immutable source path/row and update the score. Atomically publish strict JSON with `allow_nan=False`.

- [ ] **Step 5: Implement control and hybrid job execution**

Load the forward surrogate and supervised ensemble once per worker. For each sample compute the common reference/prior/sensitivity/weights, then run:

```python
control = SurrogateObjective(...)
hybrid = HybridSurrogateObjective(..., prior_lambda=selected_lambda)
lower = np.full(20, config.vs_min)
upper = np.full(20, config.vs_max)
control_run = invert_one(control, reference.vs, lower, upper, ...)
hybrid_run = invert_one(hybrid, reference.vs, lower, upper, ...)
```

Persist both outcomes independently so one optimizer failure does not erase the other. Support stable modulo cluster assignment, bounded process submission, child thread limits, completed-shard recovery, and final complete-manifest validation.

- [ ] **Step 6: Run hybrid runner and schema tests**

Run: `pytest tests/test_hybrid_results.py tests/test_hybrid_runner.py -v`

Expected: all tests pass, including interrupted publication and incompatible-resume cases.

- [ ] **Step 7: Commit the production runner**

```bash
git add src/swave/hybrid_results.py src/swave/hybrid_runner.py tests/test_hybrid_results.py tests/test_hybrid_runner.py
git commit -m "feat: run resumable hybrid inversion experiments"
```

### Task 5: CLI, Metrics, and Repository Documentation

**Files:**
- Modify: `src/swave/cli.py`
- Create: `src/swave/hybrid_report.py`
- Create: `tests/test_hybrid_report.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: tuning JSON and complete test/inversion hybrid manifests.
- Produces: `swave hybrid-invert`, `swave hybrid-report`, `summary.json`, a per-depth sensitivity/weight figure, and documented production commands.

- [ ] **Step 1: Write failing CLI tests**

Require help and lazy imports for both commands. Verify `hybrid-invert --stage tune|test|inversion|all` applies config/path/device/worker/task overrides and emits strict JSON or the tuning path. Verify `hybrid-report` passes result, dataset, baseline-summary, supervised-evaluation, and output paths without importing Matplotlib at CLI import time.

- [ ] **Step 2: Implement CLI handlers and parser entries**

Add lazy handlers `_hybrid_invert()` and `_hybrid_report()`. The `all` stage must tune first, then run test and inversion using the immutable selected lambda. Default paths come from `configs/hybrid-inversion.toml` and `results/hybrid-inversion-report`.

- [ ] **Step 3: Write failing metric/report tests**

Build tiny validated hybrid shards and truth HDF5. Assert overall, by-family, by-noise, and per-layer MAE/RMSE/P95 for control and hybrid; optimization effort; data/smoothness/prior term summaries; and per-layer mean sensitivity/weight. Require same-sample joins by ID and reject incomplete results or mismatched baseline/supervised comparison populations.

- [ ] **Step 4: Implement the reporter**

Read targets only after complete result validation, compute strict JSON metrics, and create `sensitivity-and-prior-weight-by-depth.png` with depth increasing downward and separate axes for dimensionless sensitivity and mean weight. When baseline or supervised artifacts are absent, omit their numeric comparison with an explicit machine-readable availability reason.

- [ ] **Step 5: Document exact workflow**

Add README commands:

```bash
swave hybrid-invert --config configs/hybrid-inversion.toml --stage tune
swave hybrid-invert --config configs/hybrid-inversion.toml --stage test
swave hybrid-invert --config configs/hybrid-inversion.toml --stage inversion
swave hybrid-report --results-dir results/hybrid-inversion --dataset-dir data/production
```

Explain inverse sensitivity, global-bound control, the checkpoint prerequisite, fold isolation, and the independent output directory.

- [ ] **Step 6: Run CLI and report tests**

Run: `pytest tests/test_cli.py tests/test_hybrid_report.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit CLI and repository docs**

```bash
git add src/swave/cli.py src/swave/hybrid_report.py tests/test_cli.py tests/test_hybrid_report.py README.md
git commit -m "feat: expose and report hybrid inversion workflow"
```

### Task 6: Add Report Sections 2.7 and 7

**Files:**
- Modify: `/home/jichi/zuoye/result/report.tex`

**Interfaces:**
- Consumes: the approved design, implemented CLI/config names, and production hybrid summary when available.
- Produces: method subsection 2.7 and main report section 7, with later sections automatically renumbered.

- [ ] **Step 1: Copy the report to a writable staging path and patch it**

Copy `/home/jichi/zuoye/result/report.tex` to `/tmp/report-hybrid.tex`, then use `apply_patch` on the staging file. Insert `\subsection{灵敏度加权学习先验的混合反演}` after the supervised method subsection. Include the exact sensitivity, inverse-weight, and augmented-objective equations from the design.

- [ ] **Step 2: Add the seventh main section**

Insert `\section{灵敏度加权学习先验的混合反演实验}\label{sec:hybrid}` immediately before the current discussion. Describe validation-only lambda selection, the global-bound control, expected result tables/figure, production commands, and the missing `seed-<N>-best.pt` prerequisite. Until compatible production artifacts exist, label the numerical result as not run and do not alter the abstract, discussion, or conclusion with inferred claims.

- [ ] **Step 3: Publish the staged report source with approval**

After comparing the staged file to the original, copy it back to `/home/jichi/zuoye/result/report.tex` using the required filesystem approval because that report directory is outside the repository writable root.

- [ ] **Step 4: Compile and inspect the report**

Run from `/home/jichi/zuoye/result`:

```bash
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

Expected: exit 0, resolved section references, no missing figures, and a PDF whose table of contents shows 2.7 and 7 before discussion.

### Task 7: Full Verification and Handoff

**Files:**
- Modify only files required by fixes found during verification.

**Interfaces:**
- Consumes: all previous task outputs.
- Produces: evidence-backed completion status and a production command that fails clearly until supervised best checkpoints are restored.

- [ ] **Step 1: Run focused scientific and workflow tests**

Run: `pytest tests/test_supervised_inversion.py tests/test_hybrid_inversion.py tests/test_hybrid_results.py tests/test_hybrid_runner.py tests/test_hybrid_report.py tests/test_cli.py -v`

Expected: all pass.

- [ ] **Step 2: Run the complete test suite**

Run: `pytest -q`

Expected: all pass.

- [ ] **Step 3: Run static and whitespace checks**

Run: `ruff check .`

Run: `git diff --check`

Expected: both exit 0.

- [ ] **Step 4: Run a missing-checkpoint smoke test**

Run: `swave hybrid-invert --config configs/hybrid-inversion.toml --stage tune`

Expected in the present environment: exit 2 with a specific missing `seed-0-best.pt` error before creating scientific result shards.

- [ ] **Step 5: Review tracked and untracked changes**

Run: `git status --short`

Confirm only intended source/tests/config/docs commits are tracked and preserve the pre-existing untracked `result/` directory.

- [ ] **Step 6: Invoke completion verification and report the checkpoint blocker accurately**

Use `superpowers:verification-before-completion`, cite the exact test/Ruff/LaTeX outputs, link the changed files, and provide the command to resume production once all three best checkpoints are placed in the configured supervised directory.
