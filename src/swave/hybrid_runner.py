"""Split-safe orchestration for sensitivity-weighted hybrid inversion."""

from __future__ import annotations

import json
import os
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from numpy.typing import NDArray

from .config import (
    DatasetConfig,
    HybridInversionConfig,
    NoiseScenario,
    canonical_hash,
    hybrid_inversion_identity_hash,
    load_dataset_config,
)
from .dataset import dataset_manifest_sha256, validate_dataset_files
from .hybrid_inversion import (
    HybridObjectiveTerms,
    HybridSurrogateObjective,
    LearningPrior,
    inverse_sensitivity_weights,
    mean_dimensionless_sensitivity,
)
from .hybrid_results import (
    HybridManifest,
    HybridResultBatch,
    HybridSplit,
    initialize_hybrid_manifest,
    load_hybrid_manifest,
    mark_hybrid_job_complete,
    validate_complete_hybrid_results,
    validate_hybrid_result_shard,
    write_hybrid_result_shard,
)
from .inversion import (
    DifferentiableSurrogate,
    InversionRun,
    ReferenceModel,
    SurrogateObjective,
    apply_observation_noise,
    build_reference_model,
    invert_one,
    regularization_matrix,
    resolve_inversion_device,
)
from .inversion_data import (
    InversionSample,
    iter_observation_samples,
    select_observation_samples_by_kind,
)
from .inversion_results import checkpoint_sha256, sample_id_sha256, software_sha256
from .splits import validate_checkpoint_split_policy
from .supervised_inversion import SupervisedEnsemblePredictor

HybridStage = Literal["tune", "test", "inversion", "all"]


@dataclass(frozen=True)
class HybridJob:
    """One stable split, noise, and source-shard hybrid work unit."""

    name: str
    split: HybridSplit
    noise: NoiseScenario
    samples: tuple[InversionSample, ...]


@dataclass(frozen=True)
class PreparedHybridSample:
    """Observation-only ingredients fixed before either optimizer is run."""

    observed: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    reference: ReferenceModel
    learning_prior: LearningPrior


@dataclass(frozen=True)
class HybridPairOutcome:
    """Paired global-bound control and learning-prior optimization outputs."""

    control: InversionRun
    hybrid: InversionRun
    hybrid_terms: HybridObjectiveTerms


@dataclass(frozen=True)
class _HybridInputs:
    dataset_config: DatasetConfig
    dataset_config_hash: str
    dataset_manifest_sha256: str
    forward_checkpoint_sha256: str
    surrogate: DifferentiableSurrogate
    supervised: SupervisedEnsemblePredictor


def build_hybrid_jobs(
    dataset_dir: Path | str,
    split: HybridSplit,
    noise_scenarios: tuple[NoiseScenario, ...],
) -> tuple[HybridJob, ...]:
    """Build deterministic hybrid jobs without exposing target profiles."""
    if split not in {"test", "inversion"}:
        raise ValueError("hybrid split must be test or inversion")
    if not noise_scenarios or len(set(noise_scenarios)) != len(noise_scenarios):
        raise ValueError("noise_scenarios must be nonempty and unique")
    grouped: dict[int, list[InversionSample]] = {}
    for sample in iter_observation_samples(dataset_dir, split):
        grouped.setdefault(sample.source_shard_id, []).append(sample)
    jobs: list[HybridJob] = []
    for noise in noise_scenarios:
        if noise not in {"clean", "noise_1pct"}:
            raise ValueError("noise scenario must be clean or noise_1pct")
        for shard_id, samples in sorted(grouped.items()):
            ordered = tuple(sorted(samples, key=lambda sample: sample.sample_id))
            if ordered:
                jobs.append(
                    HybridJob(
                        name=f"hybrid-{split}-{noise}-shard-{shard_id:05d}",
                        split=split,
                        noise=noise,
                        samples=ordered,
                    )
                )
    if not jobs:
        raise ValueError("selected hybrid split has no jobs")
    return tuple(jobs)


def assigned_hybrid_jobs(
    jobs: tuple[HybridJob, ...], *, task_index: int, task_count: int
) -> tuple[HybridJob, ...]:
    """Return one stable disjoint modulo partition of hybrid jobs."""
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count <= 0:
        raise TypeError("task_count must be a positive integer")
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise TypeError("task_index must be an integer")
    if not 0 <= task_index < task_count:
        raise ValueError("task_index must be in [0, task_count)")
    return tuple(
        job for index, job in enumerate(jobs) if index % task_count == task_index
    )


def choose_prior_lambda(
    errors_by_lambda: dict[float, NDArray[np.float64]],
) -> tuple[float, dict[float, float]]:
    """Choose the smallest lambda attaining the lowest finite mean error."""
    if not errors_by_lambda:
        raise ValueError("prior-lambda metrics must be nonempty")
    scores: dict[float, float] = {}
    for candidate, errors in errors_by_lambda.items():
        if not np.isfinite(candidate) or candidate <= 0:
            raise ValueError("prior-lambda candidates must be finite and positive")
        values = np.asarray(errors, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("prior-lambda errors must be nonempty finite vectors")
        if np.any(values < 0):
            raise ValueError("prior-lambda errors must be nonnegative")
        scores[float(candidate)] = float(values.mean())
    best_score = min(scores.values())
    selected = min(
        candidate
        for candidate, score in scores.items()
        if np.isclose(score, best_score, rtol=0.0, atol=1e-15)
    )
    return selected, scores


def _load_predictors(
    config: HybridInversionConfig,
) -> tuple[DifferentiableSurrogate, SupervisedEnsemblePredictor]:
    return (
        DifferentiableSurrogate.load(config.forward_checkpoint, config.device),
        SupervisedEnsemblePredictor.load(config.supervised_dir, config.device),
    )


def _validated_inputs(config: HybridInversionConfig) -> _HybridInputs:
    if not isinstance(config, HybridInversionConfig):
        raise TypeError("config must be a HybridInversionConfig")
    dataset_config = load_dataset_config(config.dataset_config)
    manifest = validate_dataset_files(config.dataset_dir)
    dataset_hash = canonical_hash(dataset_config)
    if manifest.config_hash != dataset_hash:
        raise ValueError("dataset configuration does not match the validated manifest")
    try:
        import torch

        forward_payload = torch.load(
            config.forward_checkpoint, map_location="cpu", weights_only=False
        )
    except OSError as error:
        raise ValueError("forward checkpoint is not readable") from error
    validate_checkpoint_split_policy(forward_payload)
    if forward_payload.get("dataset_config_hash") != dataset_hash:
        raise ValueError("forward checkpoint dataset identity does not match")
    manifest_digest = dataset_manifest_sha256(manifest)
    identity_path = config.supervised_dir / "run-identity.json"
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("supervised run identity is not readable") from error
    if identity.get("dataset_config_hash") != dataset_hash or identity.get(
        "dataset_manifest_sha256"
    ) != manifest_digest:
        raise ValueError("supervised ensemble dataset identity does not match")
    surrogate, supervised = _load_predictors(config)
    return _HybridInputs(
        dataset_config=dataset_config,
        dataset_config_hash=dataset_hash,
        dataset_manifest_sha256=manifest_digest,
        forward_checkpoint_sha256=checkpoint_sha256(config.forward_checkpoint),
        surrogate=surrogate,
        supervised=supervised,
    )


def _prepare_sample(
    sample: InversionSample,
    noise: NoiseScenario,
    config: HybridInversionConfig,
    dataset_config: DatasetConfig,
    surrogate: DifferentiableSurrogate,
    supervised: SupervisedEnsemblePredictor,
) -> PreparedHybridSample:
    observed = apply_observation_noise(
        sample.phase_velocity,
        sample.valid_mask,
        noise,
        config.seed,
        sample.sample_id,
    )
    frequencies = dataset_config.physics.frequencies
    reference = build_reference_model(
        frequencies,
        observed,
        sample.valid_mask,
        vs_min=config.vs_min,
        vs_max=config.vs_max,
        vs_width=config.reference_width,
    )
    supervised_vs = np.clip(
        supervised.predict(observed, sample.valid_mask), config.vs_min, config.vs_max
    )
    sensitivity = mean_dimensionless_sensitivity(
        surrogate,
        supervised_vs,
        sample.valid_mask,
        config.mode_weights,
        phase_floor=config.sensitivity_phase_floor,
    )
    weights = inverse_sensitivity_weights(
        sensitivity,
        epsilon_fraction=config.sensitivity_epsilon_fraction,
        minimum=config.prior_weight_min,
        maximum=config.prior_weight_max,
    )
    return PreparedHybridSample(
        observed=observed,
        valid_mask=np.asarray(sample.valid_mask, dtype=np.bool_).copy(),
        reference=reference,
        learning_prior=LearningPrior(supervised_vs, sensitivity, weights),
    )


def _objectives(
    prepared: PreparedHybridSample,
    prior_lambda: float,
    config: HybridInversionConfig,
    dataset_config: DatasetConfig,
    surrogate: DifferentiableSurrogate,
) -> tuple[SurrogateObjective, HybridSurrogateObjective]:
    regularization = regularization_matrix(
        prepared.reference.vs, config.regularization_type
    )
    common = {
        "surrogate": surrogate,
        "frequencies": dataset_config.physics.frequencies,
        "observed": prepared.observed,
        "valid_mask": prepared.valid_mask,
        "mode_weights": config.mode_weights,
        "reference": prepared.reference.vs,
        "regularization": regularization,
        "regularization_lambda": config.smoothness_lambda,
    }
    return (
        SurrogateObjective(**common),
        HybridSurrogateObjective(
            **common,
            learning_prior=prepared.learning_prior.vs,
            prior_weights=prepared.learning_prior.weights,
            prior_lambda=prior_lambda,
        ),
    )


def _global_bounds(
    config: HybridInversionConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.full(20, config.vs_min, dtype=np.float64),
        np.full(20, config.vs_max, dtype=np.float64),
    )


def _invert_prepared_hybrid(
    prepared: PreparedHybridSample,
    prior_lambda: float,
    config: HybridInversionConfig,
    *,
    dataset_config: DatasetConfig | None = None,
    surrogate: DifferentiableSurrogate | None = None,
) -> InversionRun:
    if dataset_config is None or surrogate is None:
        validated = _validated_inputs(config)
        dataset_config = validated.dataset_config
        surrogate = validated.surrogate
    _, objective = _objectives(
        prepared, prior_lambda, config, dataset_config, surrogate
    )
    lower, upper = _global_bounds(config)
    return invert_one(
        objective,
        prepared.reference.vs,
        lower,
        upper,
        max_iterations=config.max_iterations,
        relative_tolerance=config.relative_tolerance,
    )


def _invert_prepared_pair(
    prepared: PreparedHybridSample,
    prior_lambda: float,
    config: HybridInversionConfig,
    dataset_config: DatasetConfig,
    surrogate: DifferentiableSurrogate,
) -> HybridPairOutcome:
    control_objective, hybrid_objective = _objectives(
        prepared, prior_lambda, config, dataset_config, surrogate
    )
    lower, upper = _global_bounds(config)
    keywords = {
        "max_iterations": config.max_iterations,
        "relative_tolerance": config.relative_tolerance,
    }
    control = invert_one(
        control_objective, prepared.reference.vs, lower, upper, **keywords
    )
    hybrid = invert_one(
        hybrid_objective, prepared.reference.vs, lower, upper, **keywords
    )
    return HybridPairOutcome(
        control=control,
        hybrid=hybrid,
        hybrid_terms=hybrid_objective.detailed_terms(hybrid.vs),
    )


def _truth_vs(sample: InversionSample) -> NDArray[np.float64]:
    """Join target Vs only after one optimization result has been produced."""
    try:
        with h5py.File(sample.source_path, "r") as handle:
            sample_id = int(handle["sample_id"][sample.source_row])
            if sample_id != sample.sample_id:
                raise ValueError("target source row identity does not match")
            values = np.asarray(handle["vs"][sample.source_row], dtype=np.float64)
    except OSError as error:
        raise ValueError("target source shard is not readable") from error
    if values.shape != (20,) or not np.all(np.isfinite(values)):
        raise ValueError("target Vs profile is invalid")
    return values


def _tuning_identity(
    config: HybridInversionConfig,
    inputs: _HybridInputs,
    sample_ids: NDArray[np.uint64],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "split_policy": "mod100-v2-80-5-5-10",
        "usage": "validation_only_prior_lambda_selection",
        "dataset_config_hash": inputs.dataset_config_hash,
        "dataset_manifest_sha256": inputs.dataset_manifest_sha256,
        "forward_checkpoint_sha256": inputs.forward_checkpoint_sha256,
        "supervised_checkpoint_sha256": list(
            inputs.supervised.checkpoint_sha256
        ),
        "hybrid_config_hash": hybrid_inversion_identity_hash(config),
        "software_sha256": software_sha256(),
        "validation_sample_count": len(sample_ids),
        "validation_sample_id_sha256": sample_id_sha256(sample_ids),
        "validation_sample_ids": [int(value) for value in sample_ids],
        "prior_lambda_candidates": list(config.prior_lambda_candidates),
        "noise_scenarios": list(config.noise_scenarios),
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def select_prior_lambda(config: HybridInversionConfig) -> Path:
    """Select and persist prior strength using validation rows only."""
    inputs = _validated_inputs(config)
    samples = select_observation_samples_by_kind(
        config.dataset_dir,
        "validation",
        config.validation_samples_per_kind,
    )
    samples = sorted(samples, key=lambda sample: sample.sample_id)
    sample_ids = np.asarray([sample.sample_id for sample in samples], dtype=np.uint64)
    identity = _tuning_identity(config, inputs, sample_ids)
    output = config.output_dir / "tuning.json"
    if output.exists():
        stored = json.loads(output.read_text(encoding="utf-8"))
        for name, value in identity.items():
            if stored.get(name) != value:
                raise ValueError(f"hybrid tuning identity {name} does not match")
        return output
    errors_by_lambda: dict[float, list[float]] = {
        candidate: [] for candidate in config.prior_lambda_candidates
    }
    for sample in samples:
        truth: NDArray[np.float64] | None = None
        for noise in config.noise_scenarios:
            prepared = _prepare_sample(
                sample,
                noise,
                config,
                inputs.dataset_config,
                inputs.surrogate,
                inputs.supervised,
            )
            for candidate in config.prior_lambda_candidates:
                run = _invert_prepared_hybrid(
                    prepared,
                    candidate,
                    config,
                    dataset_config=inputs.dataset_config,
                    surrogate=inputs.surrogate,
                )
                if not run.success:
                    raise ArithmeticError(
                        "validation optimizer failed during prior-lambda selection"
                    )
                if truth is None:
                    truth = _truth_vs(sample)
                errors_by_lambda[candidate].append(float(np.abs(run.vs - truth).mean()))
    selected, scores = choose_prior_lambda(
        {key: np.asarray(value) for key, value in errors_by_lambda.items()}
    )
    report = {
        **identity,
        "candidate_mae_km_s": {
            str(candidate): scores[candidate]
            for candidate in config.prior_lambda_candidates
        },
        "selected_prior_lambda": selected,
        "selection_metric": "mean_vs_mae_across_validation_samples_layers_and_noise",
        "tie_break": "smallest_prior_lambda",
    }
    _atomic_json(output, report)
    return output


def _selected_prior_lambda(config: HybridInversionConfig) -> float:
    path = select_prior_lambda(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = float(payload["selected_prior_lambda"])
    if value not in config.prior_lambda_candidates:
        raise ValueError("selected prior lambda is not a configured candidate")
    return value


def _allocate_batch(job: HybridJob) -> dict[str, NDArray[np.generic]]:
    count = len(job.samples)
    return {
        "sample_id": np.asarray([s.sample_id for s in job.samples], dtype=np.uint64),
        "model_kind": np.asarray([s.model_kind for s in job.samples], dtype=np.uint8),
        "valid_mask": np.stack([s.valid_mask for s in job.samples]).astype(np.bool_),
        "observed_phase_velocity": np.full((count, 4, 120), np.nan, np.float32),
        "reference_vs": np.full((count, 20), np.nan, np.float32),
        "supervised_vs": np.full((count, 20), np.nan, np.float32),
        "sensitivity": np.full((count, 20), np.nan, np.float64),
        "prior_weights": np.full((count, 20), np.nan, np.float64),
        **{
            name: np.zeros(count, np.bool_)
            for name in ("control_success", "hybrid_success")
        },
        **{
            name: np.full(count, -1, np.int32)
            for name in ("control_status", "hybrid_status")
        },
        **{
            name: np.full(count, b"optimizer_failure", dtype="S64")
            for name in ("control_failure_code", "hybrid_failure_code")
        },
        **{
            name: np.zeros(count, np.int32)
            for name in (
                "control_iterations",
                "control_evaluations",
                "hybrid_iterations",
                "hybrid_evaluations",
            )
        },
        **{
            name: np.full(count, np.nan, np.float64)
            for name in (
                "control_initial_objective",
                "control_total",
                "control_data_misfit",
                "control_smoothness",
                "hybrid_initial_objective",
                "hybrid_total",
                "hybrid_data_misfit",
                "hybrid_smoothness",
                "hybrid_learning_prior",
            )
        },
        "control_vs": np.full((count, 20), np.nan, np.float32),
        "control_prediction": np.full((count, 4, 120), np.nan, np.float32),
        "hybrid_vs": np.full((count, 20), np.nan, np.float32),
        "hybrid_prediction": np.full((count, 4, 120), np.nan, np.float32),
    }


def _record_run(
    arrays: dict[str, NDArray[np.generic]],
    row: int,
    prefix: str,
    run: InversionRun,
) -> None:
    arrays[f"{prefix}_success"][row] = run.success
    arrays[f"{prefix}_status"][row] = run.status
    arrays[f"{prefix}_failure_code"][row] = b"" if run.success else b"optimizer_failure"
    arrays[f"{prefix}_iterations"][row] = run.iterations
    arrays[f"{prefix}_evaluations"][row] = run.evaluations
    arrays[f"{prefix}_initial_objective"][row] = run.initial_objective
    arrays[f"{prefix}_total"][row] = run.terms.total
    arrays[f"{prefix}_data_misfit"][row] = run.terms.data_misfit
    arrays[f"{prefix}_vs"][row] = run.vs.astype(np.float32)
    arrays[f"{prefix}_prediction"][row] = run.predicted_phase_velocity.astype(np.float32)


def run_hybrid_job(
    job: HybridJob,
    config: HybridInversionConfig,
    dataset_config: DatasetConfig,
    manifest: HybridManifest,
) -> HybridManifest:
    """Run and atomically publish one paired control/hybrid job."""
    surrogate, supervised = _load_predictors(config)
    arrays = _allocate_batch(job)
    for row, sample in enumerate(job.samples):
        prepared = _prepare_sample(
            sample, job.noise, config, dataset_config, surrogate, supervised
        )
        arrays["observed_phase_velocity"][row] = prepared.observed.astype(np.float32)
        arrays["reference_vs"][row] = prepared.reference.vs.astype(np.float32)
        arrays["supervised_vs"][row] = prepared.learning_prior.vs.astype(np.float32)
        arrays["sensitivity"][row] = prepared.learning_prior.sensitivity
        arrays["prior_weights"][row] = prepared.learning_prior.weights
        outcome = _invert_prepared_pair(
            prepared,
            manifest.selected_prior_lambda,
            config,
            dataset_config,
            surrogate,
        )
        _record_run(arrays, row, "control", outcome.control)
        arrays["control_smoothness"][row] = outcome.control.terms.regularization
        _record_run(arrays, row, "hybrid", outcome.hybrid)
        arrays["hybrid_data_misfit"][row] = outcome.hybrid_terms.data_misfit
        arrays["hybrid_smoothness"][row] = (
            outcome.hybrid_terms.smoothness_regularization
        )
        arrays["hybrid_learning_prior"][row] = (
            outcome.hybrid_terms.learning_prior_regularization
        )
    batch = HybridResultBatch(**arrays)
    path = write_hybrid_result_shard(
        config.output_dir / f"{job.name}.h5", batch, manifest, job.name
    )
    return mark_hybrid_job_complete(config.output_dir, job.name, path)


def _job_ids(job: HybridJob) -> NDArray[np.uint64]:
    return np.asarray([sample.sample_id for sample in job.samples], dtype=np.uint64)


def _run_job_process(
    job: HybridJob,
    config: HybridInversionConfig,
    dataset_config: DatasetConfig,
    manifest: HybridManifest,
) -> HybridManifest:
    return run_hybrid_job(job, config, dataset_config, manifest)


def run_hybrid_experiment(
    config: HybridInversionConfig, split: HybridSplit
) -> HybridManifest:
    """Validate, resume, and run one final hybrid experiment split."""
    inputs = _validated_inputs(config)
    selected = _selected_prior_lambda(config)
    jobs = build_hybrid_jobs(config.dataset_dir, split, config.noise_scenarios)
    expected = {job.name: _job_ids(job) for job in jobs}
    split_directory = config.output_dir / split
    split_config = replace(config, output_dir=split_directory)
    manifest = initialize_hybrid_manifest(
        split_directory,
        split=split,
        dataset_config_hash=inputs.dataset_config_hash,
        dataset_manifest_sha256=inputs.dataset_manifest_sha256,
        forward_checkpoint=config.forward_checkpoint,
        supervised_checkpoint_sha256=inputs.supervised.checkpoint_sha256,
        config=split_config,
        selected_prior_lambda=selected,
        expected_sample_ids_by_job=expected,
    )
    for job in jobs:
        path = split_directory / f"{job.name}.h5"
        if job.name in manifest.completed_jobs:
            validate_hybrid_result_shard(
                path, manifest=manifest, expected_sample_ids=expected[job.name]
            )
        elif path.exists():
            validate_hybrid_result_shard(
                path, manifest=manifest, expected_sample_ids=expected[job.name]
            )
            manifest = mark_hybrid_job_complete(split_directory, job.name, path)
    assigned = assigned_hybrid_jobs(
        jobs, task_index=config.task_index, task_count=config.task_count
    )
    pending = tuple(job for job in assigned if job.name not in manifest.completed_jobs)
    resolved_device = resolve_inversion_device(config.device).type
    if resolved_device == "cuda" and config.workers not in {0, 1}:
        raise ValueError("CUDA hybrid inversion requires one worker per task")
    workers = config.workers or (
        1 if resolved_device == "cuda" else max(1, min(len(pending), os.cpu_count() or 1))
    )
    worker_config = replace(
        split_config, device=resolved_device, workers=workers
    )
    if workers <= 1:
        for job in pending:
            run_hybrid_job(job, worker_config, inputs.dataset_config, manifest)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            iterator = iter(pending)
            in_flight: set[Future[HybridManifest]] = set()
            for _ in range(min(workers, len(pending))):
                job = next(iterator)
                in_flight.add(
                    executor.submit(
                        _run_job_process,
                        job,
                        worker_config,
                        inputs.dataset_config,
                        manifest,
                    )
                )
            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    in_flight.remove(future)
                    future.result()
                    try:
                        job = next(iterator)
                    except StopIteration:
                        continue
                    in_flight.add(
                        executor.submit(
                            _run_job_process,
                            job,
                            worker_config,
                            inputs.dataset_config,
                            manifest,
                        )
                    )
    refreshed = load_hybrid_manifest(split_directory)
    if refreshed.complete:
        return validate_complete_hybrid_results(split_directory)
    return refreshed
