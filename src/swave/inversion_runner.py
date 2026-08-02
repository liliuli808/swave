"""Leakage-free orchestration for full and deep inversion experiments."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .config import (
    DatasetConfig,
    InversionConfig,
    NoiseScenario,
    canonical_hash,
    load_dataset_config,
)
from .dataset import validate_dataset_files
from .inversion import (
    DifferentiableSurrogate,
    SurrogateObjective,
    apply_observation_noise,
    build_reference_model,
    generate_initial_models,
    invert_ensemble,
    invert_one,
    regularization_matrix,
    resolve_inversion_device,
)
from .inversion_data import (
    InversionSample,
    samples_by_source_shard,
    select_deep_samples,
)
from .inversion_results import (
    ResultBatch,
    ResultManifest,
    initialize_result_manifest,
    mark_job_complete,
    validate_complete_results,
    validate_result_shard,
    write_result_shard,
)
from .secular import LayeredModel
from .solver import DispersionSolver
from .splits import validate_checkpoint_split_policy

Experiment = Literal["full", "deep", "both"]
JobExperiment = Literal["full", "deep"]


@dataclass(frozen=True)
class InversionJob:
    """One stable experiment/noise/source-shard unit of work."""

    name: str
    experiment: JobExperiment
    noise: NoiseScenario
    samples: tuple[InversionSample, ...]


def _group_selected_samples(
    samples: list[InversionSample],
) -> dict[int, tuple[InversionSample, ...]]:
    grouped: dict[int, list[InversionSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.source_shard_id, []).append(sample)
    return {
        shard_id: tuple(sorted(rows, key=lambda sample: sample.sample_id))
        for shard_id, rows in sorted(grouped.items())
    }


def build_jobs(
    dataset_dir: Path | str,
    experiment: Experiment,
    noise_scenarios: tuple[NoiseScenario, ...],
    *,
    samples_per_kind: int,
) -> tuple[InversionJob, ...]:
    """Build deterministic jobs without reading inversion target profiles."""
    if experiment not in {"full", "deep", "both"}:
        raise ValueError("experiment must be full, deep, or both")
    if not noise_scenarios or len(set(noise_scenarios)) != len(noise_scenarios):
        raise ValueError("noise_scenarios must be nonempty and unique")
    if any(noise not in {"clean", "noise_1pct"} for noise in noise_scenarios):
        raise ValueError("noise scenario must be clean or noise_1pct")

    jobs: list[InversionJob] = []
    experiments: tuple[JobExperiment, ...] = (
        ("full", "deep") if experiment == "both" else (experiment,)
    )
    for current in experiments:
        if current == "full":
            grouped = {
                shard_id: tuple(samples)
                for shard_id, samples in samples_by_source_shard(dataset_dir).items()
            }
        else:
            grouped = _group_selected_samples(
                select_deep_samples(dataset_dir, samples_per_kind)
            )
        for noise in noise_scenarios:
            for shard_id, samples in grouped.items():
                if samples:
                    jobs.append(
                        InversionJob(
                            name=(f"{current}-{noise}-shard-{shard_id:05d}"),
                            experiment=current,
                            noise=noise,
                            samples=samples,
                        )
                    )
    if not jobs:
        raise ValueError("the selected experiment has no inversion jobs")
    return tuple(jobs)


def assigned_jobs(
    jobs: tuple[InversionJob, ...], *, task_index: int, task_count: int
) -> tuple[InversionJob, ...]:
    """Return the stable modulo partition assigned to one cluster task."""
    if isinstance(task_count, bool) or not isinstance(task_count, int):
        raise TypeError("task_count must be a positive integer")
    if task_count <= 0:
        raise ValueError("task_count must be a positive integer")
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise TypeError("task_index must be an integer")
    if not 0 <= task_index < task_count:
        raise ValueError("task_index must be in [0, task_count)")
    return tuple(
        job for index, job in enumerate(jobs) if index % task_count == task_index
    )


def _failure_code(error: Exception) -> bytes:
    text = str(error).lower()
    if isinstance(error, ArithmeticError) and (
        "objective" in text or "gradient" in text or "non-finite" in text
    ):
        return b"nonfinite_objective"
    return b"optimizer_failure"


def run_inversion_job(
    job: InversionJob,
    config: InversionConfig,
    dataset_config: DatasetConfig,
    manifest: ResultManifest,
) -> ResultManifest:
    """Run, atomically publish, and complete one observation-only job."""
    surrogate = DifferentiableSurrogate.load(config.checkpoint, config.device)
    count = len(job.samples)
    if count == 0:
        raise ValueError("inversion jobs must contain at least one sample")

    sample_id = np.asarray(
        [sample.sample_id for sample in job.samples], dtype=np.uint64
    )
    model_kind = np.asarray(
        [sample.model_kind for sample in job.samples], dtype=np.uint8
    )
    success = np.zeros(count, dtype=np.bool_)
    status = np.full(count, -1, dtype=np.int32)
    iterations = np.zeros(count, dtype=np.int32)
    evaluations = np.zeros(count, dtype=np.int32)
    initial_objective = np.full(count, np.nan, dtype=np.float64)
    final_objective = np.full(count, np.nan, dtype=np.float64)
    data_misfit = np.full(count, np.nan, dtype=np.float64)
    regularization = np.full(count, np.nan, dtype=np.float64)
    reference_vs = np.full((count, 20), np.nan, dtype=np.float32)
    inverted_vs = np.full((count, 20), np.nan, dtype=np.float32)
    observed = np.full((count, 4, 120), np.nan, dtype=np.float32)
    predicted = np.full((count, 4, 120), np.nan, dtype=np.float32)
    valid_mask = np.stack(
        [np.asarray(sample.valid_mask, dtype=np.bool_) for sample in job.samples]
    )
    failure_code = np.full(count, b"optimizer_failure", dtype="S64")
    ensemble_vs = None
    ensemble_success = None
    ensemble_status = None
    ensemble_objective = None
    ensemble_inlier_mask = None
    median_vs = None
    p10_vs = None
    p90_vs = None
    physical_phase_velocity = None
    physical_valid_mask = None
    if job.experiment == "deep":
        ensemble_shape = (count, config.initial_models)
        ensemble_vs = np.full((*ensemble_shape, 20), np.nan, dtype=np.float32)
        ensemble_success = np.zeros(ensemble_shape, dtype=np.bool_)
        ensemble_status = np.full(ensemble_shape, -1, dtype=np.int32)
        ensemble_objective = np.full(ensemble_shape, np.nan, dtype=np.float64)
        ensemble_inlier_mask = np.zeros(ensemble_shape, dtype=np.bool_)
        median_vs = np.full((count, 20), np.nan, dtype=np.float32)
        p10_vs = np.full((count, 20), np.nan, dtype=np.float32)
        p90_vs = np.full((count, 20), np.nan, dtype=np.float32)
        physical_phase_velocity = np.full((count, 4, 120), np.nan, dtype=np.float32)
        physical_valid_mask = np.zeros((count, 4, 120), dtype=np.bool_)

    frequencies = dataset_config.physics.frequencies
    if frequencies.shape != (120,):
        raise ValueError("inversion result schema requires exactly 120 frequencies")
    for row, sample in enumerate(job.samples):
        noisy = apply_observation_noise(
            sample.phase_velocity,
            sample.valid_mask,
            job.noise,
            config.seed,
            sample.sample_id,
        )
        observed[row] = noisy.astype(np.float32)
        fundamental = sample.valid_mask[0] & np.isfinite(noisy[0])
        if np.count_nonzero(fundamental) < 2:
            failure_code[row] = b"insufficient_fundamental_data"
            continue
        try:
            reference = build_reference_model(
                frequencies,
                noisy,
                sample.valid_mask,
                vs_min=config.vs_min,
                vs_max=config.vs_max,
                vs_width=config.vs_width,
            )
            reference_vs[row] = reference.vs.astype(np.float32)
            objective = SurrogateObjective(
                surrogate=surrogate,
                frequencies=frequencies,
                observed=noisy,
                valid_mask=sample.valid_mask,
                mode_weights=config.mode_weights,
                reference=reference.vs,
                regularization=regularization_matrix(
                    reference.vs, config.regularization_type
                ),
                regularization_lambda=config.regularization_lambda,
            )
            if job.experiment == "full":
                result = invert_one(
                    objective,
                    reference.vs,
                    reference.lower,
                    reference.upper,
                    max_iterations=config.max_iterations,
                    relative_tolerance=config.relative_tolerance,
                )
            else:
                starts = generate_initial_models(
                    reference,
                    config.initial_models,
                    config.seed,
                    sample.sample_id,
                    job.noise,
                )
                result = invert_ensemble(
                    objective,
                    starts,
                    reference.lower,
                    reference.upper,
                    max_iterations=config.max_iterations,
                    relative_tolerance=config.relative_tolerance,
                    minimum_valid_solutions=config.minimum_valid_solutions,
                )
        except (ArithmeticError, RuntimeError, ValueError) as error:
            failure_code[row] = _failure_code(error)
            continue

        if job.experiment == "full":
            run = result
            status[row] = run.status
            iterations[row] = run.iterations
            evaluations[row] = run.evaluations
            initial_objective[row] = run.initial_objective
            final_objective[row] = run.terms.total
            data_misfit[row] = run.terms.data_misfit
            regularization[row] = run.terms.regularization
            inverted_vs[row] = run.vs.astype(np.float32)
            predicted[row] = run.predicted_phase_velocity.astype(np.float32)
            if run.success:
                success[row] = True
                failure_code[row] = b""
            else:
                failure_code[row] = b"optimizer_failure"
            continue

        ensemble = result
        assert ensemble_vs is not None
        assert ensemble_success is not None
        assert ensemble_status is not None
        assert ensemble_objective is not None
        assert ensemble_inlier_mask is not None
        for start_index, run in enumerate(ensemble.runs):
            ensemble_vs[row, start_index] = run.vs.astype(np.float32)
            ensemble_success[row, start_index] = run.success
            ensemble_status[row, start_index] = run.status
            ensemble_objective[row, start_index] = run.terms.total
        ensemble_inlier_mask[row] = ensemble.inlier_mask
        iterations[row] = sum(run.iterations for run in ensemble.runs)
        evaluations[row] = sum(run.evaluations for run in ensemble.runs)
        if not ensemble.sufficient:
            failure_code[row] = b"insufficient_valid_solutions"
            continue

        assert median_vs is not None
        assert p10_vs is not None
        assert p90_vs is not None
        assert physical_phase_velocity is not None
        assert physical_valid_mask is not None
        median_vs[row] = ensemble.median_vs.astype(np.float32)
        p10_vs[row] = ensemble.p10_vs.astype(np.float32)
        p90_vs[row] = ensemble.p90_vs.astype(np.float32)
        inverted_vs[row] = ensemble.median_vs.astype(np.float32)
        predicted[row] = ensemble.representative_prediction.astype(np.float32)
        initial_objective[row] = ensemble.runs[0].initial_objective
        final_objective[row] = ensemble.representative_terms.total
        data_misfit[row] = ensemble.representative_terms.data_misfit
        regularization[row] = ensemble.representative_terms.regularization
        status[row] = 0
        try:
            model = LayeredModel.from_vs(
                ensemble.median_vs,
                dataset_config.geology.empirical_method,
                dataset_config.geology.thickness_km,
            )
            physical = DispersionSolver(model, dataset_config.physics).solve_grid(
                frequencies=frequencies,
                strategy="quadratic",
            )
            physical_values = np.asarray(physical.phase_velocity, dtype=np.float64)
            physical_mask = np.asarray(physical.valid_mask, dtype=np.bool_)
            if physical_values.shape != (4, 120) or physical_mask.shape != (4, 120):
                raise ValueError("physical reconstruction has an invalid shape")
            if not np.any(physical_mask):
                raise ArithmeticError("physical reconstruction has no valid cells")
            if not np.all(np.isfinite(physical_values[physical_mask])) or np.any(
                physical_values[physical_mask] <= 0
            ):
                raise ArithmeticError("physical reconstruction is non-finite")
            normalized_physical = np.full((4, 120), np.nan, dtype=np.float32)
            normalized_physical[physical_mask] = physical_values[physical_mask].astype(
                np.float32
            )
            physical_phase_velocity[row] = normalized_physical
            physical_valid_mask[row] = physical_mask
        except Exception:  # noqa: BLE001 - one sample must not abort its job
            failure_code[row] = b"physical_solver_failure"
            continue
        success[row] = True
        failure_code[row] = b""

    batch = ResultBatch(
        sample_id=sample_id,
        model_kind=model_kind,
        success=success,
        status=status,
        iterations=iterations,
        evaluations=evaluations,
        initial_objective=initial_objective,
        final_objective=final_objective,
        data_misfit=data_misfit,
        regularization=regularization,
        reference_vs=reference_vs,
        inverted_vs=inverted_vs,
        observed_phase_velocity=observed,
        surrogate_phase_velocity=predicted,
        valid_mask=valid_mask,
        failure_code=failure_code,
        ensemble_vs=ensemble_vs,
        ensemble_success=ensemble_success,
        ensemble_status=ensemble_status,
        ensemble_objective=ensemble_objective,
        ensemble_inlier_mask=ensemble_inlier_mask,
        median_vs=median_vs,
        p10_vs=p10_vs,
        p90_vs=p90_vs,
        physical_phase_velocity=physical_phase_velocity,
        physical_valid_mask=physical_valid_mask,
    )
    path = write_result_shard(config.output_dir, job.name, batch, manifest)
    return mark_job_complete(config.output_dir, job.name, path)


def _resolve_inversion_workers(device_type: str, requested: int) -> int:
    if device_type != "cpu":
        if requested not in {0, 1}:
            raise ValueError("CUDA inversion requires workers == 1 or workers == 0")
        return 1
    return requested or (os.cpu_count() or 1)


def _validate_experiment_inputs(
    config: InversionConfig,
) -> tuple[DatasetConfig, str, str, int]:
    dataset_config = load_dataset_config(config.dataset_config)
    dataset_manifest = validate_dataset_files(config.dataset_dir)
    dataset_hash = canonical_hash(dataset_config)
    if dataset_hash != dataset_manifest.config_hash:
        raise ValueError("dataset configuration hash does not match the dataset")

    try:
        payload = torch.load(
            config.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError) as error:
        raise ValueError("checkpoint is not readable") from error
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    validate_checkpoint_split_policy(payload)
    if payload.get("dataset_config_hash") != dataset_hash:
        raise ValueError("checkpoint dataset hash does not match the dataset")

    device = resolve_inversion_device(config.device)
    worker_count = _resolve_inversion_workers(device.type, config.workers)
    return dataset_config, dataset_hash, device.type, worker_count


def _job_sample_ids(job: InversionJob) -> np.ndarray:
    return np.asarray([sample.sample_id for sample in job.samples], dtype=np.uint64)


def _recover_published_jobs(
    config: InversionConfig,
    jobs: tuple[InversionJob, ...],
    manifest: ResultManifest,
) -> ResultManifest:
    current = manifest
    for job in jobs:
        path = config.output_dir / f"{job.name}.h5"
        expected_ids = _job_sample_ids(job)
        if job.name in current.completed_jobs:
            validate_result_shard(
                path,
                expected_sample_ids=expected_ids,
                manifest=current,
                expected_sha256=current.job_sha256[job.name],
            )
            continue
        if path.exists() or path.is_symlink():
            validate_result_shard(
                path,
                expected_sample_ids=expected_ids,
                manifest=current,
            )
            current = mark_job_complete(config.output_dir, job.name, path)
    return current


def run_inversion_experiment(
    config: InversionConfig,
    experiment: Experiment,
) -> ResultManifest:
    """Validate, resume, and execute one deterministic experiment assignment."""
    if not isinstance(config, InversionConfig):
        raise TypeError("config must be an InversionConfig")
    if experiment not in {"full", "deep", "both"}:
        raise ValueError("experiment must be full, deep, or both")
    dataset_config, dataset_hash, device_type, resolved_workers = (
        _validate_experiment_inputs(config)
    )
    jobs = build_jobs(
        config.dataset_dir,
        experiment,
        config.noise_scenarios,
        samples_per_kind=config.samples_per_kind,
    )
    manifest = initialize_result_manifest(
        config.output_dir,
        dataset_config_hash=dataset_hash,
        checkpoint=config.checkpoint,
        config=config,
        experiment=experiment,
        expected_jobs=tuple(job.name for job in jobs),
    )
    manifest = _recover_published_jobs(config, jobs, manifest)
    assigned = assigned_jobs(
        jobs,
        task_index=config.task_index,
        task_count=config.task_count,
    )
    pending = tuple(job for job in assigned if job.name not in manifest.completed_jobs)
    if pending:
        worker_count = min(resolved_workers, len(pending))
        worker_config = replace(
            config,
            device=device_type,
            workers=worker_count,
        )
        if worker_count == 1:
            for job in pending:
                run_inversion_job(job, worker_config, dataset_config, manifest)
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = [
                    executor.submit(
                        run_inversion_job,
                        job,
                        worker_config,
                        dataset_config,
                        manifest,
                    )
                    for job in pending
                ]
                for future in as_completed(futures):
                    future.result()

    refreshed = initialize_result_manifest(
        config.output_dir,
        dataset_config_hash=dataset_hash,
        checkpoint=config.checkpoint,
        config=config,
        experiment=experiment,
        expected_jobs=tuple(job.name for job in jobs),
    )
    if refreshed.complete:
        return validate_complete_results(config.output_dir)
    return refreshed
