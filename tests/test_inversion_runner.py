from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import swave.inversion_runner as runner_module
from swave.config import DatasetConfig, InversionConfig, canonical_hash
from swave.inversion import EnsembleResult, InversionRun, ObjectiveTerms
from swave.inversion_data import InversionSample
from swave.inversion_results import (
    initialize_result_manifest,
    validate_result_shard,
)
from swave.inversion_runner import (
    InversionJob,
    assigned_jobs,
    build_jobs,
    run_inversion_experiment,
    run_inversion_job,
)
from swave.solver import DispersionResult
from swave.splits import SPLIT_POLICY


def test_cluster_assignment_is_disjoint_and_complete() -> None:
    jobs = tuple(
        InversionJob(
            name=f"job-{index}",
            experiment="full",
            noise="clean",
            samples=(),
        )
        for index in range(17)
    )

    assigned = [
        assigned_jobs(jobs, task_index=index, task_count=4) for index in range(4)
    ]
    names = [{job.name for job in group} for group in assigned]

    assert not any(
        names[left] & names[right] for left in range(4) for right in range(left + 1, 4)
    )
    assert set().union(*names) == {job.name for job in jobs}


def test_job_names_are_stable_by_experiment_noise_and_source_shard(
    tiny_complete_dataset,
) -> None:
    jobs = build_jobs(
        tiny_complete_dataset,
        "full",
        ("clean", "noise_1pct"),
        samples_per_kind=1,
    )

    assert [job.name for job in jobs] == [
        "full-clean-shard-00000",
        "full-noise_1pct-shard-00000",
    ]
    assert [[sample.sample_id for sample in job.samples] for job in jobs] == [
        [90, 91, 92, 93],
        [90, 91, 92, 93],
    ]


def test_both_jobs_include_full_and_deep_families_for_tiny_data(
    tiny_complete_dataset,
) -> None:
    jobs = build_jobs(
        tiny_complete_dataset,
        "both",
        ("clean",),
        samples_per_kind=1,
    )

    assert [job.name for job in jobs] == [
        "full-clean-shard-00000",
        "deep-clean-shard-00000",
    ]
    assert [sample.sample_id for sample in jobs[1].samples] == [90, 91, 92, 93]


def _sample(sample_id: int, fundamental_cells: int = 120) -> InversionSample:
    observed = np.full((4, 120), 1.0, dtype=np.float32)
    mask = np.ones((4, 120), dtype=np.bool_)
    if fundamental_cells < 120:
        mask[0, fundamental_cells:] = False
        observed[0, fundamental_cells:] = np.inf
    return InversionSample(
        sample_id=sample_id,
        model_kind=sample_id % 4,
        phase_velocity=observed,
        valid_mask=mask,
        source_path=Path("must-not-be-opened.h5"),
        source_shard_id=0,
        source_row=sample_id,
    )


class _SurrogateStub:
    device = "cpu"


class _ObjectiveStub:
    def __init__(self, **kwargs) -> None:
        self.frequencies = np.asarray(kwargs["frequencies"])


def _successful_run(success: bool = True) -> InversionRun:
    return InversionRun(
        vs=np.full(20, 1.1),
        predicted_phase_velocity=np.full((4, 120), 1.2),
        success=success,
        status=0 if success else 2,
        message="ok" if success else "iteration limit",
        iterations=4,
        evaluations=5,
        initial_objective=2.0,
        terms=ObjectiveTerms(total=1.0, data_misfit=0.9, regularization=0.1),
    )


@pytest.fixture
def full_job_context(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = replace(
        InversionConfig(),
        checkpoint=checkpoint,
        output_dir=tmp_path / "results",
        device="cpu",
        workers=1,
    )
    job = InversionJob(
        name="full-clean-shard-00000",
        experiment="full",
        noise="clean",
        samples=(
            _sample(90, fundamental_cells=1),
            _sample(91),
            _sample(92),
            _sample(93),
        ),
    )
    manifest = initialize_result_manifest(
        config.output_dir,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=config,
        experiment="full",
        expected_jobs=(job.name,),
    )
    return config, job, manifest


def test_full_job_records_stable_sample_failures_without_opening_source_vs(
    monkeypatch, full_job_context
) -> None:
    config, job, manifest = full_job_context
    loads = 0

    def load_surrogate(checkpoint, device):
        nonlocal loads
        loads += 1
        return _SurrogateStub()

    outcomes = iter(
        [
            ArithmeticError("objective is non-finite"),
            RuntimeError("optimizer crashed"),
            _successful_run(success=False),
        ]
    )

    def invert(*args, **kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(runner_module.DifferentiableSurrogate, "load", load_surrogate)
    monkeypatch.setattr(runner_module, "SurrogateObjective", _ObjectiveStub)
    monkeypatch.setattr(runner_module, "invert_one", invert)

    completed = run_inversion_job(job, config, DatasetConfig(), manifest)
    batch = validate_result_shard(
        config.output_dir / f"{job.name}.h5", manifest=completed
    )

    assert loads == 1
    assert completed.complete
    assert batch.failure_code.tolist() == [
        b"insufficient_fundamental_data",
        b"nonfinite_objective",
        b"optimizer_failure",
        b"optimizer_failure",
    ]
    assert not np.any(batch.success)
    assert np.all(np.isnan(batch.observed_phase_velocity[0, 0, 1:]))
    assert np.all(np.isnan(batch.inverted_vs[:3]))
    assert np.all(np.isfinite(batch.inverted_vs[3]))


def _ensemble(*, sufficient: bool) -> EnsembleResult:
    successful_runs = (_successful_run(), _successful_run(), _successful_run())
    nan_model = np.full(20, np.nan)
    if sufficient:
        return EnsembleResult(
            runs=successful_runs,
            inlier_mask=np.ones(3, dtype=np.bool_),
            median_vs=np.full(20, 1.1),
            p10_vs=np.full(20, 1.0),
            p90_vs=np.full(20, 1.2),
            representative_terms=ObjectiveTerms(1.0, 0.9, 0.1),
            representative_prediction=np.full((4, 120), 1.2),
            sufficient=True,
        )
    failed_run = replace(
        _successful_run(),
        vs=nan_model.copy(),
        predicted_phase_velocity=np.full((4, 120), np.nan),
        success=False,
        status=-1,
        message="failed",
        initial_objective=np.nan,
        terms=ObjectiveTerms(np.nan, np.nan, np.nan),
    )
    return EnsembleResult(
        runs=(successful_runs[0], failed_run, failed_run),
        inlier_mask=np.array([True, False, False], dtype=np.bool_),
        median_vs=nan_model.copy(),
        p10_vs=nan_model.copy(),
        p90_vs=nan_model.copy(),
        representative_terms=ObjectiveTerms(np.nan, np.nan, np.nan),
        representative_prediction=np.full((4, 120), np.nan),
        sufficient=False,
    )


def _deep_context(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = replace(
        InversionConfig(),
        checkpoint=checkpoint,
        output_dir=tmp_path / "results",
        initial_models=3,
        minimum_valid_solutions=2,
        samples_per_kind=1,
        device="cpu",
        workers=1,
    )
    job = InversionJob(
        name="deep-clean-shard-00000",
        experiment="deep",
        noise="clean",
        samples=(_sample(90),),
    )
    manifest = initialize_result_manifest(
        config.output_dir,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=config,
        experiment="deep",
        expected_jobs=(job.name,),
    )
    return config, job, manifest


def test_deep_job_revalidates_only_sufficient_median_and_nan_masks_invalid_cells(
    monkeypatch, tmp_path: Path
) -> None:
    config, job, manifest = _deep_context(tmp_path)
    phase = np.full((4, 120), 1.3)
    physical_mask = np.ones((4, 120), dtype=np.bool_)
    phase[3, 0] = np.inf
    physical_mask[3, 0] = False

    monkeypatch.setattr(
        runner_module.DifferentiableSurrogate,
        "load",
        lambda checkpoint, device: _SurrogateStub(),
    )
    monkeypatch.setattr(runner_module, "SurrogateObjective", _ObjectiveStub)
    monkeypatch.setattr(
        runner_module,
        "invert_ensemble",
        lambda *args, **kwargs: _ensemble(sufficient=True),
    )
    monkeypatch.setattr(
        runner_module.DispersionSolver,
        "solve_grid",
        lambda self, frequencies, strategy: DispersionResult(
            frequencies=np.asarray(frequencies),
            phase_velocity=phase,
            valid_mask=physical_mask,
            status=np.zeros(120, dtype=np.uint8),
            evaluations=np.ones(120, dtype=np.int64),
        ),
    )

    completed = run_inversion_job(job, config, DatasetConfig(), manifest)
    batch = validate_result_shard(
        config.output_dir / f"{job.name}.h5", manifest=completed
    )

    assert batch.success.tolist() == [True]
    assert batch.failure_code.tolist() == [b""]
    assert batch.physical_valid_mask is not None
    assert batch.physical_phase_velocity is not None
    assert not batch.physical_valid_mask[0, 3, 0]
    assert np.isnan(batch.physical_phase_velocity[0, 3, 0])


def test_deep_job_records_insufficient_ensemble_without_physical_revalidation(
    monkeypatch, tmp_path: Path
) -> None:
    config, job, manifest = _deep_context(tmp_path)

    monkeypatch.setattr(
        runner_module.DifferentiableSurrogate,
        "load",
        lambda checkpoint, device: _SurrogateStub(),
    )
    monkeypatch.setattr(runner_module, "SurrogateObjective", _ObjectiveStub)
    monkeypatch.setattr(
        runner_module,
        "invert_ensemble",
        lambda *args, **kwargs: _ensemble(sufficient=False),
    )
    monkeypatch.setattr(
        runner_module.DispersionSolver,
        "solve_grid",
        lambda *args, **kwargs: pytest.fail(
            "insufficient ensembles must not run physical revalidation"
        ),
    )

    completed = run_inversion_job(job, config, DatasetConfig(), manifest)
    batch = validate_result_shard(
        config.output_dir / f"{job.name}.h5", manifest=completed
    )

    assert batch.success.tolist() == [False]
    assert batch.failure_code.tolist() == [b"insufficient_valid_solutions"]
    assert batch.median_vs is not None
    assert np.all(np.isnan(batch.median_vs))


def test_deep_job_records_physical_solver_exception_without_fabricating_success(
    monkeypatch, tmp_path: Path
) -> None:
    config, job, manifest = _deep_context(tmp_path)
    monkeypatch.setattr(
        runner_module.DifferentiableSurrogate,
        "load",
        lambda checkpoint, device: _SurrogateStub(),
    )
    monkeypatch.setattr(runner_module, "SurrogateObjective", _ObjectiveStub)
    monkeypatch.setattr(
        runner_module,
        "invert_ensemble",
        lambda *args, **kwargs: _ensemble(sufficient=True),
    )
    monkeypatch.setattr(
        runner_module.DispersionSolver,
        "solve_grid",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ArithmeticError("Dunkin failure")
        ),
    )

    completed = run_inversion_job(job, config, DatasetConfig(), manifest)
    batch = validate_result_shard(
        config.output_dir / f"{job.name}.h5", manifest=completed
    )

    assert batch.success.tolist() == [False]
    assert batch.failure_code.tolist() == [b"physical_solver_failure"]
    assert np.all(np.isfinite(batch.inverted_vs))
    assert batch.physical_phase_velocity is not None
    assert np.all(np.isnan(batch.physical_phase_velocity))


def _runnable_config(
    tiny_complete_dataset: Path,
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> InversionConfig:
    dataset_config = DatasetConfig(output_dir=tiny_complete_dataset)
    dataset_hash = canonical_hash(dataset_config)
    shard_path = tiny_complete_dataset / "shard-00000.h5"
    with h5py.File(shard_path, "r+") as handle:
        handle.attrs["config_hash"] = dataset_hash
    manifest_path = tiny_complete_dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_hash"] = dataset_hash
    manifest["shard_sha256"]["0"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    checkpoint_payload = torch.load(
        tiny_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_payload["dataset_config_hash"] = dataset_hash
    checkpoint_payload["split_policy"] = SPLIT_POLICY
    torch.save(checkpoint_payload, tiny_checkpoint)

    dataset_config_path = tmp_path / "dataset.toml"
    dataset_config_path.write_text(
        "\n".join(
            [
                "[dataset]",
                f"output_dir = {json.dumps(tiny_complete_dataset.as_posix())}",
                "samples = 1000000",
                "shard_size = 10000",
                "seed = 20260727",
                "workers = 0",
                "max_model_retries = 8",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return replace(
        InversionConfig(),
        dataset_config=dataset_config_path,
        dataset_dir=tiny_complete_dataset,
        checkpoint=tiny_checkpoint,
        output_dir=tmp_path / "results",
        initial_models=3,
        minimum_valid_solutions=2,
        samples_per_kind=1,
        device="cpu",
        workers=1,
    )


def _patch_successful_execution(monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module.DifferentiableSurrogate,
        "load",
        lambda checkpoint, device: _SurrogateStub(),
    )
    monkeypatch.setattr(runner_module, "SurrogateObjective", _ObjectiveStub)
    monkeypatch.setattr(
        runner_module, "invert_one", lambda *args, **kwargs: _successful_run()
    )
    monkeypatch.setattr(
        runner_module,
        "invert_ensemble",
        lambda *args, **kwargs: _ensemble(sufficient=True),
    )
    monkeypatch.setattr(
        runner_module.DispersionSolver,
        "solve_grid",
        lambda self, frequencies, strategy: DispersionResult(
            frequencies=np.asarray(frequencies),
            phase_velocity=np.full((4, 120), 1.3),
            valid_mask=np.ones((4, 120), dtype=np.bool_),
            status=np.zeros(120, dtype=np.uint8),
            evaluations=np.ones(120, dtype=np.int64),
        ),
    )


def test_tiny_both_experiment_is_exactly_resumable(
    monkeypatch,
    tiny_complete_dataset: Path,
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    config = _runnable_config(tiny_complete_dataset, tiny_checkpoint, tmp_path)
    _patch_successful_execution(monkeypatch)

    first = run_inversion_experiment(config, "both")
    paths = sorted(config.output_dir.glob("*.h5"))
    mtimes = {path.name: path.stat().st_mtime_ns for path in paths}

    assert first.complete
    assert [path.stem for path in paths] == [
        "deep-clean-shard-00000",
        "deep-noise_1pct-shard-00000",
        "full-clean-shard-00000",
        "full-noise_1pct-shard-00000",
    ]
    for path in paths:
        batch = validate_result_shard(path, manifest=first)
        assert all(int(sample_id) % 100 >= 90 for sample_id in batch.sample_id)

    monkeypatch.setattr(
        runner_module.DifferentiableSurrogate,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "completed jobs must not reload the surrogate"
        ),
    )
    resumed = run_inversion_experiment(config, "both")
    assert resumed == first
    assert {path.name: path.stat().st_mtime_ns for path in paths} == mtimes

    with pytest.raises(ValueError, match="identity"):
        run_inversion_experiment(replace(config, regularization_lambda=0.02), "both")


def test_valid_published_shard_is_recovered_after_pre_manifest_crash(
    monkeypatch,
    tiny_complete_dataset: Path,
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    config = replace(
        _runnable_config(tiny_complete_dataset, tiny_checkpoint, tmp_path),
        noise_scenarios=("clean",),
    )
    _patch_successful_execution(monkeypatch)
    real_mark_complete = runner_module.mark_job_complete
    monkeypatch.setattr(
        runner_module,
        "mark_job_complete",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated interruption after atomic publish")
        ),
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_inversion_experiment(config, "full")
    shard = config.output_dir / "full-clean-shard-00000.h5"
    assert shard.is_file()

    monkeypatch.setattr(runner_module, "mark_job_complete", real_mark_complete)
    monkeypatch.setattr(
        runner_module.DifferentiableSurrogate,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "a valid published shard must recover without recomputation"
        ),
    )
    recovered = run_inversion_experiment(config, "full")

    assert recovered.complete
    assert recovered.completed_jobs == ("full-clean-shard-00000",)


def test_corrupt_completed_shard_is_rejected_before_recomputation(
    monkeypatch,
    tiny_complete_dataset: Path,
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    config = replace(
        _runnable_config(tiny_complete_dataset, tiny_checkpoint, tmp_path),
        noise_scenarios=("clean",),
    )
    _patch_successful_execution(monkeypatch)
    run_inversion_experiment(config, "full")
    shard = config.output_dir / "full-clean-shard-00000.h5"
    with h5py.File(shard, "r+") as handle:
        handle["sample_id"][0] = 999
    monkeypatch.setattr(
        runner_module.DifferentiableSurrogate,
        "load",
        lambda *args, **kwargs: pytest.fail(
            "corrupt completed work must not be recomputed silently"
        ),
    )

    with pytest.raises(ValueError, match="checksum|sample_id"):
        run_inversion_experiment(config, "full")


def test_complete_resume_rejects_an_unexpected_hdf5_shard(
    monkeypatch,
    tiny_complete_dataset: Path,
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    config = replace(
        _runnable_config(tiny_complete_dataset, tiny_checkpoint, tmp_path),
        noise_scenarios=("clean",),
    )
    _patch_successful_execution(monkeypatch)
    run_inversion_experiment(config, "full")
    (config.output_dir / "unexpected.h5").write_bytes(b"not a result shard")

    with pytest.raises(ValueError, match="unexpected|files"):
        run_inversion_experiment(config, "full")


def test_experiment_paths_never_access_the_source_vs_dataset(
    monkeypatch,
    tiny_complete_dataset: Path,
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    config = replace(
        _runnable_config(tiny_complete_dataset, tiny_checkpoint, tmp_path),
        noise_scenarios=("clean",),
    )
    _patch_successful_execution(monkeypatch)
    original_getitem = h5py.Group.__getitem__

    def reject_vs_access(group, name):
        decoded = name.decode("utf-8") if isinstance(name, bytes) else str(name)
        if decoded.strip("/") == "vs":
            raise AssertionError("inversion paths must never access source HDF5 vs")
        return original_getitem(group, name)

    monkeypatch.setattr(h5py.Group, "__getitem__", reject_vs_access)

    jobs = build_jobs(
        tiny_complete_dataset,
        "both",
        ("clean",),
        samples_per_kind=1,
    )
    manifest = run_inversion_experiment(config, "full")

    assert len(jobs) == 2
    assert manifest.complete


def test_cpu_workers_submit_each_job_with_spawn_context(
    monkeypatch,
    tiny_complete_dataset: Path,
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    config = replace(
        _runnable_config(tiny_complete_dataset, tiny_checkpoint, tmp_path),
        workers=2,
    )
    _patch_successful_execution(monkeypatch)
    executor_calls = []

    class RecordingExecutor:
        def __init__(self, *, max_workers, mp_context) -> None:
            executor_calls.append((max_workers, mp_context.get_start_method()))

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def submit(self, function, *args) -> Future:
            future = Future()
            future.set_result(function(*args))
            return future

    monkeypatch.setattr(runner_module, "ProcessPoolExecutor", RecordingExecutor)

    manifest = run_inversion_experiment(config, "full")

    assert executor_calls == [
        (2, multiprocessing.get_context("spawn").get_start_method())
    ]
    assert manifest.complete
    assert manifest.completed_jobs == (
        "full-clean-shard-00000",
        "full-noise_1pct-shard-00000",
    )


def test_cuda_rejects_multiple_workers_before_job_submission(
    monkeypatch,
    tiny_complete_dataset: Path,
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    config = replace(
        _runnable_config(tiny_complete_dataset, tiny_checkpoint, tmp_path),
        workers=2,
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_inversion_device",
        lambda requested: torch.device("cuda"),
    )

    with pytest.raises(ValueError, match="workers == 1"):
        run_inversion_experiment(config, "full")
