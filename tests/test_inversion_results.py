from __future__ import annotations

import hashlib
import json
import multiprocessing
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from swave.config import InversionConfig, inversion_identity_hash
from swave.inversion_results import (
    ResultBatch,
    initialize_result_manifest,
    mark_job_complete,
    validate_complete_results,
    validate_result_shard,
    write_result_shard,
)
from swave.splits import SPLIT_POLICY

JOB_A = "full-clean-shard-00000"
JOB_B = "full-clean-shard-00001"


def _batch(sample_ids: tuple[int, ...] = (90, 91)) -> ResultBatch:
    count = len(sample_ids)
    observed = np.full((count, 4, 120), 1.2, dtype=np.float32)
    return ResultBatch(
        sample_id=np.asarray(sample_ids, dtype=np.uint64),
        model_kind=np.arange(count, dtype=np.uint8) % 4,
        success=np.ones(count, dtype=np.bool_),
        status=np.zeros(count, dtype=np.int32),
        iterations=np.full(count, 4, dtype=np.int32),
        evaluations=np.full(count, 5, dtype=np.int32),
        initial_objective=np.full(count, 2.0, dtype=np.float64),
        final_objective=np.full(count, 1.0, dtype=np.float64),
        data_misfit=np.full(count, 0.9, dtype=np.float64),
        regularization=np.full(count, 0.1, dtype=np.float64),
        reference_vs=np.full((count, 20), 1.0, dtype=np.float32),
        inverted_vs=np.full((count, 20), 1.1, dtype=np.float32),
        observed_phase_velocity=observed,
        surrogate_phase_velocity=observed.copy(),
        valid_mask=np.ones((count, 4, 120), dtype=np.bool_),
        failure_code=np.full(count, b"", dtype="S64"),
    )


def _deep_batch() -> ResultBatch:
    base = _batch()
    count = base.sample_id.size
    starts = 3
    return replace(
        base,
        ensemble_vs=np.full((count, starts, 20), 1.1, dtype=np.float32),
        ensemble_success=np.ones((count, starts), dtype=np.bool_),
        ensemble_status=np.zeros((count, starts), dtype=np.int32),
        ensemble_objective=np.ones((count, starts), dtype=np.float64),
        ensemble_inlier_mask=np.ones((count, starts), dtype=np.bool_),
        median_vs=np.full((count, 20), 1.1, dtype=np.float32),
        p10_vs=np.full((count, 20), 1.0, dtype=np.float32),
        p90_vs=np.full((count, 20), 1.2, dtype=np.float32),
        physical_phase_velocity=np.full((count, 4, 120), 1.15, dtype=np.float32),
        physical_valid_mask=np.ones((count, 4, 120), dtype=np.bool_),
    )


def _deep_config(minimum_valid_solutions: int = 2) -> InversionConfig:
    return replace(
        InversionConfig(),
        initial_models=3,
        minimum_valid_solutions=minimum_valid_solutions,
    )


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint")
    return path


@pytest.fixture
def result_manifest(tmp_path: Path, checkpoint: Path):
    return initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        inversion_config_hash="b" * 64,
        experiment="full",
        expected_jobs=(JOB_A,),
    )


def test_result_identity_binds_checkpoint_and_configuration(
    tmp_path: Path, checkpoint: Path
) -> None:
    manifest = initialize_result_manifest(
        tmp_path / "results",
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        inversion_config_hash="b" * 64,
        experiment="full",
        expected_jobs=(JOB_A,),
    )

    assert manifest.checkpoint_sha256 == hashlib.sha256(b"checkpoint").hexdigest()
    assert manifest.inversion_config_hash == "b" * 64
    assert manifest.split_policy == SPLIT_POLICY
    assert not manifest.complete


def test_manifest_can_derive_scientific_identity_without_operational_controls(
    tmp_path: Path, checkpoint: Path
) -> None:
    config = InversionConfig()
    first = initialize_result_manifest(
        tmp_path / "results",
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=config,
        experiment="full",
        expected_jobs=(JOB_A,),
    )
    operational_change = replace(
        config, device="cpu", workers=3, task_index=2, task_count=4
    )
    second = initialize_result_manifest(
        tmp_path / "results",
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=operational_change,
        experiment="full",
        expected_jobs=(JOB_A,),
    )

    assert first == second
    assert first.inversion_config_hash == inversion_identity_hash(config)


def test_full_identity_is_same_for_config_and_precomputed_hash_routes(
    tmp_path: Path, checkpoint: Path
) -> None:
    config = InversionConfig()
    first = initialize_result_manifest(
        tmp_path / "results",
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=config,
        experiment="full",
        expected_jobs=(JOB_A,),
    )
    second = initialize_result_manifest(
        tmp_path / "results",
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        inversion_config_hash=inversion_identity_hash(config),
        experiment="full",
        expected_jobs=(JOB_A,),
    )

    assert first == second
    assert first.minimum_valid_solutions is None


@pytest.mark.parametrize("experiment", ["deep", "both"])
def test_deep_capable_identity_rejects_unverifiable_precomputed_hash(
    tmp_path: Path, checkpoint: Path, experiment: str
) -> None:
    jobs = (
        ("deep-clean-shard-00000",)
        if experiment == "deep"
        else (JOB_A, "deep-clean-shard-00000")
    )
    with pytest.raises(ValueError, match="config.*required|precomputed"):
        initialize_result_manifest(
            tmp_path / experiment,
            dataset_config_hash="a" * 64,
            checkpoint=checkpoint,
            inversion_config_hash="b" * 64,
            minimum_valid_solutions=2,
            experiment=experiment,
            expected_jobs=jobs,
        )


def test_both_identity_uses_one_deep_minimum_and_full_job_ignores_it(
    tmp_path: Path, checkpoint: Path
) -> None:
    deep_job = "deep-clean-shard-00000"
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=_deep_config(),
        experiment="both",
        expected_jobs=(JOB_A, deep_job),
    )

    assert manifest.minimum_valid_solutions == 2
    write_result_shard(tmp_path, JOB_A, _batch(), manifest)
    write_result_shard(tmp_path, deep_job, _deep_batch(), manifest)
    mark_job_complete(tmp_path, JOB_A)
    completed = mark_job_complete(tmp_path, deep_job)

    assert completed.complete
    assert validate_complete_results(tmp_path) == completed


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_id", np.array([90, 90], dtype=np.uint64), "sample_id"),
        ("sample_id", np.array([91, 90], dtype=np.uint64), "sample_id"),
        ("status", np.zeros(2, dtype=np.int64), "status.*dtype"),
        ("reference_vs", np.ones((2, 19), dtype=np.float32), "reference_vs"),
        (
            "observed_phase_velocity",
            np.ones((2, 4, 119), dtype=np.float32),
            "observed_phase_velocity",
        ),
    ],
)
def test_result_batch_rejects_wrong_identity_shape_or_dtype(
    field: str, value: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_batch(), **{field: value})


def test_result_batch_requires_all_deep_fields_with_consistent_start_count() -> None:
    with pytest.raises(ValueError, match="optional.*all"):
        replace(_batch(), ensemble_vs=np.ones((2, 3, 20), dtype=np.float32))

    with pytest.raises(ValueError, match="ensemble_status"):
        replace(_deep_batch(), ensemble_status=np.zeros((2, 2), dtype=np.int32))


def test_successful_deep_row_requires_a_successful_inlier_start() -> None:
    base = _deep_batch()
    with pytest.raises(ValueError, match="successful deep.*inlier"):
        replace(
            base,
            ensemble_success=np.zeros((2, 3), dtype=np.bool_),
            ensemble_inlier_mask=np.zeros((2, 3), dtype=np.bool_),
        )


def test_result_batch_rejects_nan_success_and_allows_diagnostic_failure() -> None:
    invalid = _batch()
    objectives = invalid.final_objective.copy()
    objectives[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace(invalid, final_objective=objectives)

    failure = replace(
        _batch((90,)),
        success=np.array([False], dtype=np.bool_),
        status=np.array([-1], dtype=np.int32),
        iterations=np.array([0], dtype=np.int32),
        evaluations=np.array([0], dtype=np.int32),
        initial_objective=np.array([np.nan], dtype=np.float64),
        final_objective=np.array([np.nan], dtype=np.float64),
        data_misfit=np.array([np.nan], dtype=np.float64),
        regularization=np.array([np.nan], dtype=np.float64),
        reference_vs=np.full((1, 20), np.nan, dtype=np.float32),
        inverted_vs=np.full((1, 20), np.nan, dtype=np.float32),
        surrogate_phase_velocity=np.full((1, 4, 120), np.nan, dtype=np.float32),
        failure_code=np.array([b"insufficient_fundamental_data"], dtype="S64"),
    )
    assert not failure.success[0]

    with pytest.raises(ValueError, match="failure_code"):
        replace(failure, failure_code=np.array([b""], dtype="S64"))


def test_finite_objective_terms_must_sum_to_final_objective() -> None:
    with pytest.raises(ValueError, match="final_objective.*sum"):
        replace(
            _batch(),
            final_objective=np.array([1.5, 1.0], dtype=np.float64),
        )


def test_failed_row_can_retain_reference_when_optimizer_outputs_are_nan() -> None:
    base = _batch((90,))
    failure = replace(
        base,
        success=np.array([False], dtype=np.bool_),
        status=np.array([-1], dtype=np.int32),
        initial_objective=np.array([np.nan], dtype=np.float64),
        final_objective=np.array([np.nan], dtype=np.float64),
        data_misfit=np.array([np.nan], dtype=np.float64),
        regularization=np.array([np.nan], dtype=np.float64),
        inverted_vs=np.full((1, 20), np.nan, dtype=np.float32),
        surrogate_phase_velocity=np.full((1, 4, 120), np.nan, dtype=np.float32),
        failure_code=np.array([b"nonfinite_objective"], dtype="S64"),
    )

    assert np.all(np.isfinite(failure.reference_vs))
    assert np.all(np.isnan(failure.inverted_vs))


def test_write_round_trip_has_identity_and_content_checksums(
    tmp_path: Path, result_manifest
) -> None:
    batch = _batch()
    path = write_result_shard(tmp_path, JOB_A, batch, result_manifest)
    loaded = validate_result_shard(
        path, expected_sample_ids=batch.sample_id, manifest=result_manifest
    )

    np.testing.assert_array_equal(loaded.sample_id, batch.sample_id)
    with h5py.File(path) as handle:
        assert handle.attrs["job_name"] == JOB_A
        assert handle.attrs["sample_id_sha256"]
        assert handle.attrs["content_sha256"]
    assert not list(tmp_path.glob("*.tmp-*"))


def test_deep_result_fields_round_trip(tmp_path: Path, checkpoint: Path) -> None:
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=_deep_config(),
        experiment="deep",
        expected_jobs=("deep-clean-shard-00000",),
    )
    path = write_result_shard(
        tmp_path, "deep-clean-shard-00000", _deep_batch(), manifest
    )

    loaded = validate_result_shard(path, manifest=manifest)
    assert loaded.ensemble_vs is not None
    assert loaded.ensemble_vs.shape == (2, 3, 20)
    assert loaded.physical_phase_velocity is not None


def test_deep_job_rejects_a_full_style_result_batch(
    tmp_path: Path, checkpoint: Path
) -> None:
    job = "deep-clean-shard-00000"
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=_deep_config(),
        experiment="deep",
        expected_jobs=(job,),
    )

    with pytest.raises(ValueError, match="deep.*optional|deep.*ensemble"):
        write_result_shard(tmp_path, job, _batch(), manifest)


def test_deep_success_must_meet_bound_minimum_valid_solutions(
    tmp_path: Path, checkpoint: Path
) -> None:
    job = "deep-clean-shard-00000"
    config = _deep_config()
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=config,
        experiment="deep",
        expected_jobs=(job,),
    )
    one_inlier = replace(
        _deep_batch(),
        ensemble_success=np.array(
            [[True, False, False], [True, False, False]], dtype=np.bool_
        ),
        ensemble_inlier_mask=np.array(
            [[True, False, False], [True, False, False]], dtype=np.bool_
        ),
    )

    assert manifest.minimum_valid_solutions == 2
    with pytest.raises(ValueError, match="identity"):
        initialize_result_manifest(
            tmp_path,
            dataset_config_hash="a" * 64,
            checkpoint=checkpoint,
            config=replace(config, minimum_valid_solutions=3),
            experiment="deep",
            expected_jobs=(job,),
        )
    with pytest.raises(ValueError, match="minimum_valid_solutions|inlier.*minimum"):
        write_result_shard(tmp_path, job, one_inlier, manifest)


def test_deep_minimum_is_derived_from_and_bound_to_inversion_config(
    tmp_path: Path, checkpoint: Path
) -> None:
    config = _deep_config()
    job = "deep-clean-shard-00000"
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=config,
        experiment="deep",
        expected_jobs=(job,),
    )

    assert manifest.minimum_valid_solutions == 2
    assert manifest.inversion_config_hash == inversion_identity_hash(config)
    with pytest.raises(ValueError, match="minimum_valid_solutions.*config"):
        initialize_result_manifest(
            tmp_path / "conflict",
            dataset_config_hash="a" * 64,
            checkpoint=checkpoint,
            config=config,
            minimum_valid_solutions=1,
            experiment="deep",
            expected_jobs=(job,),
        )


def test_insufficient_valid_solutions_code_requires_too_few_inliers(
    tmp_path: Path, checkpoint: Path
) -> None:
    job = "deep-clean-shard-00000"
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=_deep_config(),
        experiment="deep",
        expected_jobs=(job,),
    )
    base = _deep_batch()
    inconsistent = replace(
        base,
        success=np.zeros(2, dtype=np.bool_),
        failure_code=np.full(2, b"insufficient_valid_solutions", dtype="S64"),
    )

    with pytest.raises(ValueError, match="insufficient_valid_solutions.*inlier"):
        write_result_shard(tmp_path, job, inconsistent, manifest)


def test_insufficient_valid_solutions_cannot_publish_arbitrary_summary(
    tmp_path: Path, checkpoint: Path
) -> None:
    job = "deep-clean-shard-00000"
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        config=_deep_config(),
        experiment="deep",
        expected_jobs=(job,),
    )
    inconsistent = replace(
        _deep_batch(),
        success=np.zeros(2, dtype=np.bool_),
        failure_code=np.full(2, b"insufficient_valid_solutions", dtype="S64"),
        ensemble_success=np.array(
            [[True, False, False], [True, False, False]], dtype=np.bool_
        ),
        ensemble_inlier_mask=np.array(
            [[True, False, False], [True, False, False]], dtype=np.bool_
        ),
    )

    with pytest.raises(ValueError, match="insufficient_valid_solutions.*summary"):
        write_result_shard(tmp_path, job, inconsistent, manifest)


@pytest.mark.parametrize("dataset", ["sample_id", "inverted_vs", "status"])
def test_corrupt_completed_result_is_rejected(
    tmp_path: Path, result_manifest, dataset: str
) -> None:
    batch = _batch()
    path = write_result_shard(tmp_path, JOB_A, batch, result_manifest)
    with h5py.File(path, "r+") as handle:
        if dataset == "sample_id":
            handle[dataset][0] = 999
        elif dataset == "inverted_vs":
            handle[dataset][0, 0] = np.nan
        else:
            handle[dataset][0] = 42

    with pytest.raises(ValueError, match="checksum|sample_id|finite"):
        validate_result_shard(path, expected_sample_ids=batch.sample_id)


def test_write_resume_is_idempotent_but_conflicting_content_is_rejected(
    tmp_path: Path, result_manifest
) -> None:
    first = write_result_shard(tmp_path, JOB_A, _batch(), result_manifest)
    before = first.stat().st_mtime_ns
    assert write_result_shard(tmp_path, JOB_A, _batch(), result_manifest) == first
    assert first.stat().st_mtime_ns == before

    changed = replace(
        _batch(),
        final_objective=np.array([0.5, 1.0], dtype=np.float64),
        data_misfit=np.array([0.4, 0.9], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="conflicting|checksum"):
        write_result_shard(tmp_path, JOB_A, changed, result_manifest)


@pytest.mark.parametrize("job", ["../escape", "nested/job", ".", "job.h5", ""])
def test_job_names_cannot_escape_the_result_directory(
    tmp_path: Path, checkpoint: Path, job: str
) -> None:
    with pytest.raises(ValueError, match="job name"):
        initialize_result_manifest(
            tmp_path / "results",
            dataset_config_hash="a" * 64,
            checkpoint=checkpoint,
            inversion_config_hash="b" * 64,
            experiment="full",
            expected_jobs=(job,),
        )


def test_duplicate_expected_jobs_are_rejected(tmp_path: Path, checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        initialize_result_manifest(
            tmp_path / "results",
            dataset_config_hash="a" * 64,
            checkpoint=checkpoint,
            inversion_config_hash="b" * 64,
            experiment="full",
            expected_jobs=(JOB_A, JOB_A),
        )


@pytest.mark.parametrize(
    ("experiment", "job"),
    [
        ("deep", JOB_A),
        ("full", "deep-clean-shard-00000"),
    ],
)
def test_manifest_experiment_rejects_contradictory_job_prefix(
    tmp_path: Path, checkpoint: Path, experiment: str, job: str
) -> None:
    with pytest.raises(ValueError, match="experiment.*job|job.*experiment"):
        initialize_result_manifest(
            tmp_path / experiment,
            dataset_config_hash="a" * 64,
            checkpoint=checkpoint,
            config=InversionConfig(),
            experiment=experiment,
            expected_jobs=(job,),
        )


@pytest.mark.parametrize(
    "expected_jobs",
    [
        (JOB_A,),
        ("deep-clean-shard-00000",),
    ],
)
def test_both_manifest_requires_full_and_deep_expected_jobs(
    tmp_path: Path, checkpoint: Path, expected_jobs: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="both.*full.*deep|full.*deep.*both"):
        initialize_result_manifest(
            tmp_path / expected_jobs[0],
            dataset_config_hash="a" * 64,
            checkpoint=checkpoint,
            config=_deep_config(),
            experiment="both",
            expected_jobs=expected_jobs,
        )


def test_result_paths_reject_symlinks(tmp_path: Path, result_manifest) -> None:
    outside = tmp_path / "outside.h5"
    outside.write_bytes(b"outside")
    (tmp_path / f"{JOB_A}.h5").symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic link"):
        write_result_shard(tmp_path, JOB_A, _batch(), result_manifest)
    assert outside.read_bytes() == b"outside"


def test_result_directory_rejects_a_symlinked_parent(
    tmp_path: Path, checkpoint: Path
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        initialize_result_manifest(
            alias / "results",
            dataset_config_hash="a" * 64,
            checkpoint=checkpoint,
            inversion_config_hash="b" * 64,
            experiment="full",
            expected_jobs=(JOB_A,),
        )


def test_result_directory_rejects_symlink_before_parent_traversal(
    tmp_path: Path, checkpoint: Path
) -> None:
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    pivot = outside / "pivot"
    pivot.mkdir(parents=True)
    (inside / "link").symlink_to(pivot, target_is_directory=True)
    apparent = inside / "link" / ".." / "escaped-results"

    with pytest.raises(ValueError, match="symbolic link"):
        initialize_result_manifest(
            apparent,
            dataset_config_hash="a" * 64,
            checkpoint=checkpoint,
            inversion_config_hash="b" * 64,
            experiment="full",
            expected_jobs=(JOB_A,),
        )

    assert not (outside / "escaped-results" / "manifest.json").exists()


def test_initialization_resumes_exact_identity_and_rejects_conflicts(
    tmp_path: Path, checkpoint: Path
) -> None:
    directory = tmp_path / "results"
    kwargs = {
        "dataset_config_hash": "a" * 64,
        "checkpoint": checkpoint,
        "inversion_config_hash": "b" * 64,
        "experiment": "full",
        "expected_jobs": (JOB_A,),
    }
    first = initialize_result_manifest(directory, **kwargs)
    assert initialize_result_manifest(directory, **kwargs) == first

    with pytest.raises(ValueError, match="identity"):
        initialize_result_manifest(
            directory, **{**kwargs, "inversion_config_hash": "c" * 64}
        )


def test_result_shard_job_attribute_must_match_its_file_name(
    tmp_path: Path, checkpoint: Path
) -> None:
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        inversion_config_hash="b" * 64,
        experiment="full",
        expected_jobs=(JOB_A, JOB_B),
    )
    path = write_result_shard(tmp_path, JOB_A, _batch((90,)), manifest)
    renamed = tmp_path / f"{JOB_B}.h5"
    path.rename(renamed)

    with pytest.raises(ValueError, match="job name|file name"):
        validate_result_shard(renamed, manifest=manifest)


def test_manifest_json_is_strictly_validated(tmp_path: Path, checkpoint: Path) -> None:
    directory = tmp_path / "results"
    initialize_result_manifest(
        directory,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        inversion_config_hash="b" * 64,
        experiment="full",
        expected_jobs=(JOB_A,),
    )
    manifest_path = directory / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.*fields"):
        initialize_result_manifest(
            directory,
            dataset_config_hash="a" * 64,
            checkpoint=checkpoint,
            inversion_config_hash="b" * 64,
            experiment="full",
            expected_jobs=(JOB_A,),
        )


def test_mark_complete_is_idempotent_and_complete_validation_is_strict(
    tmp_path: Path, checkpoint: Path
) -> None:
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        inversion_config_hash="b" * 64,
        experiment="full",
        expected_jobs=(JOB_A, JOB_B),
    )
    write_result_shard(tmp_path, JOB_A, _batch((90,)), manifest)
    first = mark_job_complete(tmp_path, JOB_A)
    assert first.completed_jobs == (JOB_A,)
    assert mark_job_complete(tmp_path, JOB_A) == first
    with pytest.raises(ValueError, match="incomplete"):
        validate_complete_results(tmp_path)

    write_result_shard(tmp_path, JOB_B, _batch((91,)), manifest)
    completed = mark_job_complete(tmp_path, JOB_B)
    assert completed.complete
    assert validate_complete_results(tmp_path) == completed

    unexpected = tmp_path / "unexpected.h5"
    unexpected.write_bytes(b"not hdf5")
    with pytest.raises(ValueError, match="unexpected|files"):
        validate_complete_results(tmp_path)


def test_partial_temporary_file_never_counts_as_a_completed_job(
    tmp_path: Path, checkpoint: Path
) -> None:
    initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        inversion_config_hash="b" * 64,
        experiment="full",
        expected_jobs=(JOB_A,),
    )
    (tmp_path / f"{JOB_A}.h5.tmp-99999").write_bytes(b"partial")

    with pytest.raises(ValueError, match="incomplete"):
        validate_complete_results(tmp_path)


def _mark_job(directory: str, job: str, queue) -> None:
    try:
        queue.put((job, mark_job_complete(directory, job).completed_jobs))
    except (OSError, ValueError) as error:  # pragma: no cover - asserted in parent
        queue.put((job, type(error).__name__, str(error)))


def test_concurrent_manifest_updates_do_not_lose_jobs(
    tmp_path: Path, checkpoint: Path
) -> None:
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        inversion_config_hash="b" * 64,
        experiment="full",
        expected_jobs=(JOB_A, JOB_B),
    )
    write_result_shard(tmp_path, JOB_A, _batch((90,)), manifest)
    write_result_shard(tmp_path, JOB_B, _batch((91,)), manifest)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_mark_job, args=(str(tmp_path), job, queue))
        for job in (JOB_A, JOB_B)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    outcomes = [queue.get(timeout=5) for _ in processes]

    assert all(len(outcome) == 2 for outcome in outcomes), outcomes
    completed = validate_complete_results(tmp_path)
    assert completed.completed_jobs == (JOB_A, JOB_B)


def test_complete_validation_rejects_duplicate_ids_within_one_shard(
    tmp_path: Path, result_manifest
) -> None:
    path = write_result_shard(tmp_path, JOB_A, _batch(), result_manifest)
    mark_job_complete(tmp_path, JOB_A)
    with h5py.File(path, "r+") as handle:
        handle["sample_id"][1] = handle["sample_id"][0]

    with pytest.raises(ValueError, match="checksum|sample_id"):
        validate_complete_results(tmp_path)


def test_complete_validation_rejects_duplicate_ids_across_same_scenario(
    tmp_path: Path, checkpoint: Path
) -> None:
    manifest = initialize_result_manifest(
        tmp_path,
        dataset_config_hash="a" * 64,
        checkpoint=checkpoint,
        inversion_config_hash="b" * 64,
        experiment="full",
        expected_jobs=(JOB_A, JOB_B),
    )
    write_result_shard(tmp_path, JOB_A, _batch((90,)), manifest)
    write_result_shard(tmp_path, JOB_B, _batch((90,)), manifest)
    mark_job_complete(tmp_path, JOB_A)
    mark_job_complete(tmp_path, JOB_B)

    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_complete_results(tmp_path)
