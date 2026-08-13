from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import swave.hybrid_runner as runner_module
from swave.config import DatasetConfig, HybridInversionConfig
from swave.hybrid_inversion import HybridObjectiveTerms, LearningPrior
from swave.hybrid_results import (
    initialize_hybrid_manifest,
    validate_hybrid_result_shard,
)
from swave.hybrid_runner import (
    HybridJob,
    HybridPairOutcome,
    PreparedHybridSample,
    assigned_hybrid_jobs,
    build_hybrid_jobs,
    choose_prior_lambda,
    run_hybrid_job,
    select_prior_lambda,
)
from swave.inversion import InversionRun, ObjectiveTerms, ReferenceModel
from swave.inversion_data import InversionSample


def test_hybrid_jobs_are_stable_disjoint_and_complete(
    tiny_complete_dataset: Path,
) -> None:
    jobs = build_hybrid_jobs(
        tiny_complete_dataset,
        "inversion",
        ("clean", "noise_1pct"),
    )

    assert [job.name for job in jobs] == [
        "hybrid-inversion-clean-shard-00000",
        "hybrid-inversion-noise_1pct-shard-00000",
    ]
    assignments = [
        assigned_hybrid_jobs(jobs, task_index=index, task_count=2)
        for index in range(2)
    ]
    assert not ({job.name for job in assignments[0]} & {job.name for job in assignments[1]})
    assert set().union(*({job.name for job in group} for group in assignments)) == {
        job.name for job in jobs
    }
    assert all(isinstance(job, HybridJob) for job in jobs)


def test_choose_prior_lambda_uses_mean_error_and_smaller_tie_break() -> None:
    errors = {
        0.001: np.array([0.4, 0.2, 0.3]),
        0.01: np.array([0.1, 0.2, 0.3]),
        0.1: np.array([0.2, 0.2, 0.2]),
    }

    selected, scores = choose_prior_lambda(errors)

    assert selected == pytest.approx(0.01)
    assert scores == pytest.approx({0.001: 0.3, 0.01: 0.2, 0.1: 0.2})


@pytest.mark.parametrize(
    "errors",
    [
        {},
        {0.1: np.array([])},
        {0.1: np.array([np.nan])},
        {0.0: np.array([0.1])},
    ],
)
def test_choose_prior_lambda_rejects_incomplete_metrics(
    errors: dict[float, np.ndarray],
) -> None:
    with pytest.raises(ValueError):
        choose_prior_lambda(errors)


def _sample(sample_id: int, path: Path) -> InversionSample:
    return InversionSample(
        sample_id=sample_id,
        model_kind=sample_id % 4,
        phase_velocity=np.ones((4, 120), dtype=np.float32),
        valid_mask=np.ones((4, 120), dtype=np.bool_),
        source_path=path,
        source_shard_id=0,
        source_row=0,
    )


def _run(values: float, *, total: float = 0.3) -> InversionRun:
    return InversionRun(
        vs=np.full(20, values, dtype=np.float64),
        predicted_phase_velocity=np.ones((4, 120), dtype=np.float64),
        success=True,
        status=0,
        message="ok",
        iterations=2,
        evaluations=3,
        initial_objective=1.0,
        terms=ObjectiveTerms(total=total, data_misfit=0.2, regularization=total - 0.2),
    )


def test_select_prior_lambda_uses_only_validation_samples_and_persists_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        HybridInversionConfig(),
        output_dir=tmp_path / "results",
        prior_lambda_candidates=(0.001, 0.01, 0.1),
        validation_samples_per_kind=1,
        device="cpu",
        workers=1,
    )
    samples = [_sample(80 + index, tmp_path / "unused.h5") for index in range(4)]
    context = SimpleNamespace(
        dataset_config=DatasetConfig(),
        dataset_config_hash="a" * 64,
        dataset_manifest_sha256="b" * 64,
        forward_checkpoint_sha256="c" * 64,
        supervised=SimpleNamespace(
            checkpoint_sha256=("d" * 64, "e" * 64, "f" * 64)
        ),
        surrogate=object(),
    )
    monkeypatch.setattr(runner_module, "_validated_inputs", lambda _: context)
    monkeypatch.setattr(
        runner_module,
        "select_observation_samples_by_kind",
        lambda dataset_dir, split, per_kind: samples,
    )
    monkeypatch.setattr(
        runner_module,
        "_prepare_sample",
        lambda *args, **kwargs: SimpleNamespace(sample_id=args[0].sample_id),
    )
    errors = {0.001: 0.2, 0.01: 0.1, 0.1: 0.1}
    monkeypatch.setattr(
        runner_module,
        "_invert_prepared_hybrid",
        lambda prepared, candidate, config, **kwargs: _run(
            1.0 + errors[candidate]
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "_truth_vs",
        lambda sample: np.ones(20, dtype=np.float64),
    )

    path = select_prior_lambda(config)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["selected_prior_lambda"] == pytest.approx(0.01)
    assert payload["validation_sample_ids"] == [80, 81, 82, 83]
    assert payload["validation_sample_count"] == 4
    assert payload["candidate_mae_km_s"] == pytest.approx(
        {"0.001": 0.2, "0.01": 0.1, "0.1": 0.1}
    )


def test_tuning_reports_missing_supervised_best_before_dataset_work(
    tmp_path: Path,
) -> None:
    supervised = tmp_path / "supervised"
    supervised.mkdir()
    (supervised / "run-identity.json").write_text(
        json.dumps({"seed_ensemble": [0]}), encoding="utf-8"
    )
    config = replace(
        HybridInversionConfig(),
        supervised_dir=supervised,
        dataset_dir=tmp_path / "missing-dataset",
        output_dir=tmp_path / "results",
        device="cpu",
    )

    with pytest.raises(ValueError, match="seed-0-best.pt"):
        select_prior_lambda(config)


def test_run_hybrid_job_persists_paired_objective_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forward = tmp_path / "forward.pt"
    forward.write_bytes(b"forward")
    config = replace(
        HybridInversionConfig(),
        forward_checkpoint=forward,
        output_dir=tmp_path / "results",
        device="cpu",
        workers=1,
    )
    sample = _sample(85, tmp_path / "unused.h5")
    job = HybridJob(
        name="hybrid-test-clean-shard-00000",
        split="test",
        noise="clean",
        samples=(sample,),
    )
    manifest = initialize_hybrid_manifest(
        config.output_dir,
        split="test",
        dataset_config_hash="a" * 64,
        dataset_manifest_sha256="b" * 64,
        forward_checkpoint=forward,
        supervised_checkpoint_sha256=("c" * 64,),
        config=config,
        selected_prior_lambda=0.1,
        expected_sample_ids_by_job={
            job.name: np.array([85], dtype=np.uint64),
        },
    )
    reference = ReferenceModel(
        vs=np.ones(20), lower=np.full(20, 0.3), upper=np.full(20, 2.6)
    )
    learning_prior = LearningPrior(
        vs=np.full(20, 1.2),
        sensitivity=np.full(20, 0.1),
        weights=np.ones(20),
    )
    prepared = PreparedHybridSample(
        observed=np.ones((4, 120)),
        valid_mask=np.ones((4, 120), dtype=np.bool_),
        reference=reference,
        learning_prior=learning_prior,
    )
    details = HybridObjectiveTerms(
        total=0.4,
        data_misfit=0.2,
        smoothness_regularization=0.05,
        learning_prior_regularization=0.15,
    )
    monkeypatch.setattr(runner_module, "_load_predictors", lambda config: (object(), object()))
    monkeypatch.setattr(runner_module, "_prepare_sample", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        runner_module,
        "_invert_prepared_pair",
        lambda *args, **kwargs: HybridPairOutcome(
            control=_run(1.1, total=0.25),
            hybrid=_run(1.15, total=0.4),
            hybrid_terms=details,
        ),
    )

    run_hybrid_job(job, config, DatasetConfig(), manifest)

    batch = validate_hybrid_result_shard(
        config.output_dir / f"{job.name}.h5", manifest=manifest
    )
    np.testing.assert_allclose(batch.control_vs, 1.1)
    np.testing.assert_allclose(batch.hybrid_vs, 1.15)
    assert batch.hybrid_learning_prior[0] == pytest.approx(0.15)
