from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from swave import kernel_training
from swave.kernel_training import (
    KernelTrainingConfig,
    kernel_checkpoint_key,
    select_kernel_fraction,
    train_with_kernels,
)
from swave.network import FourHeadForwardModel
from swave.splits import SPLIT_POLICY


@pytest.fixture
def kernel_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(7)
    model = FourHeadForwardModel(width=8, blocks=1)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    base = tmp_path / "base.pt"
    torch.save({
        "model": state,
        "architecture": {"width": 8, "blocks": 1},
        "input_mean": np.zeros(20, dtype=np.float32),
        "input_std": np.ones(20, dtype=np.float32),
        "target_mean": np.ones((4, 1), dtype=np.float32),
        "target_std": np.full((4, 1), 0.1, dtype=np.float32),
        "split_policy": SPLIT_POLICY,
    }, base)
    rows = {
        "sample_id": np.arange(4, dtype=np.uint64),
        "vs": np.random.default_rng(0).uniform(0.4, 2.0, (4, 20)).astype(np.float32),
        "phase_velocity": np.ones((4, 4, 120), dtype=np.float32),
        "valid_mask": np.ones((4, 4, 120), dtype=bool),
        "kernel": np.zeros((4, 4, 120, 20), dtype=np.float16),
        "kernel_mask": np.ones((4, 4, 120), dtype=bool),
    }
    monkeypatch.setattr(kernel_training, "load_split_rows", lambda *args: rows)
    monkeypatch.setattr(kernel_training, "load_kernel_rows", lambda *args: rows)
    evaluations = []

    def evaluate(forward, unused_rows):
        evaluations.append(forward(torch.as_tensor(rows["vs"])).detach())
        # Force every fine-tuned epoch to lose to the initial checkpoint.
        rate = 1.0 if len(evaluations) == 1 else 0.5
        return {
            "value": {"samples_all_within_1pct": rate},
            "kernel": {"rows_within_5pct": rate, "median_relative_l2": 1 - rate},
        }

    monkeypatch.setattr(kernel_training, "evaluate_rows", evaluate)
    config = KernelTrainingConfig(
        base_checkpoint=base, dataset_dir=tmp_path, kernel_dir=tmp_path,
        output_dir=tmp_path / "run", cache_dir=tmp_path / "cache",
        width=8, blocks=1, epochs=1, steps_per_epoch=1,
        batch_size=2, kernel_batch_size=2, threads=1,
    )
    return config, state, evaluations


def test_matching_checkpoint_is_loaded_and_initial_best_is_preserved(kernel_run):
    config, state, evaluations = kernel_run
    best_path = train_with_kernels(config)
    saved = torch.load(best_path, weights_only=False)
    assert saved["epoch"] == -1
    assert saved["best_score"] == 0.0
    for key, value in state.items():
        torch.testing.assert_close(saved["model"][key], value, rtol=0, atol=0)
    model = FourHeadForwardModel(width=8, blocks=1)
    model.load_state_dict(state)
    rows = kernel_training.load_split_rows(None, None, None)
    torch.testing.assert_close(
        evaluations[0], model(torch.as_tensor(rows["vs"])) * 0.1 + 1,
    )
    last = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert last["epoch"] == 0
    assert any(not torch.equal(last["model"][k], value) for k, value in state.items())
    record = json.loads((config.output_dir / "history.json").read_text())["epochs"][-1]
    assert np.isfinite(record["grad_norm_max"])
    assert 0 <= record["grad_clip_fraction"] <= 1


def test_resuming_keeps_the_last_completed_history_epoch(kernel_run):
    config, _, _ = kernel_run
    train_with_kernels(config)
    config.epochs = 2
    train_with_kernels(config)
    history = json.loads((config.output_dir / "history.json").read_text())["epochs"]
    assert [record["epoch"] for record in history] == [-1, 0, 1]


def test_nonfinite_loss_stops_before_writing_a_bad_checkpoint(
    kernel_run, monkeypatch: pytest.MonkeyPatch
):
    config, _, _ = kernel_run
    monkeypatch.setattr(
        kernel_training, "relative_value_loss",
        lambda prediction, *args: prediction.sum() * float("nan"),
    )
    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        train_with_kernels(config)
    assert (config.output_dir / "best.pt").exists()
    assert not (config.output_dir / "last.pt").exists()


def test_required_warm_start_rejects_architecture_mismatch_before_loading_data(
    kernel_run, monkeypatch: pytest.MonkeyPatch
):
    config, _, evaluations = kernel_run
    config.require_warm_start = True
    config.width = 16

    def unexpected_load(*args):
        pytest.fail("architecture mismatch must fail before loading the dataset")

    monkeypatch.setattr(kernel_training, "load_split_rows", unexpected_load)
    with pytest.raises(ValueError, match="checkpoint architecture does not match"):
        train_with_kernels(config)
    assert not evaluations
    assert not config.output_dir.exists()


def test_required_warm_start_refuses_to_resume_an_existing_run(kernel_run):
    config, _, _ = kernel_run
    train_with_kernels(config)
    config.require_warm_start = True
    config.epochs = 2
    with pytest.raises(ValueError, match="refuses automatic resume"):
        train_with_kernels(config)
    saved = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert saved["epoch"] == 0


@pytest.mark.parametrize("rate", [0.0, float("nan")])
def test_initial_score_guard_rejects_bad_start_before_any_training(
    kernel_run, monkeypatch: pytest.MonkeyPatch, rate: float
):
    config, _, _ = kernel_run
    config.require_warm_start = True
    config.max_initial_score = 0.05
    monkeypatch.setattr(kernel_training, "evaluate_rows", lambda *args: {
        "value": {"samples_all_within_1pct": rate},
        "kernel": {"rows_within_5pct": rate, "median_relative_l2": 1 - rate},
    })
    with pytest.raises(ValueError, match="initial validation score"):
        train_with_kernels(config)
    assert not (config.output_dir / "best.pt").exists()
    assert not (config.output_dir / "last.pt").exists()


def test_required_warm_start_accepts_good_checkpoint_and_reports_source(
    kernel_run, capsys: pytest.CaptureFixture
):
    config, _, _ = kernel_run
    config.require_warm_start = True
    config.max_initial_score = 0.05
    train_with_kernels(config)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["initialization"] == "base_checkpoint"
    assert records[0]["resume_checkpoint"] is None
    assert records[0]["kernel_training_source"] == str(
        Path(kernel_training.__file__).resolve()
    )
    assert records[1]["epoch"] == -1
    assert records[1]["samples_all_within_1pct"] == 1.0
    assert records[1]["kernel_rows_within_5pct"] == 1.0


def _metrics(curves: float, kernels: float) -> dict:
    return {
        "value": {"samples_all_within_1pct": curves},
        "kernel": {"rows_within_5pct": kernels, "median_relative_l2": 0.002},
    }


def test_kernel_selection_keeps_curve_constraint_and_does_not_trade_off_kernels(
    kernel_run, monkeypatch: pytest.MonkeyPatch
):
    config, _, _ = kernel_run
    config.epochs = 3
    metrics = iter([
        _metrics(0.9951, 0.9746),
        _metrics(0.9953, 0.9749),  # best feasible kernels
        _metrics(0.9960, 0.9748),  # better sum, worse kernels
        _metrics(0.9899, 0.9990),  # smallest sum, but curves miss acceptance
    ])
    monkeypatch.setattr(kernel_training, "evaluate_rows", lambda *args: next(metrics))
    path = train_with_kernels(config)
    assert path.name == "best-kernel.pt"
    selected = torch.load(path, weights_only=False)
    assert selected["epoch"] == 0
    assert selected["validation"]["kernel"]["rows_within_5pct"] == 0.9749
    assert selected["selection"]["minimum_curve_pass_rate"] == 0.99
    legacy = torch.load(config.output_dir / "best.pt", weights_only=False)
    assert legacy["epoch"] == 2

    # A later resume with only weaker kernels must retain the saved winner.
    config.epochs = 4
    monkeypatch.setattr(kernel_training, "evaluate_rows",
                        lambda *args: _metrics(0.997, 0.9748))
    resumed = train_with_kernels(config)
    assert torch.load(resumed, weights_only=False)["epoch"] == 0


def test_legacy_resume_recovers_only_checkpoints_whose_weights_still_exist(
    kernel_run, monkeypatch: pytest.MonkeyPatch
):
    config, _, _ = kernel_run
    config.epochs = 2
    metrics = iter([
        _metrics(0.9951, 0.9746), _metrics(0.9953, 0.9749),
        _metrics(0.9960, 0.9748),
    ])
    monkeypatch.setattr(kernel_training, "evaluate_rows", lambda *args: next(metrics))
    train_with_kernels(config)
    (config.output_dir / "best-kernel.pt").unlink()
    for filename in ("best.pt", "last.pt"):
        path = config.output_dir / filename
        saved = torch.load(path, weights_only=False)
        del saved["validation"]
        torch.save(saved, path)

    config.epochs = 3
    monkeypatch.setattr(kernel_training, "evaluate_rows",
                        lambda *args: _metrics(0.997, 0.9747))
    selected = torch.load(train_with_kernels(config), weights_only=False)
    # Epoch 0 was better, but legacy best.pt and last.pt both contain epoch 1.
    assert selected["epoch"] == 1
    assert selected["validation"]["kernel"]["rows_within_5pct"] == 0.9748


@pytest.mark.parametrize("curves,kernels", [(0.9899, 1), (float("nan"), 1),
                                             (1, float("nan")), (1, float("inf"))])
def test_kernel_selection_rejects_infeasible_or_nonfinite_metrics(curves, kernels):
    assert kernel_checkpoint_key(_metrics(curves, kernels)) is None


def test_kernel_selection_accepts_boundary_and_breaks_exact_ties_by_curves():
    assert kernel_checkpoint_key(_metrics(0.99, 0.97)) == (0.97, 0.99)
    assert (kernel_checkpoint_key(_metrics(0.995, 0.97))
            > kernel_checkpoint_key(_metrics(0.99, 0.97)))


def test_row_mining_training_refreshes_its_pool_and_reports_the_extra_objective(
    kernel_run, capsys: pytest.CaptureFixture
):
    config, state, _ = kernel_run
    config.epochs = 3
    config.kernel_row_weight = 0.1
    config.kernel_row_batch_size = 2
    config.kernel_mining_samples = 3
    config.kernel_mining_interval = 2
    train_with_kernels(config)
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    refreshes = [row["kernel_row_mining"] for row in output if "kernel_row_mining" in row]
    assert [row["epoch"] for row in refreshes] == [0, 2]
    assert all(row["split"] == "train" and row["pool_models"] == 3 for row in refreshes)
    epochs = json.loads((config.output_dir / "history.json").read_text())["epochs"][1:]
    assert all(row["mined_rows"] > 0 for row in epochs)
    assert all(np.isfinite(row["kernel_row_loss"]) for row in epochs)
    last = torch.load(config.output_dir / "last.pt", weights_only=False)
    assert any(not torch.equal(last["model"][key], value) for key, value in state.items())


def test_row_mining_refuses_validation_as_a_training_split(kernel_run):
    config, _, _ = kernel_run
    config.kernel_row_weight = 0.1
    config.kernel_train_split = "validation"
    with pytest.raises(ValueError, match="requires kernel_train_split='train'"):
        train_with_kernels(config)
    assert not config.output_dir.exists()


def test_row_mining_checks_sample_ids_even_when_the_directory_is_named_train(kernel_run):
    config, _, _ = kernel_run
    config.kernel_row_weight = 0.1
    rows = kernel_training.load_kernel_rows(None, None)
    rows["sample_id"][0] = 80  # validation under the fixed split policy
    with pytest.raises(ValueError, match="non-training sample IDs"):
        train_with_kernels(config)
    assert not config.output_dir.exists()


def _subset_rows(count: int) -> dict[str, np.ndarray]:
    ids = np.arange(100, 100 + count, dtype=np.uint64)[::-1].copy()
    return {"sample_id": ids, "vs": ids.astype(np.float32)[:, None].repeat(20, 1)}


def test_kernel_fraction_is_deterministic_nested_and_keeps_rows_aligned():
    rows = _subset_rows(40)
    half = select_kernel_fraction(rows, 0.5, seed=3)
    quarter = select_kernel_fraction(rows, 0.25, seed=3)
    assert len(half["sample_id"]) == 20 and len(quarter["sample_id"]) == 10
    assert set(quarter["sample_id"].tolist()) <= set(half["sample_id"].tolist())
    np.testing.assert_array_equal(
        half["sample_id"], select_kernel_fraction(rows, 0.5, seed=3)["sample_id"])
    # Selection depends on sample IDs, not on the order shards were read in.
    shuffled = {key: value[::-1].copy() for key, value in rows.items()}
    assert (set(select_kernel_fraction(shuffled, 0.5, seed=3)["sample_id"].tolist())
            == set(half["sample_id"].tolist()))
    np.testing.assert_array_equal(half["vs"][:, 0], half["sample_id"].astype(np.float32))
    assert len(rows["sample_id"]) == 40
    assert select_kernel_fraction(rows, 1.0, seed=3) is rows


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5, float("nan")])
def test_kernel_fraction_must_be_in_unit_interval(fraction):
    with pytest.raises(ValueError, match="kernel_train_fraction"):
        KernelTrainingConfig(
            base_checkpoint=Path("b"), dataset_dir=Path("d"), kernel_dir=Path("k"),
            output_dir=Path("o"), cache_dir=Path("c"), kernel_train_fraction=fraction,
        )


def test_training_uses_and_reports_the_kernel_subset(
    kernel_run, capsys: pytest.CaptureFixture
):
    config, _, _ = kernel_run
    config.kernel_train_fraction = 0.5
    train_with_kernels(config)
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()
              if line.startswith("{")]
    subset = next(row["kernel_train_subset"] for row in output
                  if "kernel_train_subset" in row)
    assert subset["fraction"] == 0.5
    assert subset["models"] == 2 and subset["available_models"] == 4
    assert len(subset["sample_id_checksum"]) == 16
