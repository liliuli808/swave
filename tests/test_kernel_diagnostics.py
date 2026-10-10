from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from swave.kernel_diagnostics import summarize_history, summarize_kernel_errors
from swave.network import FourHeadForwardModel
from swave.splits import SPLIT_POLICY


def _diagnostic_cli():
    script = Path(__file__).resolve().parents[1] / "scripts/diag_kernel_failures.py"
    spec = importlib.util.spec_from_file_location("kernel_diagnostic_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_cells_do_not_dilute_kind_failure_rates_and_boundary_fails():
    error = np.array([[[0.06, 100.0]], [[0.01, 0.05]]])
    mask = np.array([[[True, False]], [[True, True]]])
    report = summarize_kernel_errors(
        error, np.ones_like(error), mask,
        np.array([100, 200]), np.array([1, 1]), np.array([0.5, 1.0]),
    )
    assert report["overall"]["rows"] == 3
    assert report["overall"]["failed_rows"] == 2
    group = report["by_mode"]["0"]["by_kind"]["low velocity"]
    assert group["rows_within_5pct"] == pytest.approx(1 / 3)
    assert group["failed_rows"] == 2
    assert report["by_mode"]["0"]["by_frequency"][1]["rows"] == 1
    assert report["overall"]["error_band_counts"]["5_to_10pct"] == 2
    assert report["worst_rows"][0]["sample_id"] == 100
    assert report["worst_rows"][0]["frequency_hz"] == 0.5
    json.dumps(report, allow_nan=False)


def test_empty_mode_and_nonfinite_predictions_are_reported_without_nan_json():
    error = np.array([[[np.nan, np.inf], [0.0, 0.0]]])
    mask = np.array([[[True, True], [False, False]]])
    report = summarize_kernel_errors(
        error, np.ones_like(error), mask, np.array([1]), np.array([0]),
        np.array([0.5, 1.0]),
    )
    assert report["overall"]["failed_rows"] == 2
    assert report["overall"]["nonfinite_rows"] == 2
    assert report["overall"]["finite_error_quantiles"]["p99"] is None
    assert report["by_mode"]["1"]["rows_within_5pct"] is None
    assert all(row["relative_l2"] is None for row in report["worst_rows"])
    json.dumps(report, allow_nan=False)


def test_history_distinguishes_score_winner_from_feasible_kernel_winner():
    records = []
    for epoch, curves, kernels in [(-1, 0.995, 0.974), (0, 0.995, 0.975),
                                   (1, 0.998, 0.974), (2, 0.98, 1.0)]:
        records.append({
            "epoch": epoch, "score": 2 - curves - kernels,
            "validation": {
                "value": {"samples_all_within_1pct": curves},
                "kernel": {"rows_within_5pct": kernels},
            },
        })
    del records[0]["score"]  # legacy initial record
    result = summarize_history(records)
    assert result["training_epochs"] == 3
    assert result["initial"]["score"] == pytest.approx(0.031)
    assert result["best_score"]["epoch"] == 2
    assert result["best_kernel_with_curves_at_least_99pct"]["epoch"] == 0


def test_diagnostic_cli_defaults_to_validation_and_evaluates_real_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _diagnostic_cli()
    model = FourHeadForwardModel(width=8, blocks=1)
    checkpoint = tmp_path / "base.pt"
    torch.save({
        "model": model.state_dict(), "architecture": {"width": 8, "blocks": 1},
        "input_mean": np.zeros(20), "input_std": np.ones(20),
        "target_mean": np.ones((4, 1)), "target_std": np.full((4, 1), 0.1),
        "split_policy": SPLIT_POLICY, "epoch": 12,
    }, checkpoint)
    rows = {
        "sample_id": np.array([80, 180]), "model_kind": np.array([0, 1]),
        "vs": np.ones((2, 20), dtype=np.float32),
        "phase_velocity": np.ones((2, 4, 120), dtype=np.float32),
        "valid_mask": np.ones((2, 4, 120), dtype=bool),
        "kernel": np.zeros((2, 4, 120, 20), dtype=np.float16),
        "kernel_mask": np.ones((2, 4, 120), dtype=bool),
    }
    splits = []

    def load(directory, split, *args):
        splits.append(split)
        return rows

    monkeypatch.setattr(module, "load_split_rows", load)
    monkeypatch.setattr(module, "load_kernel_rows", load)
    output = tmp_path / "diagnostics.json"
    monkeypatch.setattr(sys, "argv", [module.__file__, "--checkpoint", str(checkpoint),
                                   "--output", str(output), "--threads", "1"])
    assert module.main() == 0
    report = json.loads(output.read_text())
    assert splits == ["validation", "validation"]
    assert report["split"] == "validation"
    assert report["checkpoint_epoch"] == 12
    assert report["overall"]["rows"] == 960


def test_run_directory_selection_cannot_pick_an_unsaved_historical_winner(tmp_path):
    module = _diagnostic_cli()
    records = []
    for epoch, curves, kernels in [(0, 0.995, 0.980), (1, 0.998, 0.974),
                                   (2, 0.995, 0.976)]:
        records.append({
            "epoch": epoch, "score": 2 - curves - kernels,
            "validation": {
                "value": {"samples_all_within_1pct": curves},
                "kernel": {"rows_within_5pct": kernels},
            },
        })
    (tmp_path / "history.json").write_text(json.dumps({"epochs": records}))
    torch.save({"epoch": 1}, tmp_path / "best.pt")
    torch.save({"epoch": 2}, tmp_path / "last.pt")
    selected, candidates = module.select_checkpoint(tmp_path)
    assert selected == tmp_path / "last.pt"
    assert {row["epoch"] for row in candidates} == {1, 2}
    assert all(row["eligible"] for row in candidates)
