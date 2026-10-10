from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from swave import kernel_audit, kernel_training
from swave.config import PhysicsConfig
from swave.kernel_audit import (
    audit_kernel_rows,
    kernel_model_metadata,
    select_audit_rows,
    summarize_audit,
)
from swave.kernel_training import load_kernel_rows
from swave.kernels import sensitivity_kernels
from swave.secular import LayeredModel
from swave.solver import DispersionSolver


@pytest.fixture
def audit_dataset(tmp_path):
    dataset, kernels = tmp_path / "dataset", tmp_path / "kernels"
    dataset.mkdir()
    (kernels / "validation").mkdir(parents=True)
    for part, ids in enumerate(([80, 81], [180, 181])):
        with h5py.File(dataset / f"shard-{part:05d}.h5", "w") as handle:
            handle["sample_id"] = np.array(ids[::-1], dtype=np.uint64)
            handle["model_kind"] = np.array([3, 0], dtype=np.uint8)
        with h5py.File(kernels / "validation" / f"kernels-{part:05d}.h5", "w") as handle:
            handle["sample_id"] = np.array(ids, dtype=np.uint64)
            handle["vs"] = np.ones((2, 20), dtype=np.float32)
            handle["phase_velocity"] = np.ones((2, 4, 120), dtype=np.float32)
            mask = np.ones((2, 4, 120), dtype=bool)
            mask[:, 0, 0] = False
            handle["valid_mask"] = mask
            labels = np.full((2, 4, 120, 20), 0.987654, dtype=np.float32)
            labels[:, 0, 0] = np.nan
            handle["kernel"] = labels
    return dataset, kernels


def test_subset_loading_matches_full_loader_and_applies_corrections(
    audit_dataset, tmp_path, monkeypatch,
):
    dataset, kernels = audit_dataset
    ids, kinds = kernel_model_metadata(dataset, kernels, "validation")
    np.testing.assert_array_equal(ids, [80, 81, 180, 181])
    np.testing.assert_array_equal(kinds, [0, 3, 0, 3])
    correction = tmp_path / "corrections.npz"
    np.savez(correction, sample_id=np.array([81, 180]), frequency_index=np.array([0, 1]),
             phase_velocity=np.array([[0.6, 0.7, 0.8, np.nan], [0.9, 1, 1.1, 1.2]]))

    def kernels_for_corrected_rows(vs, phase, mask, frequencies):
        return np.where(mask[..., None], np.full((*phase.shape, 20), 0.125), np.nan)

    monkeypatch.setattr(kernel_training, "sensitivity_kernels", kernels_for_corrected_rows)
    full = load_kernel_rows(kernels, "validation", correction)
    subset = load_kernel_rows(kernels, "validation", correction, sample_ids=np.array([181, 81]))
    for key in full:
        np.testing.assert_array_equal(subset[key], full[key][[1, 3]])
    assert subset["kernel"][0, 0, 0, 0] == 0.125
    assert not subset["kernel_mask"][0, 3, 0]
    with pytest.raises(ValueError, match="missing"):
        load_kernel_rows(kernels, "validation", sample_ids=np.array([999]))
    with pytest.raises(ValueError, match="unique"):
        load_kernel_rows(kernels, "validation", sample_ids=np.array([81, 81]))


def test_audit_selection_keeps_target_identity_and_excludes_invalid_rows(audit_dataset):
    _, kernels = audit_dataset
    rows = load_kernel_rows(kernels, "validation", sample_ids=np.array([81, 180]))
    targets = [{"sample_id": 180, "mode": 3, "frequency_hz": 1.5, "relative_l2": 0.5}]
    chosen = select_audit_rows(rows, targets * 2, [81, 180])
    assert chosen[0] == {"row": 1, "mode": 3, "frequency_index": 2,
                         "selection": "diagnostic_failure", "network_relative_l2": 0.5}
    assert len(chosen) == 9
    keys = {(r["row"], r["mode"], r["frequency_index"]) for r in chosen}
    assert len(keys) == len(chosen)
    assert all(rows["kernel_mask"][key] for key in keys)
    assert chosen == select_audit_rows(rows, targets, [81, 180])
    with pytest.raises(ValueError, match="invalid"):
        select_audit_rows(rows, [{"sample_id": 81, "mode": 0, "frequency_hz": 0.5}], [])


def test_metadata_rejects_a_nonvalidation_id_in_the_validation_directory(audit_dataset):
    dataset, kernels = audit_dataset
    with h5py.File(kernels / "validation/kernels-00000.h5", "r+") as handle:
        handle["sample_id"][0] = 0
    with pytest.raises(ValueError, match="requested split"):
        kernel_model_metadata(dataset, kernels, "validation")


def test_cli_rejects_mismatched_diagnostic_split_before_reading_data(tmp_path):
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(json.dumps({"split": "train", "worst_rows": []}))
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repository / "scripts/check_kernel_label_noise.py"),
         "--diagnostics", str(diagnostic), "--dataset-dir", str(tmp_path / "missing")],
        cwd=tmp_path, text=True, capture_output=True, timeout=30, check=False,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
    )
    assert result.returncode == 2
    assert "diagnostic split does not match" in result.stderr


def test_root_recheck_detects_inconsistent_roots_even_when_step_sweep_is_stable(monkeypatch):
    rows = {"sample_id": np.array([80]), "vs": np.ones((1, 20)),
            "phase_velocity": np.full((1, 1, 1), 1.2),
            "kernel": np.full((1, 1, 1, 20), 1.2)}
    chosen = [{"row": 0, "mode": 0, "frequency_index": 0, "selection": "diagnostic_failure"}]
    monkeypatch.setattr(kernel_audit, "_kernel_at", lambda vs, c, f, step: np.full(20, c))
    solution = SimpleNamespace(roots=np.array([1.0]), status=0)
    monkeypatch.setattr(kernel_audit, "DispersionSolver", lambda *args: SimpleNamespace(
        solve_frequency=lambda *args: solution,
    ))
    records = audit_kernel_rows(rows, chosen)
    checks = records[0]["checks"]
    assert checks["stored_vs_recomputed"] == 0
    assert checks["step_1e-6_at_stored_root"] == 0
    assert checks["resolved_vs_stored"] == pytest.approx(1 / 6)
    assert records[0]["phase_relative_shift"] == pytest.approx(1 / 6)
    solution.status = 1
    unresolved = audit_kernel_rows(rows, chosen)
    summary = summarize_audit(unresolved)["diagnostic_failure"]["checks"]["resolved_vs_stored"]
    assert summary["checked_rows"] == 0
    assert summary["unresolved_rows"] == 1
    assert summary["max_relative_l2"] is None
    json.dumps(unresolved, allow_nan=False)


def test_audit_checks_real_physical_rows_and_flags_a_corrupted_stored_label():
    vs = np.linspace(0.4, 2.15, 20).astype(np.float32)
    vs[5] *= 1.25
    vs[6:9] *= 0.75
    root = DispersionSolver(LayeredModel.from_vs(vs), PhysicsConfig()).solve_frequency(2.0)
    phase = np.ones((1, 4, 4), dtype=np.float32)
    phase[0, :, 3] = root.roots
    mask = np.zeros_like(phase, dtype=bool)
    mask[0, :, 3] = True
    labels = sensitivity_kernels(vs, phase, mask, [0.5, 1, 1.5, 2]).astype(np.float16)
    labels[0, 3, 3, 0] += 0.5
    rows = {"sample_id": np.array([80]), "vs": vs[None],
            "phase_velocity": phase, "kernel": labels}
    chosen = [{"row": 0, "mode": mode, "frequency_index": 3, "selection": "random_reference"}
              for mode in (0, 3)]
    records = audit_kernel_rows(rows, chosen)
    assert records[0]["checks"]["stored_vs_recomputed"] < 0.001
    assert records[0]["checks"]["resolved_vs_stored"] < 0.001
    assert records[1]["checks"]["stored_vs_recomputed"] > 0.05
    assert records[1]["checks"]["resolved_vs_stored"] > 0.05
    summary = summarize_audit(records)["random_reference"]["checks"]["stored_vs_recomputed"]
    assert summary["checked_rows"] == 2
    assert summary["rows_over_5pct"] == 1
    json.dumps(records, allow_nan=False)
