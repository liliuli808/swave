from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from swave.kernel_training import Normalizer, full_jacobian
from swave.network import FourHeadForwardModel
from swave.splits import SPLIT_POLICY


def _cli():
    script = Path(__file__).resolve().parents[1] / "scripts/ensemble_kernel_check.py"
    spec = importlib.util.spec_from_file_location("ensemble_kernel_check", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(path: Path, seed: int) -> dict:
    torch.manual_seed(seed)
    payload = {
        "model": FourHeadForwardModel(width=8, blocks=1).state_dict(),
        "architecture": {"width": 8, "blocks": 1},
        "input_mean": np.zeros(20, dtype=np.float32),
        "input_std": np.ones(20, dtype=np.float32),
        "target_mean": np.ones((4, 1), dtype=np.float32),
        "target_std": np.full((4, 1), 0.1, dtype=np.float32),
        "split_policy": SPLIT_POLICY, "epoch": seed,
    }
    torch.save(payload, path)
    return payload


def test_shared_failures_counts_members_and_treats_exact_threshold_as_failure():
    module = _cli()
    mask = np.array([[[True, True, True, False]]])
    first = np.array([[[0.01, 0.05, 0.20, 9.0]]])
    second = np.array([[[0.01, 0.01, 0.30, 9.0]]])
    result = module.shared_failures([first, second], mask)
    assert result["rows"] == 3
    assert result["rows_failed_by_n_members"] == {"0": 1, "1": 1, "2": 1}
    assert result["failed_by_any"] == 2 and result["failed_by_all"] == 1
    assert result["fraction_of_any_failed_by_all"] == pytest.approx(0.5)


def _run(module, tmp_path, monkeypatch, checkpoints, kernel):
    rows = {
        "sample_id": np.array([3, 8], dtype=np.uint64), "model_kind": np.array([1, 3]),
        "vs": np.linspace(0.5, 2.0, 40, dtype=np.float32).reshape(2, 20),
        "phase_velocity": np.ones((2, 4, 120), dtype=np.float32),
        "valid_mask": np.ones((2, 4, 120), dtype=bool),
        "kernel": kernel.astype(np.float16),
        "kernel_mask": np.ones((2, 4, 120), dtype=bool),
    }
    monkeypatch.setattr(module, "load_split_rows", lambda *args: rows)
    monkeypatch.setattr(module, "load_kernel_rows", lambda *args: rows)
    output = tmp_path / "ensemble.json"
    argv = [module.__file__, "--output", str(output), "--threads", "1"]
    for path in checkpoints:
        argv += ["--checkpoint", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    return json.loads(output.read_text()), rows


def test_identical_members_give_the_member_result(tmp_path, monkeypatch):
    module = _cli()
    _checkpoint(tmp_path / "a.pt", 1)
    report, _ = _run(module, tmp_path, monkeypatch, [tmp_path / "a.pt"] * 2,
                     np.ones((2, 4, 120, 20)))
    member = report["members"][0]["kernel"]["overall"]
    assert report["ensemble"]["kernel"]["overall"] == member
    overlap = report["shared_failures"]
    assert overlap["failed_by_any"] == overlap["failed_by_all"]


def test_ensemble_kernel_is_the_jacobian_of_the_averaged_network(tmp_path, monkeypatch):
    module = _cli()
    payloads = [_checkpoint(tmp_path / f"{seed}.pt", seed) for seed in (1, 2)]
    vs = torch.as_tensor(np.linspace(0.5, 2.0, 40, dtype=np.float32).reshape(2, 20))
    forwards = []
    for payload in payloads:
        model = FourHeadForwardModel(width=8, blocks=1)
        model.load_state_dict(payload["model"])
        model.eval()
        forwards.append(Normalizer.from_payload(payload).physical(model))
    averaged = full_jacobian(lambda x: (forwards[0](x) + forwards[1](x)) / 2, vs)
    # Labels equal to the averaged network's kernel: the ensemble must pass every row.
    report, _ = _run(module, tmp_path, monkeypatch,
                     [tmp_path / "1.pt", tmp_path / "2.pt"], averaged.detach().numpy())
    ensemble = report["ensemble"]["kernel"]["overall"]
    assert ensemble["rows_within_5pct"] == 1.0
    assert ensemble["finite_error_quantiles"]["max"] < 2e-3  # float16 label rounding
