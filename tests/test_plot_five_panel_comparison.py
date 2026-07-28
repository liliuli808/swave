from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


def _load_script():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "plot_five_panel_comparison.py"
    )
    specification = importlib.util.spec_from_file_location(
        "plot_five_panel_comparison", script
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_plot_comparison_writes_four_mode_figure(tmp_path: Path) -> None:
    module = _load_script()
    frequencies = np.arange(0.5, 60.0 + 0.25, 0.5)
    true_curves = np.stack(
        [
            0.35 + 0.02 * mode + 0.1 / (frequencies + 1.0)
            for mode in range(4)
        ]
    )
    predicted_curves = true_curves + 0.005
    valid_mask = np.ones_like(true_curves, dtype=np.bool_)
    output = tmp_path / "comparison.png"

    metrics = module.plot_comparison(
        vs=np.linspace(0.4, 2.0, 20),
        true_curves=true_curves,
        predicted_curves=predicted_curves,
        valid_mask=valid_mask,
        frequencies=frequencies,
        sample_id=95,
        model_kind="COUPLED",
        output=output,
        thickness_km=0.1,
        anomaly_first_layer=3,
        anomaly_last_layer=12,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert len(metrics) == 4
    assert all(metric["mae_m_s"] == 5.0 for metric in metrics)


def test_load_sample_selects_coupled_test_row_or_requested_id(
    tmp_path: Path,
) -> None:
    module = _load_script()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    with h5py.File(dataset_dir / "shard-00000.h5", "w") as handle:
        handle.create_dataset("sample_id", data=[94, 95, 96], dtype="u8")
        handle.create_dataset("model_kind", data=[3, 0, 3], dtype="u1")
        handle.create_dataset(
            "vs", data=np.ones((3, 20), dtype=np.float32), dtype="f4"
        )
        handle.create_dataset(
            "phase_velocity",
            data=np.ones((3, 4, 120), dtype=np.float32),
            dtype="f4",
        )
        handle.create_dataset(
            "valid_mask",
            data=np.ones((3, 4, 120), dtype=np.bool_),
            dtype="?",
        )

    automatic = module._load_sample(dataset_dir, None)
    requested = module._load_sample(dataset_dir, 94)

    assert automatic[:2] == (96, "COUPLED")
    assert requested[:2] == (94, "COUPLED")


def test_validate_inputs_rejects_checkpoint_from_other_dataset(
    tmp_path: Path,
) -> None:
    module = _load_script()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    shard_path = dataset_dir / "shard-00000.h5"
    config_hash = "a" * 64
    with h5py.File(shard_path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["shard_id"] = 0
        handle.attrs["config_hash"] = config_hash
    checksum = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_hash": config_hash,
                "global_seed": 20260727,
                "expected_shards": 1,
                "completed_shards": [0],
                "accepted_by_kind": {},
                "rejected_by_kind": {},
                "rejected_by_reason": {},
                "recovered_models": 0,
                "package_version": "0.1.0",
                "created_at": "2026-07-28T00:00:00+00:00",
                "shard_sha256": {"0": checksum},
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best.pt"
    torch.save({"dataset_config_hash": "b" * 64}, checkpoint)

    with pytest.raises(ValueError, match="checkpoint dataset"):
        module._validate_inputs(dataset_dir, checkpoint)


def test_argument_parser_defaults_to_automatic_device() -> None:
    module = _load_script()

    arguments = module._parse_arguments([])

    assert arguments.device == "auto"
