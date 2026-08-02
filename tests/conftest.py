from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from swave.network import FourHeadForwardModel
from swave.splits import SPLIT_POLICY


@pytest.fixture
def tiny_complete_dataset(tmp_path: Path) -> Path:
    """Write a complete, checksummed dataset containing four inversion rows."""
    directory = tmp_path / "dataset"
    directory.mkdir()
    shard_path = directory / "shard-00000.h5"
    sample_ids = np.array([89, 90, 91, 92, 93], dtype=np.uint64)
    model_kinds = np.array([0, 0, 1, 2, 3], dtype=np.uint8)
    vs = np.linspace(0.4, 2.0, 100, dtype=np.float32).reshape(5, 20)
    vp = vs + np.float32(0.5)
    density = np.full((5, 20), 2.0, dtype=np.float32)
    phase_velocity = np.broadcast_to(
        np.arange(600, dtype=np.float32).reshape(5, 4, 30).repeat(4, axis=2),
        (5, 4, 120),
    ).copy()
    valid_mask = np.ones((5, 4, 120), dtype=np.bool_)
    with h5py.File(shard_path, "w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["config_hash"] = "tiny-complete-fixture"
        handle.attrs["shard_id"] = 0
        handle.attrs["first_sample_id"] = 89
        handle.attrs["last_sample_id"] = 93
        handle.attrs["accepted_count"] = 5
        handle.attrs["sample_id_sha256"] = hashlib.sha256(
            sample_ids.tobytes()
        ).hexdigest()
        handle.attrs["accepted_by_kind"] = json.dumps(
            {"COUPLED": 1, "HIGH_VELOCITY": 1, "LOW_VELOCITY": 1, "NORMAL": 2},
            sort_keys=True,
        )
        handle.attrs["rejected_by_kind"] = "{}"
        handle.attrs["rejected_by_reason"] = "{}"
        handle.attrs["recovered_models"] = 0
        handle.create_dataset("sample_id", data=sample_ids, dtype="u8")
        handle.create_dataset("model_kind", data=model_kinds, dtype="u1")
        handle.create_dataset("vs", data=vs, dtype="f4")
        handle.create_dataset("vp", data=vp, dtype="f4")
        handle.create_dataset("density", data=density, dtype="f4")
        handle.create_dataset("phase_velocity", data=phase_velocity, dtype="f4")
        handle.create_dataset("valid_mask", data=valid_mask, dtype="?")
        handle.create_dataset("quality_flags", data=np.zeros(5, dtype=np.uint16))
        handle.create_dataset("retry_count", data=np.zeros(5, dtype=np.uint8))

    shard_sha256 = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_hash": "tiny-complete-fixture",
                "global_seed": 20260727,
                "expected_shards": 1,
                "completed_shards": [0],
                "accepted_by_kind": {
                    "COUPLED": 1,
                    "HIGH_VELOCITY": 1,
                    "LOW_VELOCITY": 1,
                    "NORMAL": 2,
                },
                "rejected_by_kind": {},
                "rejected_by_reason": {},
                "recovered_models": 0,
                "package_version": "0.1.0",
                "created_at": "2026-08-02T00:00:00+00:00",
                "shard_sha256": {"0": shard_sha256},
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def tiny_checkpoint(tiny_complete_dataset: Path, tmp_path: Path) -> Path:
    """Write a predictor checkpoint compatible with ``tiny_complete_dataset``."""
    checkpoint = tmp_path / "tiny.pt"
    torch.save(
        {
            "model": FourHeadForwardModel().state_dict(),
            "input_mean": np.zeros(20, dtype=np.float32),
            "input_std": np.ones(20, dtype=np.float32),
            "target_mean": np.zeros((4, 1), dtype=np.float32),
            "target_std": np.ones((4, 1), dtype=np.float32),
            "dataset_config_hash": "tiny-complete-fixture",
            "split_policy": SPLIT_POLICY,
        },
        checkpoint,
    )
    return checkpoint
