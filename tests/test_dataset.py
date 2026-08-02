from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

import swave.dataset as dataset_module
from swave.config import DatasetConfig
from swave.dataset import generate_dataset, load_manifest


def _smoke_config(output_dir: Path, samples: int = 2) -> DatasetConfig:
    return replace(
        DatasetConfig(),
        samples=samples,
        shard_size=2,
        workers=1,
        output_dir=output_dir,
    )


def test_smoke_dataset_has_declared_schema(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, samples=4)
    manifest = generate_dataset(config)
    assert manifest.complete
    assert manifest.package_version == "0.1.0"
    assert manifest.created_at
    assert manifest.rejected_by_kind == {}
    assert set(manifest.shard_sha256) == {"0", "1"}
    assert load_manifest(tmp_path / "manifest.json") == manifest
    files = sorted(tmp_path.glob("shard-*.h5"))
    assert len(files) == 2
    with h5py.File(files[0]) as handle:
        assert handle["sample_id"].shape == (2,)
        assert handle["model_kind"].dtype == np.dtype("u1")
        assert handle["vs"].shape == (2, 20)
        assert handle["vp"].shape == (2, 20)
        assert handle["density"].shape == (2, 20)
        assert handle["phase_velocity"].shape == (2, 4, 120)
        assert handle["valid_mask"].shape == (2, 4, 120)
        assert handle["quality_flags"].dtype == np.dtype("u2")
        assert handle["retry_count"].dtype == np.dtype("u1")
        assert handle.attrs["accepted_count"] == 2


def test_dataset_manifest_digest_is_canonical_and_checksum_sensitive(
    tmp_path: Path,
) -> None:
    manifest = generate_dataset(_smoke_config(tmp_path))
    first = dataset_module.dataset_manifest_sha256(manifest)
    assert first == dataset_module.dataset_manifest_sha256(
        load_manifest(tmp_path / "manifest.json")
    )
    changed = replace(
        manifest,
        shard_sha256={"0": "f" * 64},
    )
    assert dataset_module.dataset_manifest_sha256(changed) != first


def test_resume_does_not_rewrite_complete_shard(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    generate_dataset(config)
    shard = next(tmp_path.glob("shard-*.h5"))
    before = shard.read_bytes()
    generate_dataset(config)
    assert shard.read_bytes() == before


def test_conflicting_configuration_is_rejected(tmp_path: Path) -> None:
    first = _smoke_config(tmp_path)
    generate_dataset(first)
    second = replace(first, seed=first.seed + 1)
    with pytest.raises(ValueError, match="configuration hash"):
        generate_dataset(second)


def test_resume_rejects_a_shard_with_corrupt_sample_ids(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    generate_dataset(config)
    shard = next(tmp_path.glob("shard-*.h5"))
    with h5py.File(shard, "r+") as handle:
        handle["sample_id"][0] = 999
    with pytest.raises(ValueError, match="checksum|sample_id"):
        generate_dataset(config)
