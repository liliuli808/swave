from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from swave.config import DatasetConfig, GeologyConfig
from swave.dataset import generate_dataset
from swave.dataset_audit import (
    audit_dataset,
    check_geology_rule,
    summarize_duplicate_digests,
)
from swave.geology import ModelKind, generate_model


def test_duplicate_summary_counts_rows_and_cross_split_groups() -> None:
    digests = np.asarray(
        [b"a" * 32, b"a" * 32, b"b" * 32, b"b" * 32, b"c" * 32],
        dtype="S32",
    )
    sample_ids = np.asarray([0, 85, 1, 2, 3], dtype=np.uint64)

    summary = summarize_duplicate_digests(digests, sample_ids)

    assert summary == {
        "duplicate_groups": 2,
        "duplicate_rows": 4,
        "extra_duplicate_rows": 2,
        "cross_split_groups": 1,
        "examples": [[0, 85], [1, 2]],
    }


def test_duplicate_summary_confirms_raw_bytes_after_digest_match() -> None:
    digests = np.asarray([b"a" * 32] * 3, dtype="S32")
    sample_ids = np.asarray([0, 85, 1], dtype=np.uint64)

    summary = summarize_duplicate_digests(
        digests,
        sample_ids,
        raw_rows={0: b"same", 85: b"same", 1: b"collision"},
    )

    assert summary == {
        "duplicate_groups": 1,
        "duplicate_rows": 2,
        "extra_duplicate_rows": 1,
        "cross_split_groups": 1,
        "examples": [[0, 85]],
    }


def test_generated_models_satisfy_declared_family_rules() -> None:
    config = GeologyConfig()
    seen: Counter[ModelKind] = Counter()
    for sample_id in range(200):
        generated = generate_model(sample_id, config, global_seed=20260727)
        result = check_geology_rule(
            generated.kind,
            generated.vs,
            generated.background_vs,
            config,
        )
        assert result["valid"], (sample_id, generated.kind, result)
        seen[generated.kind] += 1

    assert set(seen) == set(ModelKind)


def test_geology_rule_rejects_normal_velocity_reversal() -> None:
    config = GeologyConfig()
    background = np.linspace(0.4, 2.0, 20)
    values = background.copy()
    values[8] = values[7] - 0.1

    result = check_geology_rule(
        ModelKind.NORMAL,
        values,
        background,
        config,
    )

    assert not result["valid"]
    assert result["reason"] == "normal_not_nondecreasing"


def test_audit_accepts_a_complete_deterministic_dataset(tmp_path: Path) -> None:
    config = replace(
        DatasetConfig(),
        samples=4,
        shard_size=2,
        workers=1,
        output_dir=tmp_path / "dataset",
    )
    generate_dataset(config)

    report = audit_dataset(config.output_dir, config)

    assert report["identity"]["sample_count"] == 4
    assert report["identity"]["sample_ids_unique"]
    assert report["identity"]["sample_ids_contiguous"]
    assert report["duplicates"]["vs"]["duplicate_rows"] == 0
    assert report["duplicates"]["full_record"]["duplicate_rows"] == 0
    assert report["geology"]["violations"] == 0
    assert sum(report["geology"]["by_kind"].values()) == 4
