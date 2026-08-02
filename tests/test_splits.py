import numpy as np

from swave.splits import SPLIT_POLICY, mask_for_split, split_for_sample_id


def test_mod100_policy_has_exact_80_5_5_10_partition() -> None:
    ids = np.arange(100, dtype=np.uint64)
    counts = {
        split: int(mask_for_split(ids, split).sum())
        for split in ("train", "validation", "test", "inversion")
    }
    assert counts == {"train": 80, "validation": 5, "test": 5, "inversion": 10}
    assert SPLIT_POLICY == "mod100-v2-80-5-5-10"


def test_split_boundaries_are_stable() -> None:
    expected = {
        79: "train",
        80: "validation",
        84: "validation",
        85: "test",
        89: "test",
        90: "inversion",
        99: "inversion",
        190: "inversion",
    }
    assert {value: split_for_sample_id(value) for value in expected} == expected
