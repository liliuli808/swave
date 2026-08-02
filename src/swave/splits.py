from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Split = Literal["train", "validation", "test", "inversion"]
SPLIT_POLICY = "mod100-v2-80-5-5-10"


def split_for_sample_id(sample_id: int) -> Split:
    if sample_id < 0:
        raise ValueError("sample_id must be nonnegative")
    remainder = sample_id % 100
    if remainder < 80:
        return "train"
    if remainder < 85:
        return "validation"
    if remainder < 90:
        return "test"
    return "inversion"


def mask_for_split(sample_ids: ArrayLike, split: Split) -> NDArray[np.bool_]:
    if split not in {"train", "validation", "test", "inversion"}:
        raise ValueError("split must be train, validation, test, or inversion")
    values = np.asarray(sample_ids)
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("sample_ids must be a nonnegative one-dimensional array")
    remainder = values % 100
    bounds = {
        "train": (0, 80),
        "validation": (80, 85),
        "test": (85, 90),
        "inversion": (90, 100),
    }
    lower, upper = bounds[split]
    return np.asarray((remainder >= lower) & (remainder < upper), dtype=np.bool_)
