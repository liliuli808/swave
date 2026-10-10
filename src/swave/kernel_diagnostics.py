"""Validation reports for kernel failures and checkpoint selection."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .kernel_training import kernel_checkpoint_key

KIND_NAMES = ("normal", "low velocity", "high velocity", "coupled HVL+LVL")
THRESHOLD = 0.05


def _error_stats(values: NDArray) -> dict[str, object]:
    """Use valid rows only; a nonfinite prediction always fails acceptance."""
    finite = values[np.isfinite(values)]
    passed = int((finite < THRESHOLD).sum())
    bands = (
        ("below_1pct", 0.0, 0.01), ("1_to_3pct", 0.01, 0.03),
        ("3_to_5pct", 0.03, 0.05), ("5_to_10pct", 0.05, 0.1),
        ("10_to_20pct", 0.1, 0.2), ("at_least_20pct", 0.2, math.inf),
    )
    return {
        "rows": int(values.size),
        "failed_rows": int(values.size - passed),
        "rows_within_5pct": passed / values.size if values.size else None,
        "nonfinite_rows": int(values.size - finite.size),
        "finite_error_quantiles": {
            name: float(np.quantile(finite, q)) if finite.size else None
            for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1.0))
        },
        "error_band_counts": {
            name: int(((finite >= low) & (finite < high)).sum())
            for name, low, high in bands
        },
    }


def summarize_kernel_errors(
    relative_error: NDArray,
    physical_norm: NDArray,
    mask: NDArray,
    sample_ids: NDArray,
    kinds: NDArray,
    frequencies: NDArray,
) -> dict[str, object]:
    """Group the exact row errors without treating invalid cells as successes."""
    if relative_error.shape != mask.shape or physical_norm.shape != mask.shape:
        raise ValueError("error, physical norm and mask must have matching shapes")
    if mask.ndim != 3 or not mask.any():
        raise ValueError("expected nonempty valid (sample, mode, frequency) rows")
    if len(sample_ids) != len(mask) or len(kinds) != len(mask):
        raise ValueError("sample IDs and kinds must align with the kernel rows")
    if len(frequencies) != mask.shape[2]:
        raise ValueError("frequencies must align with the kernel rows")
    if not np.isfinite(physical_norm[mask]).all():
        raise ValueError("physical kernel norms must be finite on valid rows")
    passing = np.isfinite(relative_error) & (relative_error < THRESHOLD)

    def stats(selected: NDArray) -> dict[str, object]:
        result = _error_stats(relative_error[selected])
        failed = selected & ~passing
        passed = selected & passing
        valid_norms = physical_norm[selected]
        median_norm = float(np.median(valid_norms)) if valid_norms.size else None
        result["physical_kernel_norm"] = {
            "median_all": median_norm,
            "median_failed": float(np.median(physical_norm[failed]))
            if failed.any() else None,
            "median_passed": float(np.median(physical_norm[passed]))
            if passed.any() else None,
            "failed_below_1pct_of_median": int((
                physical_norm[failed] < 0.01 * median_norm
            ).sum()) if median_norm is not None else 0,
        }
        return result

    kind_values = sorted(set(kinds.tolist()))

    def kind_name(kind: int) -> str:
        return KIND_NAMES[kind] if 0 <= kind < len(KIND_NAMES) else str(kind)

    result = {
        "threshold_relative_l2": THRESHOLD,
        "overall": stats(mask),
        "by_kind": {
            kind_name(kind): stats(mask & (kinds == kind)[:, None, None])
            for kind in kind_values
        },
        "by_mode": {},
    }
    for mode in range(mask.shape[1]):
        selected = np.zeros_like(mask)
        selected[:, mode] = mask[:, mode]
        group = stats(selected)
        # Both numerator and denominator exclude invalid modal/frequency cells.
        group["by_kind"] = {
            kind_name(kind): _error_stats(
                relative_error[:, mode][mask[:, mode] & (kinds == kind)[:, None]]
            ) for kind in kind_values
        }
        group["by_frequency"] = [
            {"frequency_hz": float(frequency), **_error_stats(
                relative_error[:, mode, index][mask[:, mode, index]]
            )}
            for index, frequency in enumerate(frequencies)
        ]
        result["by_mode"][str(mode)] = group

    flat = np.flatnonzero(mask & ~passing)
    # Invalid rows never enter the ranking; nonfinite errors sort first.
    ranking = np.nan_to_num(
        relative_error.ravel()[flat], nan=np.inf, posinf=np.inf, neginf=np.inf,
    )
    top = flat[np.argsort(ranking)[-20:][::-1]]
    result["worst_rows"] = []
    for index in top:
        sample, mode, frequency = np.unravel_index(index, mask.shape)
        error = float(relative_error[sample, mode, frequency])
        result["worst_rows"].append({
            "sample_id": int(sample_ids[sample]),
            "kind": kind_name(int(kinds[sample])),
            "mode": int(mode),
            "frequency_hz": float(frequencies[frequency]),
            "relative_l2": error if math.isfinite(error) else None,
            "physical_kernel_norm": float(physical_norm[sample, mode, frequency]),
        })
    return result


def summarize_history(records: list[dict[str, object]]) -> dict[str, object]:
    """Report the best validation epochs, without assuming their weights exist."""
    if not records:
        raise ValueError("training history is empty")

    def score(row: dict[str, object]) -> float:
        # Earlier histories omitted score on the initial epoch -1 record.
        if "score" in row:
            return float(row["score"])
        return (1 - row["validation"]["value"]["samples_all_within_1pct"]
                + 1 - row["validation"]["kernel"]["rows_within_5pct"])

    def brief(row: dict[str, object]) -> dict[str, object]:
        return {
            "epoch": row["epoch"],
            "score": score(row),
            "samples_all_within_1pct": row["validation"]["value"][
                "samples_all_within_1pct"
            ],
            "kernel_rows_within_5pct": row["validation"]["kernel"][
                "rows_within_5pct"
            ],
        }

    eligible = [row for row in records
                if kernel_checkpoint_key(row["validation"]) is not None]
    finite = [row for row in records if math.isfinite(score(row))]
    return {
        "training_epochs": sum(row["epoch"] >= 0 for row in records),
        "initial": brief(records[0]) if records[0]["epoch"] == -1 else None,
        "last": brief(records[-1]),
        "best_score": brief(min(finite, key=score))
        if finite else None,
        "best_kernel_with_curves_at_least_99pct": brief(max(
            eligible, key=lambda row: kernel_checkpoint_key(row["validation"]),
        )) if eligible else None,
        "last_five": [brief(row) for row in records if row["epoch"] >= 0][-5:],
        "note": "History metrics do not recover weights from unsaved epochs.",
    }
