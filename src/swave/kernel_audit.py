"""Small, read-only checks of individual physical kernel labels."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np

from .config import PhysicsConfig
from .kernels import sensitivity_kernels
from .secular import LayeredModel
from .solver import DispersionSolver
from .splits import mask_for_split

STEPS = (1e-6, 1e-5, 1e-4)


def kernel_model_metadata(dataset_dir: Path, kernel_dir: Path, split: str):
    """Read IDs/kinds only, without allocating all training kernels or curves."""
    kernel_ids = []
    for path in sorted((kernel_dir / split).glob("kernels-*.h5")):
        with h5py.File(path, "r") as handle:
            kernel_ids.append(np.asarray(handle["sample_id"], dtype=np.uint64))
    if not kernel_ids:
        raise FileNotFoundError(f"no kernel files for split {split!r}")
    ids = np.concatenate(kernel_ids)
    if not len(ids) or len(np.unique(ids)) != len(ids) or not mask_for_split(ids, split).all():
        raise ValueError("kernel IDs are duplicated or disagree with the requested split")
    source_ids, source_kinds = [], []
    for path in sorted(dataset_dir.glob("shard-*.h5")):
        with h5py.File(path, "r") as handle:
            source_ids.append(np.asarray(handle["sample_id"], dtype=np.uint64))
            source_kinds.append(np.asarray(handle["model_kind"]))
    if not source_ids:
        raise FileNotFoundError("no dataset shards found")
    source_ids, source_kinds = np.concatenate(source_ids), np.concatenate(source_kinds)
    order = np.argsort(source_ids)
    source_ids, source_kinds = source_ids[order], source_kinds[order]
    positions = np.searchsorted(source_ids, ids)
    if (len(np.unique(source_ids)) != len(source_ids)
            or np.any(positions >= len(source_ids))
            or not np.array_equal(source_ids[positions], ids)):
        raise ValueError("kernel IDs do not map uniquely to dataset model kinds")
    return ids, source_kinds[positions]


def select_audit_rows(rows, diagnostic_rows, reference_ids, *, seed=20261007):
    """Keep reported failures plus one random valid row per reference model/mode.

    Reference rows are background samples, not asserted neural successes.
    Neither this selection nor its aggregate statistics estimate prevalence.
    """
    position = {int(sample): i for i, sample in enumerate(rows["sample_id"])}
    chosen, keys = [], set()
    modes, frequencies = rows["kernel_mask"].shape[1:]
    for item in diagnostic_rows:
        sample, mode, hz = int(item["sample_id"]), int(item["mode"]), item["frequency_hz"]
        index = round((hz - 0.5) / 0.5)
        if (sample not in position or not 0 <= mode < modes
                or not 0 <= index < frequencies or abs(hz - (0.5 + 0.5 * index)) > 1e-8):
            raise ValueError("diagnostic row does not map to the kernel dataset")
        row = position[sample]
        if not rows["kernel_mask"][row, mode, index]:
            raise ValueError("diagnostic row is invalid under the current labels/corrections")
        key = (row, mode, index)
        if key not in keys:
            chosen.append({"row": row, "mode": mode, "frequency_index": index,
                           "selection": "diagnostic_failure",
                           "network_relative_l2": item.get("relative_l2")})
            keys.add(key)
    rng = np.random.default_rng(seed)
    for sample in reference_ids:
        row = position[int(sample)]
        for mode in range(modes):
            valid = np.flatnonzero(rows["kernel_mask"][row, mode])
            valid = [int(i) for i in valid if (row, mode, int(i)) not in keys]
            if valid:
                index = int(rng.choice(valid))
                chosen.append({"row": row, "mode": mode, "frequency_index": index,
                               "selection": "random_reference"})
                keys.add((row, mode, index))
    return chosen


def relative_row_error(candidate, reference):
    """Use one full layer vector, with the same denominator as kernel acceptance."""
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        return None
    result = np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1e-12)
    return float(result) if np.isfinite(result) else None


def _kernel_at(vs, phase, frequency, step):
    return sensitivity_kernels(
        vs, np.array([[phase]]), np.ones((1, 1), dtype=bool), [frequency], step=step,
    )[0, 0, 0]


def audit_kernel_rows(rows, chosen, *, progress: Callable | None = None):
    """Compare training labels, step sizes, and independently re-searched roots.

    This audits consistency of the existing physics implementation. The root
    searches share its secular function; agreement is not independent proof of
    physical correctness. No labels, masks, or checkpoints are modified.
    """
    records, solved = [], {}
    config = PhysicsConfig(root_tolerance=1e-11)
    for item in chosen:
        row, mode, index = item["row"], item["mode"], item["frequency_index"]
        frequency = 0.5 + 0.5 * index
        vs = rows["vs"][row].astype(np.float64)
        phase = float(rows["phase_velocity"][row, mode, index])
        stored = rows["kernel"][row, mode, index].astype(np.float64)
        at_stored = [_kernel_at(vs, phase, frequency, step) for step in STEPS]
        reference = at_stored[1]
        checks = {
            "stored_vs_recomputed": relative_row_error(stored, reference),
            "float16_rounding": relative_row_error(
                reference.astype(np.float16).astype(np.float64), reference,
            ),
            "step_1e-6_at_stored_root": relative_row_error(at_stored[0], reference),
            "step_1e-4_at_stored_root": relative_row_error(at_stored[2], reference),
            "resolved_vs_stored": None,
            "step_1e-6_at_resolved_root": None,
            "step_1e-4_at_resolved_root": None,
        }
        key = (row, index)
        if key not in solved:
            solved[key] = DispersionSolver(LayeredModel.from_vs(vs), config).solve_frequency(
                frequency, "consensus",
            )
        solution = solved[key]
        record = {
            **{k: v for k, v in item.items() if k != "row"},
            "sample_id": int(rows["sample_id"][row]), "frequency_hz": frequency,
            "stored_phase_velocity": phase,
            "physical_kernel_norm": float(np.linalg.norm(stored)),
            "solver_status": int(solution.status), "resolved_roots": solution.roots.tolist(),
            "resolved_phase_velocity": None, "phase_relative_shift": None,
            "checks": checks,
        }
        if solution.status == 0 and len(solution.roots) > mode:
            root = float(solution.roots[mode])
            at_resolved = [_kernel_at(vs, root, frequency, step) for step in STEPS]
            checks.update({
                "resolved_vs_stored": relative_row_error(at_resolved[1], stored),
                "step_1e-6_at_resolved_root": relative_row_error(at_resolved[0], at_resolved[1]),
                "step_1e-4_at_resolved_root": relative_row_error(at_resolved[2], at_resolved[1]),
            })
            record.update({"resolved_phase_velocity": root,
                           "phase_relative_shift": abs(root - phase) / abs(phase)})
        records.append(record)
        if progress is not None:
            progress(len(records), len(chosen))
    return records


def summarize_audit(records):
    """Nonfinite/unresolved comparisons remain explicit, not counted as passes."""
    result = {}
    for group in sorted({record["selection"] for record in records}):
        selected = [record for record in records if record["selection"] == group]
        checks = {}
        for name in selected[0]["checks"]:
            values = [record["checks"][name] for record in selected]
            finite = np.array([v for v in values if v is not None and np.isfinite(v)])
            checks[name] = {
                "checked_rows": len(finite), "unresolved_rows": len(values) - len(finite),
                "median_relative_l2": float(np.median(finite)) if len(finite) else None,
                "p95_relative_l2": float(np.quantile(finite, 0.95)) if len(finite) else None,
                "max_relative_l2": float(finite.max()) if len(finite) else None,
                "rows_over_1pct": int((finite > 0.01).sum()),
                "rows_over_5pct": int((finite > 0.05).sum()),
            }
        result[group] = {"rows": len(selected), "checks": checks}
    return result
