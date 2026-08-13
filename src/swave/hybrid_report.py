"""Validated truth joins, metrics, and figures for hybrid inversion."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from numpy.typing import NDArray

from .dataset import dataset_manifest_sha256, validate_dataset_files
from .geology import ModelKind
from .hybrid_results import (
    HybridManifest,
    HybridResultBatch,
    validate_complete_hybrid_results,
    validate_hybrid_result_shard,
)

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

_KIND_NAMES = {int(kind): kind.name.lower() for kind in ModelKind}


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _truth_by_id(
    dataset_dir: Path,
    required_ids: set[int],
) -> dict[int, tuple[int, NDArray[np.float64]]]:
    result: dict[int, tuple[int, NDArray[np.float64]]] = {}
    for path in sorted(dataset_dir.glob("shard-*.h5")):
        with h5py.File(path, "r") as handle:
            sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
            indexes = [
                index
                for index, sample_id in enumerate(sample_ids)
                if int(sample_id) in required_ids
            ]
            for index in indexes:
                sample_id = int(sample_ids[index])
                if sample_id in result:
                    raise ValueError("dataset contains a duplicate result sample ID")
                result[sample_id] = (
                    int(handle["model_kind"][index]),
                    np.asarray(handle["vs"][index], dtype=np.float64),
                )
    missing = required_ids - set(result)
    if missing:
        raise ValueError(f"dataset is missing {len(missing)} hybrid result sample IDs")
    return result


def _method_metrics(
    truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
    model_kind: NDArray[np.uint8],
    success: NDArray[np.bool_],
) -> dict[str, object]:
    selected = np.asarray(success, dtype=np.bool_)
    if not np.any(selected):
        return {
            "sample_count": 0,
            "failure_count": len(selected),
            "overall": None,
            "by_model_kind": {},
            "per_layer": [],
        }

    def metrics(
        true_values: NDArray[np.float64], predicted: NDArray[np.float64]
    ) -> dict[str, float | int]:
        difference = predicted - true_values
        absolute = np.abs(difference)
        return {
            "sample_count": len(true_values),
            "cell_count": int(difference.size),
            "mae_km_s": float(absolute.mean()),
            "rmse_km_s": float(np.sqrt(np.square(difference).mean())),
            "p95_absolute_error_km_s": float(np.percentile(absolute, 95)),
            "bias_km_s": float(difference.mean()),
        }

    true_selected = truth[selected]
    predicted_selected = prediction[selected]
    by_kind: dict[str, object] = {}
    for kind, name in _KIND_NAMES.items():
        kind_rows = selected & (model_kind == kind)
        if np.any(kind_rows):
            by_kind[name] = metrics(truth[kind_rows], prediction[kind_rows])
    per_layer: list[dict[str, float | int]] = []
    for layer in range(20):
        layer_metrics = metrics(
            true_selected[:, layer : layer + 1],
            predicted_selected[:, layer : layer + 1],
        )
        per_layer.append({"layer": layer, "depth_km": layer / 10.0, **layer_metrics})
    return {
        "sample_count": int(selected.sum()),
        "failure_count": int((~selected).sum()),
        "overall": metrics(true_selected, predicted_selected),
        "by_model_kind": by_kind,
        "per_layer": per_layer,
    }


def _reconstruction_mae(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    success: NDArray[np.bool_],
) -> float | None:
    selected = valid_mask & success[:, None, None]
    if not np.any(selected):
        return None
    return float(np.abs(predicted[selected] - observed[selected]).mean())


def _successful_mean(values: NDArray[np.generic], success: NDArray[np.bool_]) -> float | None:
    if not np.any(success):
        return None
    return float(values[success].mean())


def _noise_from_job(job: str) -> str:
    if "-noise_1pct-" in job:
        return "noise_1pct"
    if "-clean-" in job:
        return "clean"
    raise ValueError("hybrid result job has an unknown noise scenario")


def _collect_batches(
    directory: Path,
    manifest: HybridManifest,
) -> dict[str, list[HybridResultBatch]]:
    grouped: dict[str, list[HybridResultBatch]] = {}
    for job in manifest.expected_jobs:
        batch = validate_hybrid_result_shard(
            directory / f"{job}.h5", manifest=manifest
        )
        grouped.setdefault(_noise_from_job(job), []).append(batch)
    return grouped


def _concatenate(
    batches: list[HybridResultBatch], name: str
) -> NDArray[Any]:
    return np.concatenate([getattr(batch, name) for batch in batches], axis=0)


def _noise_summary(
    batches: list[HybridResultBatch],
    truth_map: dict[int, tuple[int, NDArray[np.float64]]],
) -> dict[str, object]:
    sample_id = _concatenate(batches, "sample_id")
    model_kind = _concatenate(batches, "model_kind")
    truth_rows: list[NDArray[np.float64]] = []
    for sample, stored_kind in zip(sample_id, model_kind, strict=True):
        truth_kind, truth = truth_map[int(sample)]
        if truth_kind != int(stored_kind):
            raise ValueError("hybrid result model kind does not match the dataset")
        truth_rows.append(truth)
    truth = np.stack(truth_rows)
    control_vs = _concatenate(batches, "control_vs").astype(np.float64)
    hybrid_vs = _concatenate(batches, "hybrid_vs").astype(np.float64)
    supervised_vs = _concatenate(batches, "supervised_vs").astype(np.float64)
    control_success = _concatenate(batches, "control_success")
    hybrid_success = _concatenate(batches, "hybrid_success")
    all_success = np.ones(len(sample_id), dtype=np.bool_)
    observed = _concatenate(batches, "observed_phase_velocity").astype(np.float64)
    valid = _concatenate(batches, "valid_mask")
    control_prediction = _concatenate(batches, "control_prediction").astype(np.float64)
    hybrid_prediction = _concatenate(batches, "hybrid_prediction").astype(np.float64)
    return {
        "sample_count": len(sample_id),
        "control": _method_metrics(
            truth, control_vs, model_kind, control_success
        ),
        "hybrid": _method_metrics(truth, hybrid_vs, model_kind, hybrid_success),
        "supervised_prior": _method_metrics(
            truth, supervised_vs, model_kind, all_success
        ),
        "mean_dimensionless_sensitivity_by_layer": _concatenate(
            batches, "sensitivity"
        ).mean(axis=0).tolist(),
        "mean_prior_weight_by_layer": _concatenate(
            batches, "prior_weights"
        ).mean(axis=0).tolist(),
        "reconstruction_mae_km_s": {
            "control": _reconstruction_mae(
                observed, control_prediction, valid, control_success
            ),
            "hybrid": _reconstruction_mae(
                observed, hybrid_prediction, valid, hybrid_success
            ),
        },
        "optimization": {
            "control_mean_iterations": _successful_mean(
                _concatenate(batches, "control_iterations"), control_success
            ),
            "control_mean_evaluations": _successful_mean(
                _concatenate(batches, "control_evaluations"), control_success
            ),
            "hybrid_mean_iterations": _successful_mean(
                _concatenate(batches, "hybrid_iterations"), hybrid_success
            ),
            "hybrid_mean_evaluations": _successful_mean(
                _concatenate(batches, "hybrid_evaluations"), hybrid_success
            ),
        },
        "objective_terms": {
            "control_mean_data_misfit": _successful_mean(
                _concatenate(batches, "control_data_misfit"), control_success
            ),
            "control_mean_smoothness": _successful_mean(
                _concatenate(batches, "control_smoothness"), control_success
            ),
            "hybrid_mean_data_misfit": _successful_mean(
                _concatenate(batches, "hybrid_data_misfit"), hybrid_success
            ),
            "hybrid_mean_smoothness": _successful_mean(
                _concatenate(batches, "hybrid_smoothness"), hybrid_success
            ),
            "hybrid_mean_learning_prior": _successful_mean(
                _concatenate(batches, "hybrid_learning_prior"), hybrid_success
            ),
        },
    }


def _comparison(path: Path | None, label: str) -> dict[str, object]:
    if path is None:
        return {"available": False, "reason": f"{label} path was not provided"}
    if not path.is_file():
        return {"available": False, "reason": f"{label} file is missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON") from error
    return {"available": True, "path": path.as_posix(), "payload": payload}


def _plot_sensitivity(summary: dict[str, object], output: Path) -> None:
    split_payload = summary["splits"]
    assert isinstance(split_payload, dict)
    selected: dict[str, object] | None = None
    for split in ("inversion", "test"):
        current = split_payload.get(split)
        if isinstance(current, dict):
            noises = current.get("by_noise")
            if isinstance(noises, dict) and noises:
                selected = next(iter(noises.values()))
                break
    if selected is None:
        return
    sensitivity = np.asarray(selected["mean_dimensionless_sensitivity_by_layer"])
    weights = np.asarray(selected["mean_prior_weight_by_layer"])
    depth = np.arange(20) * 0.1
    figure, left = plt.subplots(figsize=(7, 6))
    right = left.twiny()
    left.plot(sensitivity, depth, color="tab:blue", marker="o", label="Sensitivity")
    right.plot(weights, depth, color="tab:orange", marker="s", label="Prior weight")
    left.set_xlabel("Mean dimensionless sensitivity", color="tab:blue")
    right.set_xlabel("Mean learning-prior weight", color="tab:orange")
    left.set_ylabel("Depth (km)")
    left.invert_yaxis()
    left.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_hybrid_report(
    results_dir: Path | str,
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    baseline_summary: Path | None = None,
    supervised_evaluation: Path | None = None,
) -> dict[str, object]:
    """Validate complete hybrid artifacts, then join truth and report metrics."""
    results_root = Path(results_dir)
    dataset_root = Path(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_manifest = validate_dataset_files(dataset_root)
    dataset_digest = dataset_manifest_sha256(dataset_manifest)
    manifests: dict[str, tuple[Path, HybridManifest]] = {}
    for split in ("test", "inversion"):
        directory = results_root / split
        if (directory / "manifest.json").is_file():
            manifest = validate_complete_hybrid_results(directory)
            if manifest.split != split:
                raise ValueError("hybrid result directory and manifest split disagree")
            if manifest.dataset_config_hash != dataset_manifest.config_hash:
                raise ValueError("hybrid result dataset configuration does not match")
            if manifest.dataset_manifest_sha256 != dataset_digest:
                raise ValueError("hybrid result dataset manifest does not match")
            manifests[split] = (directory, manifest)
    if not manifests and (results_root / "manifest.json").is_file():
        manifest = validate_complete_hybrid_results(results_root)
        manifests[manifest.split] = (results_root, manifest)
    if not manifests:
        raise ValueError("no complete hybrid test or inversion result was found")
    required_ids: set[int] = set()
    grouped_by_split: dict[str, dict[str, list[HybridResultBatch]]] = {}
    for split, (directory, manifest) in manifests.items():
        grouped = _collect_batches(directory, manifest)
        grouped_by_split[split] = grouped
        for batches in grouped.values():
            for batch in batches:
                required_ids.update(int(value) for value in batch.sample_id)
    truth_map = _truth_by_id(dataset_root, required_ids)
    split_summaries: dict[str, object] = {}
    for split, grouped in grouped_by_split.items():
        manifest = manifests[split][1]
        split_summaries[split] = {
            "selected_prior_lambda": manifest.selected_prior_lambda,
            "by_noise": {
                noise: _noise_summary(batches, truth_map)
                for noise, batches in grouped.items()
            },
        }
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset_config_hash": dataset_manifest.config_hash,
        "dataset_manifest_sha256": dataset_digest,
        "splits": split_summaries,
        "comparisons": {
            "baseline": _comparison(baseline_summary, "baseline summary"),
            "supervised_evaluation": _comparison(
                supervised_evaluation, "supervised evaluation"
            ),
        },
    }
    _plot_sensitivity(
        summary, output / "sensitivity-and-prior-weight-by-depth.png"
    )
    _atomic_json(output / "summary.json", summary)
    return summary
