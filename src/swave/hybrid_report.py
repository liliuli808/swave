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

from .config import inversion_identity_hash, load_inversion_config
from .dataset import dataset_manifest_sha256, validate_dataset_files
from .geology import ModelKind
from .hybrid_results import (
    HybridManifest,
    HybridResultBatch,
    validate_complete_hybrid_results,
    validate_hybrid_result_shard,
)
from .inversion_results import checkpoint_sha256, sample_id_sha256, software_sha256
from .splits import SPLIT_POLICY, mask_for_split

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
            required = np.fromiter(required_ids, dtype=np.uint64)
            indexes = np.flatnonzero(np.isin(sample_ids, required))
            if indexes.size == 0:
                continue
            kinds = np.asarray(handle["model_kind"][indexes], dtype=np.uint8)
            profiles = np.asarray(handle["vs"][indexes], dtype=np.float64)
            for sample_value, kind, profile in zip(
                sample_ids[indexes], kinds, profiles, strict=True
            ):
                sample_id = int(sample_value)
                if sample_id in result:
                    raise ValueError("dataset contains a duplicate result sample ID")
                result[sample_id] = (int(kind), profile.copy())
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
            directory / f"{job}.h5", manifest=manifest, expected_job=job
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
    preparation_success = _concatenate(batches, "preparation_success")
    control_success = _concatenate(batches, "control_success")
    hybrid_success = _concatenate(batches, "hybrid_success")
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
            truth, supervised_vs, model_kind, preparation_success
        ),
        "preparation_failure_count": int((~preparation_success).sum()),
        "mean_dimensionless_sensitivity_by_layer": (
            _concatenate(batches, "sensitivity")[preparation_success]
            .mean(axis=0)
            .tolist()
            if np.any(preparation_success)
            else None
        ),
        "mean_prior_weight_by_layer": (
            _concatenate(batches, "prior_weights")[preparation_success]
            .mean(axis=0)
            .tolist()
            if np.any(preparation_success)
            else None
        ),
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


def _comparison(
    path: Path | None,
    label: str,
    *,
    expected_dataset_config_hash: str,
    expected_dataset_manifest_sha256: str,
    expected_populations: dict[str, dict[str, object]],
    supervised: bool,
    expected_forward_checkpoint_sha256: str,
    expected_supervised_checkpoint_sha256: tuple[str, ...],
    expected_supervised_run_identity_sha256: str,
) -> dict[str, object]:
    if path is None:
        return {"available": False, "reason": f"{label} path was not provided"}
    if not path.is_file():
        return {"available": False, "reason": f"{label} file is missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    identity = payload if supervised else payload.get("result_identity")
    if not isinstance(identity, dict):
        raise TypeError(f"{label} identity is missing")
    expected_identity = {
        "dataset_config_hash": expected_dataset_config_hash,
        "dataset_manifest_sha256": expected_dataset_manifest_sha256,
        "split_policy": SPLIT_POLICY,
    }
    for name, expected in expected_identity.items():
        if identity.get(name) != expected:
            raise ValueError(f"{label} {name} does not match hybrid results")
    if not supervised and (
        identity.get("checkpoint_sha256")
        != expected_forward_checkpoint_sha256
    ):
        raise ValueError(f"{label} forward checkpoint does not match hybrid results")
    if not supervised:
        expected_baseline_hash = inversion_identity_hash(
            load_inversion_config("configs/inversion.toml")
        )
        if (
            identity.get("inversion_config_hash") != expected_baseline_hash
            or identity.get("experiment") != "both"
        ):
            raise ValueError(f"{label} inversion protocol does not match")
    if supervised:
        supervised_identity = (
            payload.get("supervised_checkpoint_sha256"),
            payload.get("supervised_run_identity_sha256"),
            payload.get("seed_ensemble"),
        )
        expected_supervised_identity = (
            list(expected_supervised_checkpoint_sha256),
            expected_supervised_run_identity_sha256,
            [0, 1, 2],
        )
        if supervised_identity != expected_supervised_identity:
            return {
                "available": False,
                "reason": f"{label} lacks the exact supervised checkpoint identity",
            }
    population_field = (
        "split_sample_identity" if supervised else "comparison_populations"
    )
    populations = payload.get(population_field)
    if not isinstance(populations, dict):
        return {
            "available": False,
            "reason": f"{label} lacks exact sample-population identity",
        }
    for split, expected in expected_populations.items():
        actual = populations.get(split)
        if actual != expected:
            raise ValueError(f"{label} {split} sample population does not match")
    return {"available": True, "path": path.as_posix(), "payload": payload}


def _manifest_scientific_identity(manifest: HybridManifest) -> tuple[object, ...]:
    return (
        manifest.dataset_config_hash,
        manifest.dataset_manifest_sha256,
        manifest.forward_checkpoint_sha256,
        manifest.supervised_checkpoint_sha256,
        manifest.supervised_seeds,
        manifest.supervised_run_identity_sha256,
        manifest.tuning_sha256,
        manifest.split_policy,
        manifest.hybrid_config_hash,
        manifest.selected_prior_lambda,
        manifest.vs_min,
        manifest.vs_max,
        manifest.prior_weight_min,
        manifest.prior_weight_max,
        manifest.software_sha256,
    )


def _population_identity(
    grouped: dict[str, list[HybridResultBatch]],
) -> dict[str, object]:
    populations: list[NDArray[np.uint64]] = []
    for batches in grouped.values():
        raw_ids = _concatenate(batches, "sample_id")
        ids = np.unique(raw_ids)
        if len(ids) != len(raw_ids):
            raise ValueError(
                "hybrid result sample IDs are duplicated across shards"
            )
        populations.append(np.sort(ids))
    first = populations[0]
    if any(not np.array_equal(first, current) for current in populations[1:]):
        raise ValueError("hybrid noise scenarios do not share one sample population")
    return {
        "sample_count": len(first),
        "sample_id_sha256": sample_id_sha256(first),
    }


def _dataset_population_identity(dataset_dir: Path, split: str) -> dict[str, object]:
    pieces: list[NDArray[np.uint64]] = []
    for path in sorted(dataset_dir.glob("shard-*.h5")):
        with h5py.File(path, "r") as handle:
            sample_ids = np.asarray(handle["sample_id"], dtype=np.uint64)
        selected = sample_ids[mask_for_split(sample_ids, split)]
        if selected.size:
            pieces.append(selected)
    if not pieces:
        raise ValueError(f"dataset {split} split is empty")
    values = np.sort(np.concatenate(pieces)).astype(np.uint64, copy=False)
    return {
        "sample_count": len(values),
        "sample_id_sha256": sample_id_sha256(values),
    }


def _same_population_methods(
    split_summaries: dict[str, object],
    comparisons: dict[str, dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    baseline = comparisons["baseline"]
    for split, raw_summary in split_summaries.items():
        assert isinstance(raw_summary, dict)
        combined = raw_summary["combined_noise_scenarios"]
        assert isinstance(combined, dict)
        methods: dict[str, object] = {
            "global_bound_control": combined["control"],
            "sensitivity_weighted_hybrid": combined["hybrid"],
            "direct_supervised_prior": combined["supervised_prior"],
        }
        if split == "inversion" and baseline["available"]:
            payload = baseline["payload"]
            assert isinstance(payload, dict)
            try:
                methods["narrow_bound_lbfgsb"] = payload["experiment_scopes"][
                    "full"
                ]["groups"]["overall"]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    "baseline summary lacks full-population inversion metrics"
                ) from error
        else:
            methods["narrow_bound_lbfgsb"] = {
                "available": False,
                "reason": (
                    baseline.get("reason", "baseline is only defined for inversion")
                    if split == "inversion"
                    else "narrow-bound baseline is not defined for the test split"
                ),
            }
        result[split] = methods
    return result


def _validated_tuning_summary(
    results_root: Path,
    manifest: HybridManifest,
) -> dict[str, object]:
    path = results_root / "tuning.json"
    if checkpoint_sha256(path) != manifest.tuning_sha256:
        raise ValueError("hybrid tuning artifact identity does not match")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("hybrid tuning artifact is not readable JSON") from error
    expected_fields = {
        "schema_version",
        "split_policy",
        "usage",
        "dataset_config_hash",
        "dataset_manifest_sha256",
        "forward_checkpoint_sha256",
        "supervised_checkpoint_sha256",
        "supervised_seeds",
        "supervised_run_identity_sha256",
        "hybrid_config_hash",
        "software_sha256",
        "validation_sample_count",
        "validation_sample_id_sha256",
        "validation_sample_ids",
        "prior_lambda_candidates",
        "noise_scenarios",
        "candidate_mae_km_s",
        "selected_prior_lambda",
        "selection_metric",
        "tie_break",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("hybrid tuning artifact fields do not match the schema")
    expected_identity = {
        "schema_version": 1,
        "split_policy": SPLIT_POLICY,
        "usage": "validation_only_prior_lambda_selection",
        "dataset_config_hash": manifest.dataset_config_hash,
        "dataset_manifest_sha256": manifest.dataset_manifest_sha256,
        "forward_checkpoint_sha256": manifest.forward_checkpoint_sha256,
        "supervised_checkpoint_sha256": list(
            manifest.supervised_checkpoint_sha256
        ),
        "supervised_seeds": list(manifest.supervised_seeds),
        "supervised_run_identity_sha256": (
            manifest.supervised_run_identity_sha256
        ),
        "hybrid_config_hash": manifest.hybrid_config_hash,
        "software_sha256": manifest.software_sha256,
        "noise_scenarios": ["clean", "noise_1pct"],
        "selection_metric": (
            "mean_vs_mae_across_validation_samples_layers_and_noise"
        ),
        "tie_break": "smallest_prior_lambda",
    }
    for name, expected in expected_identity.items():
        if payload[name] != expected:
            raise ValueError(f"hybrid tuning {name} does not match final results")
    raw_ids = payload["validation_sample_ids"]
    if not isinstance(raw_ids, list):
        raise TypeError("hybrid tuning validation sample IDs must be a list")
    sample_ids = np.asarray(raw_ids, dtype=np.uint64)
    if (
        payload["validation_sample_count"] != len(sample_ids)
        or payload["validation_sample_id_sha256"] != sample_id_sha256(sample_ids)
    ):
        raise ValueError("hybrid tuning validation sample identity is inconsistent")
    candidates = payload["prior_lambda_candidates"]
    metrics = payload["candidate_mae_km_s"]
    if (
        not isinstance(candidates, list)
        or not isinstance(metrics, dict)
        or set(metrics) != {str(value) for value in candidates}
    ):
        raise ValueError("hybrid tuning candidate metrics are incomplete")
    scores = {float(value): float(metrics[str(value)]) for value in candidates}
    if (
        not scores
        or any(not np.isfinite(key) or key <= 0 for key in scores)
        or any(not np.isfinite(value) or value < 0 for value in scores.values())
    ):
        raise ValueError("hybrid tuning candidate metrics are invalid")
    best = min(scores.values())
    selected = min(
        candidate
        for candidate, score in scores.items()
        if np.isclose(score, best, rtol=0.0, atol=1e-15)
    )
    if selected != payload["selected_prior_lambda"] or selected != (
        manifest.selected_prior_lambda
    ):
        raise ValueError("hybrid tuning selected prior lambda is inconsistent")
    return payload


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
    if (
        selected.get("mean_dimensionless_sensitivity_by_layer") is None
        or selected.get("mean_prior_weight_by_layer") is None
    ):
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
    if set(manifests) != {"test", "inversion"}:
        raise ValueError(
            "complete hybrid test and inversion results are both required"
        )
    identities = {
        _manifest_scientific_identity(manifest)
        for _, manifest in manifests.values()
    }
    if len(identities) != 1:
        raise ValueError("hybrid test and inversion scientific identities do not match")
    reference_manifest = next(iter(manifests.values()))[1]
    if reference_manifest.software_sha256 != software_sha256():
        raise ValueError("hybrid result software identity does not match this checkout")
    tuning_summary = _validated_tuning_summary(results_root, reference_manifest)
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
    population_identities: dict[str, dict[str, object]] = {}
    for split, grouped in grouped_by_split.items():
        manifest = manifests[split][1]
        population_identities[split] = _population_identity(grouped)
        if population_identities[split] != _dataset_population_identity(
            dataset_root, split
        ):
            raise ValueError(
                f"hybrid {split} results do not cover the complete dataset split"
            )
        combined_batches = [
            batch for batches in grouped.values() for batch in batches
        ]
        split_summaries[split] = {
            "selected_prior_lambda": manifest.selected_prior_lambda,
            "population_identity": population_identities[split],
            "combined_noise_scenarios": _noise_summary(
                combined_batches, truth_map
            ),
            "by_noise": {
                noise: _noise_summary(batches, truth_map)
                for noise, batches in grouped.items()
            },
        }
    comparisons = {
        "baseline": _comparison(
            baseline_summary,
            "baseline summary",
            expected_dataset_config_hash=dataset_manifest.config_hash,
            expected_dataset_manifest_sha256=dataset_digest,
            expected_populations=(
                {"inversion": population_identities["inversion"]}
                if "inversion" in population_identities
                else population_identities
            ),
            supervised=False,
            expected_forward_checkpoint_sha256=(
                reference_manifest.forward_checkpoint_sha256
            ),
            expected_supervised_checkpoint_sha256=(
                reference_manifest.supervised_checkpoint_sha256
            ),
            expected_supervised_run_identity_sha256=(
                reference_manifest.supervised_run_identity_sha256
            ),
        ),
        "supervised_evaluation": _comparison(
            supervised_evaluation,
            "supervised evaluation",
            expected_dataset_config_hash=dataset_manifest.config_hash,
            expected_dataset_manifest_sha256=dataset_digest,
            expected_populations=population_identities,
            supervised=True,
            expected_forward_checkpoint_sha256=(
                reference_manifest.forward_checkpoint_sha256
            ),
            expected_supervised_checkpoint_sha256=(
                reference_manifest.supervised_checkpoint_sha256
            ),
            expected_supervised_run_identity_sha256=(
                reference_manifest.supervised_run_identity_sha256
            ),
        ),
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset_config_hash": dataset_manifest.config_hash,
        "dataset_manifest_sha256": dataset_digest,
        "tuning": tuning_summary,
        "splits": split_summaries,
        "comparisons": comparisons,
        "same_population_methods": _same_population_methods(
            split_summaries, comparisons
        ),
    }
    _plot_sensitivity(
        summary, output / "sensitivity-and-prior-weight-by-depth.png"
    )
    _atomic_json(output / "summary.json", summary)
    return summary
