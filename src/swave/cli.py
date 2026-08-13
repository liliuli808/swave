"""Command-line workflows for generation, training, prediction, and plots."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import (
    DatasetConfig,
    InversionConfig,
    TrainingConfig,
    load_dataset_config,
    load_inversion_config,
    load_training_config,
)
from .dataset import generate_dataset
from .geology import generate_model
from .inference import ForwardPredictor
from .secular import LayeredModel
from .solver import DispersionSolver
from .training import evaluate, train


def _dataset_config(path: str | None) -> DatasetConfig:
    return load_dataset_config(path) if path else DatasetConfig()


def _training_config(path: str | None) -> TrainingConfig:
    return load_training_config(path) if path else TrainingConfig()


def _inversion_config(path: str | None) -> InversionConfig:
    return load_inversion_config(path) if path else InversionConfig()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _print_json(value: object) -> None:
    payload = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    print(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
    )


def _generate(arguments: argparse.Namespace) -> int:
    config = _dataset_config(arguments.config)
    overrides = {
        name: value
        for name, value in {
            "samples": arguments.samples,
            "workers": arguments.workers,
            "output_dir": (
                Path(arguments.output_dir) if arguments.output_dir else None
            ),
        }.items()
        if value is not None
    }
    manifest = generate_dataset(replace(config, **overrides))
    print(json.dumps(asdict(manifest), indent=2))
    return 0


def _audit_dataset(arguments: argparse.Namespace) -> int:
    from .dataset_audit import audit_dataset, write_audit_report

    config = load_dataset_config(arguments.dataset_config)
    report = audit_dataset(Path(arguments.dataset_dir), config)
    output = write_audit_report(arguments.output, report)
    print(output)
    return 0


def _train(arguments: argparse.Namespace) -> int:
    config = _training_config(arguments.config)
    overrides = {
        name: value
        for name, value in {
            "dataset_dir": (
                Path(arguments.dataset_dir) if arguments.dataset_dir else None
            ),
            "output_dir": (
                Path(arguments.output_dir) if arguments.output_dir else None
            ),
            "device": arguments.device,
            "epochs": arguments.epochs,
            "num_workers": arguments.num_workers,
        }.items()
        if value is not None
    }
    print(train(replace(config, **overrides)))
    return 0


def _train_inverse(arguments: argparse.Namespace) -> int:
    from .supervised_inversion import load_supervised_config, train_supervised

    config = load_supervised_config(arguments.config)
    overrides = {
        name: value
        for name, value in {
            "dataset_dir": (
                Path(arguments.dataset_dir) if arguments.dataset_dir else None
            ),
            "output_dir": (
                Path(arguments.output_dir) if arguments.output_dir else None
            ),
            "device": arguments.device,
            "num_workers": arguments.num_workers,
            "epochs": arguments.epochs,
        }.items()
        if value is not None
    }
    print(train_supervised(replace(config, **overrides)))
    return 0


def _evaluate(arguments: argparse.Namespace) -> int:
    metrics = evaluate(
        arguments.checkpoint,
        arguments.dataset_dir,
        device=arguments.device,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def _predict(arguments: argparse.Namespace) -> int:
    if arguments.input_file and arguments.vs:
        raise ValueError("use either command-line Vs values or --input-file")
    if arguments.input_file:
        values = np.loadtxt(arguments.input_file, dtype=np.float32)
    else:
        values = np.asarray(arguments.vs, dtype=np.float32)
    if values.size == 0:
        raise ValueError("provide 20 Vs values or --input-file")
    predictor = ForwardPredictor.load(arguments.checkpoint, device=arguments.device)
    frequencies, curves = predictor.predict_with_frequencies(values)
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, frequencies=frequencies, phase_velocity=curves)
        print(output)
    else:
        print(
            json.dumps(
                {
                    "frequencies_hz": frequencies.tolist(),
                    "phase_velocity_km_s": curves.tolist(),
                }
            )
        )
    return 0


def _plot_model(arguments: argparse.Namespace) -> int:
    from .plotting import plot_model

    config = _dataset_config(arguments.config)
    seed = config.seed if arguments.seed is None else arguments.seed
    generated = generate_model(arguments.sample_id, config.geology, seed)
    print(
        plot_model(
            generated,
            arguments.output,
            thickness_km=config.geology.thickness_km,
        )
    )
    return 0


def _plot_dispersion(arguments: argparse.Namespace) -> int:
    from .plotting import plot_dispersion

    config = _dataset_config(arguments.config)
    generated = generate_model(arguments.sample_id, config.geology, config.seed)
    model = LayeredModel(
        depth=np.arange(config.geology.layers) * config.geology.thickness_km,
        density=generated.density,
        vs=generated.vs,
        vp=generated.vp,
    )
    result = DispersionSolver(model, config.physics).solve_grid(
        strategy=arguments.strategy
    )
    print(plot_dispersion(result, arguments.output))
    return 0


def _plot_history(arguments: argparse.Namespace) -> int:
    from .plotting import plot_training_history

    print(plot_training_history(arguments.history, arguments.output))
    return 0


def _invert(arguments: argparse.Namespace) -> int:
    from .inversion_runner import run_inversion_experiment

    config = _inversion_config(arguments.config)
    overrides = {
        name: value
        for name, value in {
            "dataset_config": (
                Path(arguments.dataset_config) if arguments.dataset_config else None
            ),
            "dataset_dir": (
                Path(arguments.dataset_dir) if arguments.dataset_dir else None
            ),
            "checkpoint": (
                Path(arguments.checkpoint) if arguments.checkpoint else None
            ),
            "output_dir": (
                Path(arguments.output_dir) if arguments.output_dir else None
            ),
            "device": arguments.device,
            "workers": arguments.workers,
            "deep_samples_per_job": arguments.deep_samples_per_job,
            "threads_per_worker": arguments.threads_per_worker,
            "task_index": arguments.task_index,
            "task_count": arguments.task_count,
        }.items()
        if value is not None
    }
    manifest = run_inversion_experiment(
        replace(config, **overrides), arguments.experiment
    )
    _print_json(manifest)
    return 0


def _inversion_report(arguments: argparse.Namespace) -> int:
    from .inversion_report import build_inversion_report

    summary = build_inversion_report(
        Path(arguments.results_dir),
        Path(arguments.dataset_dir),
        Path(arguments.output_dir),
        dataset_config=load_dataset_config(arguments.dataset_config),
    )
    _print_json(summary)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swave",
        description="Rayleigh-wave forward modeling and four-head surrogate",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="generate or resume HDF5 dataset shards"
    )
    generate.add_argument("--config")
    generate.add_argument("--samples", type=int)
    generate.add_argument("--workers", type=int)
    generate.add_argument("--output-dir")
    generate.set_defaults(handler=_generate)

    audit = subparsers.add_parser(
        "audit-dataset", help="audit dataset identities, duplicates, and geology"
    )
    audit.add_argument("--dataset-config", default="configs/dataset.toml")
    audit.add_argument("--dataset-dir", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(handler=_audit_dataset)

    training = subparsers.add_parser(
        "train", help="train or resume the neural surrogate"
    )
    training.add_argument("--config")
    training.add_argument("--dataset-dir")
    training.add_argument("--output-dir")
    training.add_argument("--epochs", type=int)
    training.add_argument("--num-workers", type=int)
    training.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    training.set_defaults(handler=_train)

    inverse_training = subparsers.add_parser(
        "train-inverse", help="train the split-safe supervised inversion ensemble"
    )
    inverse_training.add_argument(
        "--config", default="configs/supervised-inversion-48g.toml"
    )
    inverse_training.add_argument("--dataset-dir")
    inverse_training.add_argument("--output-dir")
    inverse_training.add_argument("--epochs", type=int)
    inverse_training.add_argument("--num-workers", type=int)
    inverse_training.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps")
    )
    inverse_training.set_defaults(handler=_train_inverse)

    evaluation = subparsers.add_parser(
        "evaluate", help="evaluate a checkpoint on the test split"
    )
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--dataset-dir", required=True)
    evaluation.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    evaluation.set_defaults(handler=_evaluate)

    prediction = subparsers.add_parser(
        "predict", help="predict four modal curves from 20 Vs values"
    )
    prediction.add_argument("checkpoint")
    prediction.add_argument("vs", nargs="*", type=float)
    prediction.add_argument("--input-file")
    prediction.add_argument("--output")
    prediction.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    prediction.set_defaults(handler=_predict)

    model_plot = subparsers.add_parser(
        "plot-model", help="plot one deterministic geological model"
    )
    model_plot.add_argument("--config")
    model_plot.add_argument("--sample-id", required=True, type=int)
    model_plot.add_argument("--seed", type=int)
    model_plot.add_argument("--output", required=True)
    model_plot.set_defaults(handler=_plot_model)

    dispersion_plot = subparsers.add_parser(
        "plot-dispersion", help="solve and plot one four-mode dispersion result"
    )
    dispersion_plot.add_argument("--config")
    dispersion_plot.add_argument("--sample-id", required=True, type=int)
    dispersion_plot.add_argument(
        "--strategy",
        choices=("raw", "degraded", "quadratic"),
        default="quadratic",
    )
    dispersion_plot.add_argument("--output", required=True)
    dispersion_plot.set_defaults(handler=_plot_dispersion)

    history_plot = subparsers.add_parser(
        "plot-history", help="plot training loss and validation MAE"
    )
    history_plot.add_argument("--history", required=True)
    history_plot.add_argument("--output", required=True)
    history_plot.set_defaults(handler=_plot_history)

    inversion = subparsers.add_parser(
        "invert", help="run or resume full and deep inversion experiments"
    )
    inversion.add_argument("--config")
    inversion.add_argument(
        "--experiment", choices=("full", "deep", "both"), default="both"
    )
    inversion.add_argument("--dataset-config")
    inversion.add_argument("--dataset-dir")
    inversion.add_argument("--checkpoint")
    inversion.add_argument("--output-dir")
    inversion.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    inversion.add_argument("--workers", type=int)
    inversion.add_argument("--deep-samples-per-job", type=int)
    inversion.add_argument("--threads-per-worker", type=int)
    inversion.add_argument("--task-index", type=int)
    inversion.add_argument("--task-count", type=int)
    inversion.set_defaults(handler=_invert)

    inversion_report = subparsers.add_parser(
        "inversion-report", help="validate inversion results and build the report"
    )
    inversion_report.add_argument("--dataset-config", default="configs/dataset.toml")
    inversion_report.add_argument("--results-dir", default="results/inversion")
    inversion_report.add_argument("--dataset-dir", default="data/production")
    inversion_report.add_argument("--output-dir", default="results/inversion-report")
    inversion_report.set_defaults(handler=_inversion_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        return int(arguments.handler(arguments))
    except (ArithmeticError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
