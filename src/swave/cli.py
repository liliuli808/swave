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
    TrainingConfig,
    load_dataset_config,
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
        }.items()
        if value is not None
    }
    print(train(replace(config, **overrides)))
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
    predictor = ForwardPredictor.load(
        arguments.checkpoint, device=arguments.device
    )
    frequencies, curves = predictor.predict_with_frequencies(values)
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output, frequencies=frequencies, phase_velocity=curves
        )
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

    training = subparsers.add_parser(
        "train", help="train or resume the neural surrogate"
    )
    training.add_argument("--config")
    training.add_argument("--dataset-dir")
    training.add_argument("--output-dir")
    training.add_argument("--epochs", type=int)
    training.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps")
    )
    training.set_defaults(handler=_train)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        return int(arguments.handler(arguments))
    except (ArithmeticError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
