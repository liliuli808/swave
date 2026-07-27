"""Development-only generator for the immutable QEDispInv golden fixture.

This script is not used by the package or tests. It expects a locally compiled
helper around QEDispInv's ``Dispersion::search`` as its first argument.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from swave.empirical import material_properties


def _models() -> list[np.ndarray]:
    monotonic = np.linspace(0.35, 2.20, 20)
    low_velocity = np.linspace(0.40, 2.15, 20)
    low_velocity[5:8] *= 0.70
    coupled = np.linspace(0.40, 2.15, 20)
    coupled[5] *= 1.25
    coupled[6:9] *= 0.75
    paper = np.loadtxt("tests/fixtures/paper_model.txt")
    return [paper[:, 3], monotonic, low_velocity, coupled]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: generate_reference_fixture.py QED_HELPER MODE_HELPER"
        )
    qed_helper = Path(sys.argv[1]).resolve()
    mode_helper = Path(sys.argv[2]).resolve()
    frequencies = np.arange(0.5, 60.0 + 0.25, 0.5)
    models = _models()
    padded = np.full((len(models), 20), np.nan, dtype=np.float64)
    depth = padded.copy()
    density = padded.copy()
    vp = padded.copy()
    phase = np.full((len(models), 4, frequencies.size), np.nan)
    valid = np.zeros(phase.shape, dtype=np.bool_)
    degraded_phase = np.full_like(phase, np.nan)
    degraded_valid = np.zeros_like(valid)
    layer_count = np.array([len(values) for values in models], dtype=np.uint8)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for model_index, vs in enumerate(models):
            if model_index == 0:
                raw = np.loadtxt("tests/fixtures/paper_model.txt")
                current_depth = raw[:, 1]
                current_density = raw[:, 2]
                current_vp = raw[:, 4]
            else:
                current_depth = np.arange(20, dtype=np.float64) * 0.1
                current_vp, current_density = material_properties(vs)
                raw = np.column_stack(
                    [
                        np.arange(1, 21),
                        current_depth,
                        current_density,
                        vs,
                        current_vp,
                    ]
                )
            padded[model_index, : len(vs)] = vs
            depth[model_index, : len(vs)] = current_depth
            density[model_index, : len(vs)] = current_density
            vp[model_index, : len(vs)] = current_vp
            model_path = directory / f"model-{model_index}.txt"
            output_path = directory / f"qed-dispersion-{model_index}.txt"
            np.savetxt(model_path, raw, fmt="%.15g")
            subprocess.run(
                [
                    qed_helper,
                    model_path,
                    "0.5",
                    "60.0",
                    "0.5",
                    output_path,
                ],
                check=True,
            )
            result = np.loadtxt(output_path, ndmin=2)
            for frequency, velocity, mode in result:
                frequency_index = round((frequency - 0.5) / 0.5)
                phase[model_index, int(mode), frequency_index] = velocity
                valid[model_index, int(mode), frequency_index] = True
            degraded_output = directory / f"mode-dispersion-{model_index}.txt"
            subprocess.run(
                [
                    mode_helper,
                    model_path,
                    "0.5",
                    "60.0",
                    "0.5",
                    degraded_output,
                ],
                check=True,
            )
            degraded_result = np.loadtxt(degraded_output, ndmin=2)
            for frequency, velocity, mode in degraded_result:
                frequency_index = round((frequency - 0.5) / 0.5)
                degraded_phase[
                    model_index, int(mode), frequency_index
                ] = velocity
                degraded_valid[
                    model_index, int(mode), frequency_index
                ] = True

    np.savez_compressed(
        "tests/fixtures/golden_dispersion.npz",
        names=np.array(["paper", "monotonic", "low_velocity", "coupled"]),
        layer_count=layer_count,
        depth=depth,
        density=density,
        vs=padded,
        vp=vp,
        frequencies=frequencies,
        phase_velocity=phase,
        valid_mask=valid,
        degraded_phase_velocity=degraded_phase,
        degraded_valid_mask=degraded_valid,
        reference_repository=np.array("pan3rock/QEDispInv"),
        reference_commit=np.array("ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa"),
        reference_command=np.array(
            "QEDispInv Dispersion::search(frequency, 4), Rayleigh wave"
        ),
        degraded_reference_repository=np.array("pan3rock/mode-kissing"),
        degraded_reference_commit=np.array(
            "c8a1d2c67e83c95ae42fab863bdbaf347465f732"
        ),
        degraded_reference_command=np.array(
            "degraded-model search with Dispersion::search(frequency, 4)"
        ),
        generated_date=np.array("2026-07-27"),
        units=np.array("depth=km, velocity=km/s, density=g/cm3, frequency=Hz"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
