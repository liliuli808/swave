# Surface-Wave Forward Modeling

`swave-forward` is a Python-native Rayleigh-wave forward solver and neural
surrogate for layered models containing low-velocity zones. It generates
deterministic geological models, resolves closely spaced “mode-kissing” roots,
writes resumable HDF5 training data, and trains a shared-backbone network with
four modal output heads.

The fixed production problem is:

- 20 `Vs` values from 0.30 to 2.60 km/s;
- 19 finite 0.1 km layers plus a half-space beginning at 1.9 km;
- frequencies 0.5–60.0 Hz at 0.5 Hz spacing (120 values);
- Rayleigh modes 0, 1, 2, and 3;
- normal, low-velocity, high-velocity, and coupled high-then-low model families.

All public depths are in km, velocities in km/s, frequency in Hz, and density
in g/cm³.

## Install

Python 3.11 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
swave --help
```

The runtime does not build, invoke, link, or shell out to either reference C++
project. The Dunkin delta-matrix implementation is in Python and its scalar hot
path is cached by Numba using the LLVM runtime included in its binary wheel. A
C++ compiler is not required. The first solver call in each environment includes
a one-time JIT compilation delay.

## Generate data

A small end-to-end generation run:

```bash
swave generate \
  --config configs/dataset.toml \
  --samples 16 \
  --workers 1 \
  --output-dir data/smoke
```

The production configuration contains one million accepted samples in 100
shards:

```bash
swave generate --config configs/dataset.toml
```

Generation is deterministic from `(global_seed, sample_id, retry_count)`.
Completed shards are validated and skipped, so running the same command resumes
an interrupted job. Changing a configuration covered by the manifest hash is
rejected rather than mixed into an existing dataset.

Each `shard-NNNNN.h5` contains:

| Dataset | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `sample_id` | `(N,)` | `uint64` | Stable sample identity |
| `model_kind` | `(N,)` | `uint8` | Normal/LVL/HVL/coupled family |
| `vs`, `vp`, `density` | `(N, 20)` | `float32` | Layer properties |
| `phase_velocity` | `(N, 4, 120)` | `float32` | Modal targets; missing cells are `NaN` |
| `valid_mask` | `(N, 4, 120)` | `bool` | Cells valid for training and metrics |
| `quality_flags` | `(N,)` | `uint16` | Recovery and quality bit flags |
| `retry_count` | `(N,)` | `uint8` | Deterministic model retry used |

Files are gzip-compressed, written to process-specific temporary paths, and
atomically renamed only after a complete close. `manifest.json` records the
configuration hash, package version, creation time, completed shards, per-shard
SHA-256 digests, accepted/rejected family counts, rejection reasons, and
recovery count. Resume verifies every required dataset, shape, dtype, sample-ID
digest, material-property invariant, curve mask, and file checksum before a
shard is accepted as complete.

## Train and evaluate

The split is stable across shards: `sample_id % 100` assigns 0–89 to training,
90–94 to validation, and 95–99 to testing. Consequently, a training smoke run
needs sample IDs spanning at least 0–95; use at least 100 generated samples.

```bash
swave train \
  --config configs/training.toml \
  --dataset-dir data/production \
  --output-dir runs/smoke \
  --epochs 1 \
  --num-workers 0 \
  --device cpu
```

Production training uses the defaults in `configs/training.toml`:

```bash
swave train --config configs/training.toml
```

Training streams HDF5 rows, computes normalization from valid training cells
only, balances the masked Smooth-L1 loss across nonempty modes, and writes
`last.pt`, `best.pt`, and `history.json`. Repeating the command resumes
`last.pt` when `resume = true`. Training and evaluation first require the exact
completed shard set and recheck every manifest SHA-256; evaluation also requires
the checkpoint dataset hash to match.

```bash
swave evaluate \
  --checkpoint runs/default/best.pt \
  --dataset-dir data/production
```

Evaluation reports per-mode MAE, RMSE, 95th-percentile absolute error, and valid
cell counts in physical units.

## Predict and plot

Pass exactly 20 `Vs` values:

```bash
swave predict runs/default/best.pt \
  0.40 0.48 0.56 0.64 0.72 0.80 0.88 0.96 1.04 1.12 \
  1.20 1.28 1.36 1.44 1.52 1.60 1.68 1.76 1.84 1.92
```

For a text matrix with one model per row, use `--input-file models.txt`.
`--output prediction.npz` writes `frequencies` and `phase_velocity` arrays.

```bash
swave plot-model --sample-id 4 --output figures/model-4.png
swave plot-dispersion --sample-id 4 --output figures/dispersion-4.png
swave plot-history \
  --history runs/default/history.json \
  --output figures/training.png
```

Python callers, including an external inversion program, can use the stable
NumPy interface:

```python
import numpy as np
from swave.inference import ForwardPredictor

predictor = ForwardPredictor.load("runs/default/best.pt", device="cpu")
vs = np.linspace(0.4, 2.0, 20)
frequencies, phase_velocity = predictor.predict_with_frequencies(vs)
# phase_velocity.shape == (4, 120)
```

Batch input with shape `(N, 20)` produces `(N, 4, 120)`.

To compare one test-set `Vs` profile against its true and predicted M0–M3
curves in a five-panel figure, run:

```bash
python scripts/plot_five_panel_comparison.py \
  --dataset-dir data/production \
  --checkpoint runs/production-48g/best.pt \
  --output results/five-panel-comparison.png
```

Without `--sample-id`, the script selects the first coupled model in the test
split. The printed JSON reports the selected sample, output path, and per-mode
MAE and relative error. The script verifies the complete dataset and checkpoint
dataset hash before plotting.

## Numerical methods and quality policy

The solver propagates the unnormalized Rayleigh-wave Dunkin delta matrix and
uses TOMS 748 on sign-changing brackets. Initial sampling follows the modal
count estimate used by the reference projects. Two recovery strategies are
available:

- `degraded`: roots from shallower degraded models supplement the full model;
- `quadratic` (default): quadratic extrema add samples around same-sign
  intervals that may contain a close root pair.

The quadratic coarse scan stops after four requested roots plus two bias roots,
matching the reference algorithm’s bounded search. Valid leading gaps in higher
modes remain masked. Internal gaps or numerical failures receive one bounded
refinement pass and then cause deterministic model regeneration; the code never
silently fills missing physical targets with zeros or interpolation.

## Measured production estimate

The following CPU benchmark was recorded on 2026-07-27 with the same production
physics and geology configuration. A complete 100-model shard ran in one worker;
the separate evaluation-count sample used the first 10 deterministic models.
Projections assume ideal linear shard-level scaling and therefore are planning
estimates, not completion guarantees.

| Measurement | Result |
| --- | ---: |
| Accepted models | 100/100 |
| Wall time | about 210 s |
| Throughput | 0.476 models/s |
| Mean secular evaluations/model (10-model sample) | 13,057 |
| Recovery/retry rate | 0% / 0% |
| Compressed shard bytes/model | 1,359 bytes |
| Projected 1,000,000-model time, 1 worker | 24.3 days |
| Projected time, 4 workers | 6.1 days |
| Projected time, 100 concurrent shards | 5.8 hours |
| Projected compressed storage | 1.36 GB |

The detected host reports 192 logical CPUs, so `workers = 0` resolves to 191
processes, but the 100 production shards limit useful concurrency to 100. Real
throughput will normally be lower because of CPU contention, memory bandwidth,
JIT startup, difficult models, retries, and filesystem load. Run a representative
shard on the target host and set `--workers` explicitly before committing to the
full production job.

## Reproduce checks

```bash
python -m pytest -q
ruff check src tests
python -m build
```

## Sources

- Lei Pan and Xiaofei Chen, “Efficient Computation of Dispersion Curves in
  Low-Velocity-Layered Half-Spaces,” supplied as `../bssa-2025003.1.pdf`;
  reference code: [pan3rock/mode-kissing](https://github.com/pan3rock/mode-kissing).
- Lei Pan, Jiannan Wang, and Xiaofei Chen, “Surface-Wave Dispersion Curve
  Computation and Inversion: A Framework Integrating Quadratic Extrema
  Interpolation and Randomized Layering with Multiple Initial Models,” supplied
  as `../bssa-2025207.1.pdf`; reference code:
  [pan3rock/QEDispInv](https://github.com/pan3rock/QEDispInv).

The reference repositories informed the algorithms and test fixtures but are
not runtime dependencies. See the licenses in those repositories before reusing
their original source code.
