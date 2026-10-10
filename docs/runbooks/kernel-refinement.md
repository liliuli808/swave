# Runbook: forward-surrogate kernel refinement (Sobolev fine-tuning)

This document is written to be executed by an AI coding agent or a person on
the GPU machine that holds the production dataset. Follow the steps in order;
each command is idempotent and resumable.

## 1. Goal

Make the four-head forward network (`FourHeadForwardModel`) agree with the
physical dispersion solver (Python port of Pan & Chen's secular-function root
search, validated against QEDispInv to 1e-5 km/s) in **both** phase velocity
and sensitivity kernel `dc/dVs`.

Acceptance targets on the test split (sample IDs ending 85–89):

| Metric (`results/forward-kernel/summary.json`, key `finetuned`) | Target |
| --- | --- |
| `value.samples_all_within_1pct` — models whose 4 modes × 120 frequencies are all within 1 % | ≥ 0.99 |
| `kernel.rows_within_5pct` — (mode, frequency) kernel rows with relative L2 error ≤ 5 % | ≥ 0.99 |

Baseline measured on `runs/production-48g/best.pt` (before this work):

| Metric | Mode 0 | Mode 1 | Mode 2 | Mode 3 |
| --- | --- | --- | --- | --- |
| points within 1 % | 99.999 % | 99.995 % | 99.98 % | 99.94 % |
| kernel rows within 1 % | 89 % | 76 % | 59 % | 42 % |
| kernel rows within 5 % | 99 % | 96 % | 92 % | 84 % |

Overall baseline: `samples_all_within_1pct` = 0.9667, `kernel.rows_within_5pct`
= 0.921. Worst point error is 16 % against the raw shards but 6.4 % once the
kissing-pair corrections are applied — the 16 % cells were solver defects, not
network error. Baseline figures (raw shards, no corrections) are in
`results/forward-kernel-base/`.

A CPU end-to-end check of `scripts/run_kernel_refinement.sh` (1 epoch,
`BATCH_SIZE=16384`) completed all stages on 2026-10-07; one epoch is not
expected to meet the targets.

## 2. What changed and why

* `src/swave/kernels.py` — physical kernels from the Dunkin determinant by the
  implicit-function theorem `dc/dm = -(∂F/∂m)/(∂F/∂c)` at the dataset roots.
  Total derivative: Vp and density follow Vs through Brocher (2005), matching
  the network input. Verified against finite differences of re-solved roots
  (`tests/test_kernels.py`). ~13 ms per model per core.
* `src/swave/solver.py` — new `consensus` root strategy: union of Pan & Chen's
  truncated-model (degraded) roots and an 8-iteration quadratic search. The
  production data used `quadratic` alone, which occasionally skipped a
  mode-kissing pair (two roots ~1e-4 km/s apart) and shifted modes up by two.
* `results/kissing-repair/corrections.npz` — 140 re-solved cells over the 1 M
  models (train 114, validation 7, test 8, inversion 11). Produced by
  `scripts/repair_kissing_defects.py`; the original SHA-checked shards are not
  modified, the corrections are applied when data are loaded.
* `src/swave/kernel_training.py`, `scripts/finetune_forward_kernels.py` —
  fine-tunes the base checkpoint on (a) squared relative phase-velocity error
  and (b) squared relative error of random Jacobian–vector products `J v`
  versus `K v` (an unbiased estimate of the Frobenius kernel error), with
  hard-example resampling. Architecture and normalization are unchanged, so the
  output checkpoint is a drop-in replacement for inversion code.
* `scripts/evaluate_forward_kernels.py` — metrics and figures.
* `src/swave/network.py` — `model_from_checkpoint` reads an optional
  `architecture` entry, so wider networks can be loaded later.

## 3. Preconditions

* Repository at branch `forward-kernel-refinement`.
* `data/production/` with the 100 production shards (`shard-00000.h5` …
  `shard-00099.h5`).
* `runs/production-48g/best.pt` — the base forward checkpoint.
* Python env with the package installed: `python -m pip install -e ".[dev]"`.
* GPU with CUDA PyTorch (any GPU with ≥ 12 GB; everything is kept on the GPU,
  ~6 GB). Host RAM ≥ 16 GB. CPU-only also works but is ~10 min/epoch.

## 4. One-command execution

```bash
git fetch origin && git checkout forward-kernel-refinement && git pull
python -m pip install -e ".[dev]"
bash scripts/run_kernel_refinement.sh
```

Defaults: `DEVICE=cuda EPOCHS=50 BATCH_SIZE=4096 KERNEL_BATCH_SIZE=2048
LEARNING_RATE=2e-4 WARMUP_STEPS=300 KERNEL_WEIGHT=1.0`. Override any of these,
and the paths `DATASET_DIR`, `BASE_CHECKPOINT`, `OUTPUT_DIR`, `RESULTS_DIR`,
through environment variables. For a long run use `nohup` or `tmux`:

```bash
nohup bash scripts/run_kernel_refinement.sh > kernel-refinement.out 2>&1 &
```

Stages (the script logs `stage N:` lines):

| Stage | Work | Output | Typical time |
| --- | --- | --- | --- |
| 0 | preflight: 100 shards, checkpoint, corrections, CUDA, unit tests | — | 1 min |
| 1 | kernel labels: validation/test stride 5 (10 k each), train stride 4 (200 k) | `data/kernels/{split}/kernels-*.h5` (~8 GB) | 45 min on 6 cores; minutes on many cores |
| 2 | fine-tuning, one JSON line per epoch | `runs/kernel-finetune/{best,last}.pt`, `history.json`, `runs/kernel-finetune.log` | GPU: minutes per epoch |
| 3 | evaluation on the full test split and figures | `results/forward-kernel/*` | ~5 min |
| 4 | acceptance table, prints `ACCEPTANCE: PASS` or `NOT YET` | stdout | — |

Rerunning the script resumes: finished kernel files are skipped and training
continues from `last.pt`. To restart training from scratch delete
`runs/kernel-finetune/`.

## 5. Reading progress

Each epoch prints, for the validation subset (10 k models with kernels):

```json
{"epoch": 3, "value_loss": ..., "kernel_loss": ..., "seconds": ..., "score": ...,
 "samples_all_within_1pct": 0.97, "kernel_rows_within_5pct": 0.95, "kernel_median": 0.004}
```

`epoch -1` is the base model. `score = (1 - samples_all_within_1pct) + (1 -
kernel_rows_within_5pct)`; `best.pt` is the lowest score. Expected trend:
`kernel_rows_within_5pct` rises within the first epochs; `samples_all_within_1pct`
must not fall below the epoch -1 value for long.

## 6. Tuning if acceptance is NOT YET

**Current recommendation (2026-10-10):** use the corrected-gradient warm-start
experiment below. The earlier capacity and estimator-noise explanations were
working hypotheses; the mixed-derivative defect found on 2026-10-10 invalidates
using those runs alone to diagnose a capacity limit.

### Result of the first GPU run (2026-10-07, 50-epoch fine-tune, defaults)

| model | curves ≤ 1 % | points ≤ 1 % | max rel | kernel rows ≤ 5 % | kernel median |
| --- | --- | --- | --- | --- | --- |
| base | 96.674 % | 99.9755 % | 6.438 % | 92.129 % | 0.740 % |
| finetuned | 95.934 % | 99.9751 % | 7.133 % | 94.345 % | 0.721 % |

Kernel agreement improved only slightly while value agreement fell: the
1.4 M-parameter network (width 256, 4 blocks) cannot fit both the curves and
the abrupt M2/M3 kernel switches at mode-kissing frequencies. **Next run: a
wider network trained from scratch with the kernel loss** (7.6 M parameters):

```bash
OUTPUT_DIR=runs/kernel-wide RESULTS_DIR=results/forward-kernel-wide \
WIDTH=512 BLOCKS=6 EPOCHS=300 LEARNING_RATE=1e-3 WARMUP_STEPS=2000 \
nohup bash scripts/run_kernel_refinement.sh > kernel-wide.out 2>&1 &
```

Kernel labels from the first run are reused (stage 1 skips them). Epoch -1
prints `score 2.0` because training starts from random weights; that is
expected. The wide checkpoint loads through `ForwardPredictor` and the inversion
loader without changes (architecture is stored in the checkpoint).

### Result of the wide run (2026-10-08, width 512, 6 blocks, 300 epochs)

| mode | curves ≤ 1 % | kernel rows ≤ 5 % | kernel median |
| --- | --- | --- | --- |
| M0 | 99.97 % | 99.42 % | 0.19 % |
| M1 | 99.92 % | 98.50 % | 0.24 % |
| M2 | 99.85 % | 97.09 % | 0.30 % |
| M3 | 99.73 % | 94.99 % | 0.38 % |

Overall: curves 99.608 % (PASS), kernel rows ≤ 5 % 97.505 % (not yet).
Validation `kernel_rows_within_5pct` was still rising when the learning rate
reached its floor, so the run was under-trained rather than capacity-bound.
Kernel-labelled models used to be drawn uniformly; `KERNEL_HARD_POWER=1` now
resamples them in proportion to their worst kernel row. **Next run:**

```bash
OUTPUT_DIR=runs/kernel-wide-hard RESULTS_DIR=results/forward-kernel-wide-hard \
WIDTH=512 BLOCKS=6 EPOCHS=500 LEARNING_RATE=1e-3 WARMUP_STEPS=2000 \
KERNEL_WEIGHT=3 KERNEL_HARD_POWER=1 \
nohup bash scripts/run_kernel_refinement.sh > kernel-wide-hard.out 2>&1 &
```

Always use a new `OUTPUT_DIR` when changing settings: training resumes from
`OUTPUT_DIR/last.pt` without checking that the settings match.

### Results of the kw3 and width-768 runs (2026-10-08)

| model | curves ≤ 1 % | points ≤ 1 % | max rel | kernel rows ≤ 5 % | kernel median |
| --- | --- | --- | --- | --- | --- |
| base | 96.674 % | 99.9755 % | 6.438 % | 92.129 % | 0.740 % |
| kw3 finetuned (512×6, w=3, hard) | 99.584 % | 99.9974 % | 5.729 % | 97.562 % | 0.232 % |
| 768×8 finetuned (w=3, hard) | 99.562 % | 99.9970 % | 4.546 % | 97.583 % | 0.176 % |

Loss reweighting (`KERNEL_WEIGHT=3` + hard mining), double capacity, and more
epochs each moved kernel rows ≤ 5 % by ≤ 0.06 pt; validation plateaued at
~0.9747 with the learning rate at its floor in both runs. The remaining >5 %
rows are a hard tail, not an overall-fit problem (median 0.18 %). **Stop tuning
the network; diagnose the failing rows first:**

```bash
python3 scripts/diag_kernel_failures.py \
    --checkpoint runs/kernel-wide-768/best.pt \
    --dataset-dir data/production --kernel-dir data/kernels \
    --cache-dir data/cache --corrections results/kissing-repair/corrections.npz
```

It reports the failing rows by mode, frequency, model kind, and physical-kernel
norm. Use those groups to investigate labels or data coverage. Keep the
acceptance masks and 5% relative threshold fixed; do not exempt failing rows.

### Diagnosis of the 768 run's failing rows (2026-10-08)

`scripts/diag_kernel_failures.py` on the width-768 checkpoint showed the
>5 % rows are not a denominator artifact (failing rows have normal kernel
norms) and not frequency-localized; they track the model kind:

| mode | normal | low velocity | high velocity | coupled HVL+LVL |
| --- | --- | --- | --- | --- |
| M0 | 0.01 % | 1.49 % | 0.05 % | 0.49 % |
| M1 | 0.01 % | 3.54 % | 0.07 % | 1.63 % |
| M2 | 0.03 % | 6.22 % | 0.16 % | 3.71 % |
| M3 | 0.13 % | 9.93 % | 0.58 % | 6.82 % |

The earlier finite-difference sweep (1e-6/1e-5/1e-4) reported drift medians
of 0.02–0.06% and p95 <=3.4%. That script aggregated all valid frequencies
of each mode, so these statistics do not establish the accuracy of individual
failing rows. Its old test-split default also made it unsuitable for choosing
training changes. The row-level validation audit below supersedes that check.
Kernel-labelled training models have the expected kind mix (LVL 15%), but
that alone neither certifies coverage nor proves a network capacity limit.
The hybrid inverter consumes the network Jacobian (`torch.func.jacfwd`),
so these derivative errors matter for inversion.

**Historical experiment: profile-feature conditioning.** `PROFILE_FEATURES=1` appends
anomaly-localizing features (per-layer deficit vs running maximum, excess vs
future minimum, anomaly depths and contrasts — 46 dims) computed inside the
network, so the vs-only interface (inversion, evaluation) is unchanged and
the features flow through the Jacobian. Fresh training, 512×6:

```bash
OUTPUT_DIR=runs/kernel-wide-pf RESULTS_DIR=results/forward-kernel-wide-pf \
WIDTH=512 BLOCKS=6 EPOCHS=500 LEARNING_RATE=1e-3 WARMUP_STEPS=2000 \
KERNEL_WEIGHT=3 KERNEL_HARD_POWER=1 PROFILE_FEATURES=1 \
nohup bash scripts/run_kernel_refinement.sh > kernel-wide-pf.out 2>&1 &
```

A separate smaller issue: M3 rows at 0.5 Hz fail at 96 % — mode 3's kernel is
pathological near its cut-off frequency. Consider a cut-off-aware validity
mask in the kernel dataset builder if it remains material after the run
above.

### Result of the 2026-10-09 run and next step

| model | curves ≤ 1 % | points ≤ 1 % | max rel | kernel rows ≤ 5 % | kernel median |
| --- | --- | --- | --- | --- | --- |
| finetuned | 99.066 % | 99.9945 % | 6.483 % | 96.547 % | 0.508 % |

**Invalid run — not evidence about profile features.** The command was typed
on one line with the `> ` continuation prompts copied in, so bash treated them
as redirections: `WIDTH=512`, `KERNEL_WEIGHT=3` and `nohup` became empty files
and the checkpoint stored `width 256, blocks 6, kernel_weight 1.0`. Rerun the
profile-feature command above (with trailing `\`, or on one line without `>`)
after `rm -rf runs/kernel-wide-pf 'WIDTH=512' 'KERNEL_WEIGHT=3' nohup`.

Every run so far estimated the kernel error from `kernel_directions = 2` random
Gaussian directions out of 20 layers. That estimate is unbiased but noisy for
the sharp, few-layer kernels of LVL and coupled models — the rows that fail —
and the hard-example weights are computed from the same noisy estimate.
`KERNEL_DIRECTIONS=0` uses the 20 unit vectors instead, i.e. the exact
`‖J − K‖_F` of every row. It costs 10× the JVPs per kernel sample, so the kernel
batch is reduced to keep step cost and GPU memory near the previous runs.
Fresh training, 512×6, without profile features (one change at a time):

```bash
OUTPUT_DIR=runs/kernel-wide-exact RESULTS_DIR=results/forward-kernel-wide-exact \
WIDTH=512 BLOCKS=6 EPOCHS=500 LEARNING_RATE=1e-3 WARMUP_STEPS=2000 \
KERNEL_WEIGHT=3 KERNEL_HARD_POWER=1 KERNEL_DIRECTIONS=0 KERNEL_BATCH_SIZE=512 \
nohup bash scripts/run_kernel_refinement.sh > kernel-wide-exact.out 2>&1 &
```

Compare validation `kernel_rows_within_5pct` against the 512×6 run's history
at the same epoch. If it is not clearly ahead by epoch ~30, the estimator noise
was not the bottleneck: stop the run and return to the per-kind options above.

### 2026-10-10: fix mixed derivatives before another architecture experiment

The latest completed run is `kernel-wide-exact` (2026-10-09 22:19). Comparing
the saved test summaries gives:

| Run | Models with every valid value within 1 % | Kernel rows within 5 % | Median kernel error |
| --- | --- | --- | --- |
| wide, 512×6 | 99.608 % | 97.505 % | 0.263 % |
| kw3, 512×6 | 99.584 % | 97.562 % | 0.232 % |
| wide, 768×8 | 99.562 % | 97.583 % | 0.176 % |
| profile features | 96.344 % | 95.271 % | 0.933 % |
| exact Jacobian | 97.894 % | 96.458 % | 0.736 % |

Both acceptance rates must remain at least 99 %. A small median does not
remove the failing tail. In the kw3 test result, M0/M1/M2/M3 kernel pass rates
are 99.446/98.489/97.165/95.125 %, respectively.

The exact run did **not** converge to its reported summary: `best.pt` came
from epoch 252 (zero-based). At epoch 499 its validation curve pass rate was
0 % and kernel pass rate 11.152 %. Several intervening epochs had loss spikes;
at epoch 341 the training kernel loss reached 8.54 million. The profile-feature
run also selected an earlier checkpoint, epoch 223. These histories indicate
optimization instability, not just a persistent small tail.

#### Reproduced implementation defect

On local CPU PyTorch `2.14.1+cu130`, native `nn.LayerNorm` inside
`torch.func.jvp` produced the correct Jacobian loss **value**, but backpropagating
that loss gave incorrect gradients for parameters before the normalization.
A float64 16-wide, one-block reproduction showed 33–46 % relative gradient
errors in affected parameter groups. For one input weight:

| Gradient computation | Derivative of the same scalar kernel loss |
| --- | --- |
| Native LayerNorm, JVP then backward | −115.693467 |
| Reverse-mode Jacobian then backward | −209.348536 |
| Central finite difference in that weight | −209.348536 |

Writing LayerNorm as centered values, mean squared deviation, and reciprocal
square root reduced the maximum parameter-gradient discrepancy in that
reproduction from 93.66 to about `8.5e-14`. This uses the same normalization
formula, epsilon, and biased variance as [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).
`HigherOrderLayerNorm` implements this change and retains the existing
weight/bias keys, so old checkpoints still load. It fixes both random-direction
and exact-Jacobian loss backpropagation. The old tests checked loss values and
physical kernels, but did not check these mixed parameter/input derivatives.

`tests/test_kernels.py` now compares the parameter gradients against a
reverse-mode Jacobian and a scalar finite difference, for both loss modes and
CPU/CUDA where available. The pipeline runs these tests before training. The
GPU run's PyTorch version was not recorded, and CUDA is unavailable locally;
the defect is reproduced locally, while its contribution to the production
training failures must be verified by the corrected GPU experiment. Startup
logs and checkpoints now record the PyTorch version.

Two additional training defects are fixed:

* `BASE_CHECKPOINT` now loads weights whenever its architecture matches the
  requested one, including wide networks. Previously only 256×4 loaded weights;
  pointing at a wide checkpoint still initialized a random model.
* Epoch −1 is saved as `best.pt`, so a fine-tune that never improves cannot
  silently return its worse final model. Resuming also retains every completed
  history epoch instead of dropping one due to the epoch −1 record.

Training now rejects nonfinite losses/gradient norms before an optimizer update
and logs the mean/max gradient norm before clipping and clipping fraction.
These guards do not detect every finite-valued loss spike; inspect validation
and stop a deteriorating experiment rather than allowing another 500 epochs.

#### Other evidence and limits

An additional local check used 300 validation models, 75 per kind, sampled
from the first ten kernel shards with seed 20261010 (143,631 valid kernel
rows). Recomputing with steps `1e-6` versus `1e-5` gave p99 row disagreement
0.0323 %, with no rows above 5 %. Float16 storage versus the recomputed
`1e-5` labels gave p99 error 0.0469 %. A larger `1e-4` step gave p99 drift
3.286 % and 0.515 % of rows above 5 %. This sample does not support label
roundoff as the dominant explanation; it does not certify all training labels
or rule out root/branch defects in rare cases.

Keep `PROFILE_FEATURES=0` for the next experiment. That branch currently
computes anomaly features on layer-wise standardized inputs, not physical Vs.
All 75 monotone normal profiles in the above sample acquired nonzero anomaly
features after normalization, while none did in physical units. Its `argmax`
depth features also jump at ties and have zero derivative away from switches.
A future feature experiment should use physical Vs and smooth depth features,
with explicit checkpoint versioning; this change does not reinterpret existing
profile-feature checkpoints.

#### Next GPU experiment: short warm-start with corrected gradients

Use the kw3 checkpoint because its saved validation score (about 0.03026) is
better than the 768 checkpoint's (about 0.03093); it is also smaller. Keep the
test split for the final acceptance check. Use a **new** output directory and
a fresh optimizer, not the diverged exact run's `last.pt` or its optimizer.

```bash
BASE_CHECKPOINT=runs/kernel-wide-kw3/best.pt \
OUTPUT_DIR=runs/kernel-corrected-grad-v2 RESULTS_DIR=results/forward-kernel-corrected-grad-v2 \
WIDTH=512 BLOCKS=6 PROFILE_FEATURES=0 \
REQUIRE_WARM_START=1 MAX_INITIAL_SCORE=0.05 \
EPOCHS=50 LEARNING_RATE=2e-5 WARMUP_STEPS=300 MAX_GRAD_NORM=1 \
KERNEL_WEIGHT=1 KERNEL_DIRECTIONS=2 KERNEL_BATCH_SIZE=2048 \
HARD_EXAMPLE_POWER=0 KERNEL_HARD_POWER=0 \
nohup bash scripts/run_kernel_refinement.sh > kernel-corrected-grad-v2.out 2>&1 &
```

Set `PYTHON=python3` if that is the interpreter for the installed project.
Startup must report `"initialization": "base_checkpoint"`; epoch −1 should
approximately reproduce the kw3 validation rates (99.51 % curves, 97.464 %
kernel rows), allowing floating-point differences from LayerNorm. A score near
2 instead indicates incorrect initialization or incompatible data/settings.

The pipeline prepends this checkout's `src` to `PYTHONPATH`, verifies the loaded
module path and `HigherOrderLayerNorm`, and logs the Python executable. This
prevents an older installed `swave` from silently taking precedence.
`REQUIRE_WARM_START=1` requires matching base weights and rejects an existing
`OUTPUT_DIR/last.pt`; `MAX_INITIAL_SCORE=0.05` stops a bad initial evaluation
before any training update. These opt-in guards preserve deliberate scratch
training and resume behavior for the historical commands. To intentionally
resume a verified corrected run, unset `REQUIRE_WARM_START`; the initial-score
guard applies to fresh runs only.

These are conservative starting settings, not a guarantee of meeting 99 %.
The first 10–20 epochs should retain the curve pass rate and improve kernel
validation agreement without loss spikes. Finish the 50-epoch run only if it
remains stable. If stable but flat, compare `KERNEL_WEIGHT=3` in a separate
short run, then test more random directions at the same kernel batch size.
The old exact experiment changed both directions and batch size (2048→512),
so it was not an isolated test of estimator noise. Defer architecture changes,
feature conditioning, and row-targeted hard mining until the corrected baseline
is measured. Do not relax acceptance masks to hide failing rows.

#### Diagnosing the reported failed launch

The command reported on 2026-10-10 used the intended kw3 checkpoint and
`LEARNING_RATE=2e-5`, but its log started with epoch −1 score 2.0, followed by
epoch 0 curve/kernel pass rates both 0 %. Epochs 34–38 then slowly improved
from 11.82 to 13.91 % curves and 76.55 to 77.23 % kernel rows. This is a bad
starting state, not a collapse from the expected kw3 validation performance.

The pasted raw epoch lines also lack the learning-rate and gradient fields
always emitted by commit `cd8db0e`; no initialization record appears between
stage 2 and epoch −1. This strongly indicates an older training implementation
was executed (old checkout, installed package, process, or log). In the old
implementation width 512 ignored the base checkpoint weights and initialized
randomly, which is consistent with these observations. The remote source path
was not available, so the specific installation/process cause is not yet proven.

Stop that specific failed training process before relaunching. Update the GPU
checkout with `git pull --ff-only`; pulling does not update an already running
Python process. Run the guarded command above in a new output directory so the
failed run's `last.pt` and optimizer are not restored. Preserve the failed
directory for diagnosis; do not delete it or overwrite the kw3 checkpoint.
The expected initial score is about 0.03026. A score of 2 now stops immediately.

### Corrected v2 result: diagnose the remaining failures before extending training

The 50-epoch run reported on 2026-10-10 completed normally. Final **test**
evaluation of the checkpoint chosen by the original summed score was:

| checkpoint | curves within 1% | kernel rows within 5% | kernel median |
| --- | --- | --- | --- |
| kw3 base | 99.5840% | 97.5617% | 0.232% |
| corrected v2 | 99.6060% | 97.5884% | 0.226% |

The kernel pass rate improved only 0.0267 percentage points. Readable validation
records at epochs 43, 44, 46 and 48 remain around 97.53%, and the final learning
rate reached 1e-6. These results do not justify assuming that simply extending
this configuration to 500 epochs will meet 99%. The remote epoch-49 text was
partly corrupted; use the original history JSON for the exact final metrics.

First inspect the actual v2 validation failures, using the **run directory**
to compare the validation metrics of all available saved checkpoints:

```bash
PYTHONPATH=src python scripts/diag_kernel_failures.py \
  --checkpoint runs/kernel-corrected-grad-v2 \
  --split validation --device cuda \
  --output results/forward-kernel-corrected-grad-v2/diagnostics-validation.json \
  > kernel-v2-diagnostics.out 2>&1
```

Use `tail -n 80 kernel-v2-diagnostics.out` to view the report. This reads existing
weights and labels; it does not start another training run or overwrite them.
The directory form selects the strongest kernels among available checkpoints
whose validation curve pass rate is at least 99%. A file path instead diagnoses
that exact checkpoint. The report includes:

* The best summed-score epoch and best feasible kernel epoch from history,
  plus metrics for the last five training epochs.
* Which weights are still available (`best.pt`, `last.pt`, `best-kernel.pt`).
  A better historical epoch cannot be recovered if its weights were not saved.
* Failure counts and rates by mode, model kind and frequency, using **valid
  kernel rows** as the denominator. The old kind table included invalid cells
  in the denominator, so its failure rates should not be reused as exact rates.
* Error bands below 1%, 1–3%, 3–5%, 5–10%, 10–20% and at least 20%, physical
  kernel norms, finite-error quantiles and the worst failing sample/row IDs.

The diagnostic defaults to validation, supports CUDA, and counts exactly 5%
as a failure, matching the evaluator's strict `<0.05` test. Nonfinite predicted
errors count as failures and are reported separately from finite quantiles.
Keep test data for final acceptance. Earlier claims assigning the residual
tail solely to capacity or label noise are not established by these v2 results.

New training runs retain two independent best checkpoints: `best.pt` minimizes
the historical summed score, while `best-kernel.pt` maximizes kernel pass rate
subject to validation curve pass rate >=99% (curve rate breaks exact ties).
The pipeline evaluates `best-kernel.pt` when it exists. This prevents curve
improvement from compensating for kernel deterioration during selection, but
does not itself improve the trained weights. Resume preserves this checkpoint
and can seed it from available legacy weights and their matching history rows.

Choose the next experiment using the validation report. If failures have
ordinary physical norms and appear learnable, the first controlled comparison
is `KERNEL_WEIGHT=3`, keeping the same kw3 base, 50 epochs, batch size, two
directions, learning-rate schedule and hard-example powers at zero. Use a new
output directory and compare validation kernel improvement while retaining the
curve constraint. Do not simultaneously increase the weight, enable hard
sampling, add directions and extend the epoch count. If improvement remains
flat, compare more directions at the same batch size; targeted row weighting
or label checks should follow the observed failure groups. Do not relax the
acceptance masks or thresholds to remove the remaining failures.

### Weight-3 v3 result and exact training-row experiment

The corrected weight-3 run did not improve the baseline. The user verified the
stored best feasible kernel checkpoint, not just the final training epoch:

| available checkpoint | validation curves within 1% | validation kernels within 5% |
| --- | --- | --- |
| weight-1 v2 `last.pt`, epoch 49 | 99.5300% | 97.5319% |
| weight-3 v3 `best-kernel.pt`, epoch 46 | 99.4700% | 97.5220% |

Keep v2 as the better available candidate. These observations concern this
matched 50-epoch comparison; they do not show that kernel weighting can never
help. Do not infer convergence to 99% merely from a falling mean training loss.

The v2 validation diagnosis found 118,175 failing rows out of 4,788,089 valid
rows. Low-velocity and coupled models accounted for 98.64% of failures; modes
M2/M3 accounted for 78.63%. Physical kernel norms of failing rows were ordinary,
with none below 1% of the modal median. About 48.86% of failures exceeded 10%
relative error. This is not evidence to drop small-norm or cutoff rows from
acceptance, and it does not certify all labels or establish a capacity limit.

The next optional objective changes **which kernel rows receive updates**:

1. Every five epochs, sample up to 8,192 models from the kernel **training**
   split. Calculate exact row errors using all 20 layer directions, without a
   parameter-gradient graph, and select valid rows with relative error >=3%.
   The 3% mining threshold supplies margin below the unchanged 5% acceptance
   threshold. Rotate the pool; do not collect training IDs from validation.
2. Each optimizer step samples 256 selected (model, mode, frequency) rows.
   A reverse derivative with `create_graph=True` computes each full 20-layer
   kernel row exactly. This relies on the network having independent batch
   elements. The new row objective is additional to the existing value loss
   and uniformly sampled, two-direction kernel objective.
3. Use twice pseudo-Huber loss of the row's relative error in percent, with
   transition delta=5%. It matches squared error near zero and grows linearly
   for large residuals, reducing their influence **in the extra row objective**.
   The ordinary kernel objective remains squared error. Start the extra weight
   at 0.1 as an experiment, not as a validated optimum.

The derivative construction follows
[PyTorch autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html).
The robust scalar loss is checked against
[SciPy pseudo-Huber](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.pseudo_huber.html).
Tests also compare selected rows to a full reverse Jacobian, parameter gradients
to central differences, and an actual optimizer update to a known linear target.
The implementation rejects non-training split IDs and excludes invalid rows.

The dedicated recipe uses the same kw3 starting weights and baseline settings
as v2, with the extra row objective enabled. It checks the relevant CPU/CUDA
regressions first and runs training/validation, leaving test evaluation for a
selected candidate:

```bash
git pull --ff-only
nohup bash scripts/run_kernel_row_refinement.sh > kernel-corrected-rows-v4.out 2>&1 &
tail -n 30 -f kernel-corrected-rows-v4.out
```

Default output is `runs/kernel-corrected-rows-v4`; a used directory is rejected
by the fresh-warm-start guard. The default remains 50 epochs; mining adds work
and its time is logged separately. The original `run_kernel_refinement.sh`
also accepts `KERNEL_ROW_WEIGHT`, `KERNEL_ROW_BATCH_SIZE`,
`KERNEL_MINING_SAMPLES`, and `KERNEL_MINING_INTERVAL`; its extra row objective
defaults to zero so older experiments keep their original objective.

Inspect `kernel_row_mining` records first: `pool_rows_within_5pct` measures a
random training pool, and `selected_rows_by_mode` shows where extra updates go.
If the training pool is already much better than validation, investigate
generalization/data coverage instead of assuming insufficient optimization.
Epoch records add `kernel_row_loss`, `mined_rows`, and
`mined_row_pass_fraction`. The latter is a training-bank metric, not validation;
it can drop when the bank is refreshed and newly difficult rows enter it.
Judge improvement using the unchanged validation kernel rate and the >=99%
curve constraint. This method has not yet been validated on the remote GPU
run and is not a guarantee of reaching acceptance.

### Completed v4 and a targeted label consistency audit

The v4 `best-kernel.pt` is epoch 48, with validation curve pass rate 99.52%
and kernel pass rate 97.590270%. It is the best available kernel candidate
from these experiments, while remaining short of 99% acceptance. Compared
with v2 `last.pt`, the kernel gain is 0.058374 percentage points: a net decrease
of 2,795 failed rows, or 2.37% of v2's failures. Aggregate counts do not reveal
which individual rows became successes or regressed.

| v4 kernel-labelled split | models | valid rows | failed rows | kernel pass rate |
| --- | ---: | ---: | ---: | ---: |
| train | 200,000 | 95,760,563 | 1,703,579 | 98.2210% |
| validation | 10,000 | 4,788,089 | 115,380 | 97.5903% |

Training is also below target. The 0.6307 percentage-point train/validation
gap is not evidence that generalization is the only bottleneck, and these
metrics do not distinguish optimization, conflicting labels, and capacity.
On validation, M2/M3 account for 78.48% of failures, and low/coupled models
for 98.64%. There are still 28,266 rows with error >=20% (v2: 28,388).
Reaching 99% requires a net reduction of at least 67,500 failures. Even fixing
all 59,408 rows in the 5–10% band would only reach 98.8310%.

Before another long run, audit the actual worst rows recorded in the JSON:

```bash
git pull --ff-only
PYTHONPATH=src python scripts/check_kernel_label_noise.py \
  --split validation \
  --diagnostics results/forward-kernel-corrected-rows-v4/diagnostics-validation.json \
  --output results/forward-kernel-corrected-rows-v4/label-audit-validation.json
```

The audit uses CPU physics and does not require the neural checkpoint or a
GPU. It loads only the models needed for the diagnostic's worst rows and
four random reference models per kind. Each reference model supplies one
valid row per mode. These are background references, not verified neural
successes; neither group estimates population prevalence. IDs and split
must agree with the diagnostic. No labels or acceptance masks are changed.

The report compares individual 20-component kernel rows:

- `stored_vs_recomputed`: training's float16 label versus fresh float64
  evaluation at the same stored phase velocity, alongside `float16_rounding`.
- `step_1e-6_at_stored_root` and `step_1e-4_at_stored_root`: changes relative
  to the existing 1e-5 step, without aggregating over other frequencies.
- `resolved_vs_stored`: kernel after re-searching all four roots with the
  consensus solver and root tolerance 1e-11, versus the training label.
  `phase_relative_shift` records the accompanying phase change. The two
  step comparisons are repeated at the resolved root.

Root tolerance controls numerical root refinement, as documented for the
solver's [SciPy TOMS748](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.toms748.html)
routine; it does not guarantee that a search found every physical branch.
Failed/unresolved checks remain explicit in `unresolved_rows`. All these
checks share the existing secular function, so agreement is not independent
physical validation. A difference at the larger 1e-4 step alone also does not
establish an error in the 1e-5 label; inspect smaller-step convergence and
root changes before deciding on corrections.

If severe failing labels disagree, first investigate their roots, mode
identities and derivative convergence. If these checks agree, the next
controlled experiment is fitting a fixed small set of difficult **training**
models to test optimization/representation limits before expanding the full
network or dataset. The v4 metrics alone do not justify either diagnosis.

### 2026-10-10: the capacity plateau was never measured with correct gradients

Every from-scratch run that defined the ~97.5% plateau (wide, kw3, 768×8,
profile features, exact) finished before commit `cd8db0e`, i.e. it trained the
kernel objective with the native-LayerNorm mixed-derivative defect (33–46%
parameter-gradient errors in the reproduction). The only corrected-gradient
runs (v2, v3, v4) are 50-epoch fine-tunes at `LEARNING_RATE=2e-5` starting from
the kw3 weights, which can move only a short distance from the optimum found
with incorrect gradients. "Width, epochs and loss weights do not help" is
therefore not established; it is the next thing to test, not a conclusion.

Local CPU checks on 200 deterministic production models (sample IDs 0–199,
200 models: 45 normal, 35 LVL, 22 HVL, 98 coupled) also weaken the other
explanations:

* Production (`quadratic`) roots and `consensus` roots agree in every cell
  (0 differing cells), so these models show no missed kissing pairs.
* Re-solving after a random 0.01 km/s per-layer perturbation, the fraction of
  rows whose physical kernel changes by more than 5% is similar across kinds
  (M3: normal 28.7%, LVL 34.0%, HVL 33.6%, coupled 31.3%). Measured this way,
  LVL kernels are only modestly less smooth, which does not explain a
  ~70× higher M3 failure rate (9.93% vs 0.13%).
* Rows within 0.1% of an adjacent mode (osculation) are sharp (54% change
  >5%) but rare (0.2% of rows), far fewer than the 2.5% failing rows.

This sample is small and does not prove the cause; it shows that the defective
gradients are now the leading untested explanation.

#### Next experiment: the kw3 recipe from scratch with corrected gradients

The v4 numbers above already answer the train/validation question: training
rows pass at 98.22% and validation at 97.59%, so the network does not fit its
own training rows either. Labelling more training models is therefore not
expected to reach 99% alone. What has not been run is full-strength training
with correct gradients. All settings are kw3's, so the only change is the
gradient fix (~23 s/epoch, about 3–4 h for 500 epochs, ~6 GB GPU):

```bash
OUTPUT_DIR=runs/kernel-wide-kw3-fixed RESULTS_DIR=results/forward-kernel-wide-kw3-fixed \
WIDTH=512 BLOCKS=6 EPOCHS=500 LEARNING_RATE=1e-3 WARMUP_STEPS=2000 \
KERNEL_WEIGHT=3 KERNEL_HARD_POWER=1 \
nohup bash scripts/run_kernel_refinement.sh > kernel-wide-kw3-fixed.out 2>&1 &
```

Do not set `REQUIRE_WARM_START`: the 256×4 base checkpoint only supplies
normalization, so epoch −1 prints score 2.0 as in the kw3 run. Compare
validation `kernel_rows_within_5pct` with the kw3 history
(`results/forward-kernel-wide-kw3/finetune-history.json`) at the same epoch:
kw3 had 0.9585 at epoch 249, 0.9684 at 349 and 0.9742 at 449. Clearly ahead by
epoch ~250 means the defect was the bottleneck: let it finish, and then warm
start the v4 row objective from its `best-kernel.pt`. Behind or equal means it
was not: stop it and follow the label audit and small-set fitting test in the
previous section.

The label audit of the previous section runs on CPU and needs no checkpoint,
so run it at the same time. This GPU run does not depend on its outcome, but
severe disagreements in the audited rows would take priority over any
training change.

### 2026-10-10 results: labels agree, corrected gradients lift the plateau

**Label audit** (`results/forward-kernel-corrected-rows-v4/label-audit-validation.json`,
84 rows from 19 models). The 20 worst v4 validation rows (network relative L2
193–350%, all coupled models, mostly M2/M3) and 64 random reference rows
show no label problem:

| check | worst rows: max relative L2 | reference rows: max relative L2 |
| --- | ---: | ---: |
| `stored_vs_recomputed` (float16 vs float64) | 0.033% | 0.048% |
| `resolved_vs_stored` (consensus roots, tol 1e-11) | 0.032% | 0.048% |
| `step_1e-6_at_stored_root` | 8.1e-8 | 0.034% |
| `step_1e-4_at_stored_root` | 8.1e-6 | 3.4% (4 rows >1%, none >5%) |

No row is unresolved, and the largest phase shift after re-searching roots is
6.9e-8. The stored-label differences equal float16 rounding. The worst failures
are network errors, not wrong roots, mode identities or derivative steps. These
checks share one secular function, so this is not independent physical
validation.

**kw3 recipe from scratch with corrected gradients**
(`runs/kernel-wide-kw3-fixed`, commit `f285dc9`, GPU 1, 3.5 h). Validation
`kernel_rows_within_5pct` against the old kw3 run at the same epoch:

| epoch | fixed | kw3 |
| ---: | ---: | ---: |
| 100 | 0.9544 | 0.9314 |
| 249 | 0.9713 | 0.9585 |
| 349 | 0.9774 | 0.9684 |
| 449 | 0.9799 | 0.9742 |
| 499 | 0.9801 | 0.9747 |

It led at the epoch-249 check by 1.28 points and was allowed to finish.
`best-kernel.pt` is epoch 493 (validation kernel 98.005%, curves 99.84%).
Test acceptance:

| model | curves<=1% | points<=1% | max rel | kernel<=5% | kernel med |
| --- | ---: | ---: | ---: | ---: | ---: |
| kw3 | 99.5840% | 99.9974% | 5.729% | 97.5617% | 0.232% |
| kw3-fixed | 99.8460% | 99.9991% | 3.644% | 98.1160% | 0.141% |

Test kernel pass rate by mode: M0 99.61%, M1 98.87%, M2 97.77%, M3 96.20%.
Acceptance is still NOT YET. The gradient defect was a real bottleneck: the same
recipe gains about 0.55 points and beats every earlier candidate, including
v4 (97.59% validation). The last 100 epochs add only 0.02 points (validation
0.9790 → 0.9801, with the learning rate decayed), so a longer run with the
same recipe is not expected to close the remaining gap.

#### Next run: v5, the row objective warm-started from kw3-fixed

This is the v4 recipe (`scripts/run_kernel_row_refinement.sh`, 50 epochs,
`LEARNING_RATE=2e-5`, `KERNEL_ROW_WEIGHT=0.1`) with only the starting weights
changed:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHON=.venv/bin/python \
BASE_CHECKPOINT=runs/kernel-wide-kw3-fixed/best-kernel.pt \
OUTPUT_DIR=runs/kernel-fixed-rows-v5 \
setsid nohup bash scripts/run_kernel_row_refinement.sh > kernel-fixed-rows-v5.out 2>&1 < /dev/null &
```

The warm-start guard score (2 − curve rate − kernel rate) is about 0.02, under
the script's 0.05 limit. Compare validation `kernel_rows_within_5pct` with
the starting 0.98005: v4 added only 0.058 points to its own start.

The first `kernel_row_mining` record of v5 changes the diagnosis. On a random
pool of 8,192 training models (3.92 M rows), kw3-fixed already passes
**99.65%** of rows (M0 99.98%, M1 99.92%, M2 99.71%, M3 99.00%). Validation is
98.01%. v4's starting model passed 98.19% of the same pool. Once the gradients
are correct, the network fits its training rows above the 99% target, and
the remaining gap (~1.6 points) is between training and validation. The v5
bank also shrinks from 130,510 to 50,189 rows (error >3%).

The row objective trains only on training rows, so v5 is expected to widen
this gap rather than close it. Still let it finish (50 epochs), because it costs
little and confirms the effect. The small-set fitting test is no longer
needed: capacity and optimization are not the limit on training rows. Next,
look at generalization and data coverage. Options include more labelled kernel
training models (only 200,000 of the production models have kernels),
regularization or weight decay (`weight_decay` is 0), and early stopping on
validation kernel rate. Also compare the train/validation gap by model kind
and mode before choosing.

### Earlier tuning knobs (historical; validate corrected gradients first)

1. Kernel target missed, values fine: `KERNEL_WEIGHT=3`.
2. Values regress during training: `LEARNING_RATE` halved or `KERNEL_WEIGHT=0.3`.
3. Both still improving at the last epoch: more `EPOCHS`.
4. Wide model plateaus below target: `WIDTH=768 BLOCKS=8`.

## 7. Report back

Return these to the requester:

* the stage-4 acceptance table (stdout),
* `results/forward-kernel/summary.json`,
* `results/forward-kernel/finetune-history.json`,
* the figures `results/forward-kernel/*.png`.

Commit the run's `RESULTS_DIR` (small) on the branch, together with the
`.out` log copied into it. Do **not** commit
`data/` or `runs/` (gitignored, large). Do not modify the production shards.

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `DEVICE=cuda but torch.cuda.is_available() is False` | install CUDA PyTorch, or run with `DEVICE=cpu BATCH_SIZE=1024 KERNEL_BATCH_SIZE=512` |
| `expected 100 shards` | set `DATASET_DIR` to the production dataset |
| process killed / out of host memory | ~7 GB host RAM is needed while loading; close other jobs |
| CUDA out of memory | halve `BATCH_SIZE` and `KERNEL_BATCH_SIZE` |
| `checkpoint split policy does not match` | the base checkpoint must be the `mod100-v2-80-5-5-10` one |
