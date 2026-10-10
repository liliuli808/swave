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
norm (a near-zero denominator near mode-kissing frequencies inflates the
relative error). Decide based on its output: fix labels/augment data for the
affected kinds, or exempt small-norm rows from the 5 % relative threshold.

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

`scripts/check_kernel_label_noise.py` (finite-difference step sweep
1e-6/1e-5/1e-4) rules out label noise: drift medians are 0.02–0.06 % with
p95 ≤ 3.4 %, so the 5 % target is physically reachable. Kernel-labelled
training models are also kind-balanced (LVL 15 %, matching the train split).
The remaining cause is approximation capacity: thin anomalous zones (LVL and
coupled kinds) produce sharp, high-amplitude sensitivity kernels that one
shared MLP fits poorly. The hybrid inverter consumes the network Jacobian
(`torch.func.jacfwd`), so these rows are inversion sensitivities and matter.

**Next run: profile-feature conditioning.** `PROFILE_FEATURES=1` appends
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
