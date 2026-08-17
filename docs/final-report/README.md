# Formal Technical Report

This directory is the version-controlled delivery of the formal technical
report. It contains the LaTeX source, compiled PDF, figures referenced by the
source, and three small audit/evaluation evidence files.

## Compile

From this directory:

```bash
/tmp/tectonic-0.16.9/tectonic -X compile report.tex
```

Tectonic 0.16.9 was used for the archived PDF. The source uses paths relative
to this directory.

## Retained numerical sources

The report's aggregate metrics are checked against version-controlled sources
in the repository:

- `../../results/inversion-report/summary.json`
- `../../results/hybrid-inversion/tuning.json`
- `../../results/hybrid-inversion-report/summary.json`
- `../../runs/supervised-inversion-v2/evaluation.json`
- `../../runs/supervised-inversion-v2/run-identity.json`
- `evidence/dataset-audit.json`
- `evidence/forward-test-evaluation.json`

These files support summary-level verification of the published tables,
figures, identities, and derived percentages.

## Reproducibility boundary

The current Git checkout does **not** contain the large raw production
artifacts required to regenerate the experiments:

- `data/production` dataset shards and manifest;
- `runs/production-48g` forward-model checkpoints and training artifacts;
- `results/inversion` legacy inversion HDF5 shards;
- hybrid inversion HDF5 shards; or
- supervised `seed-*-best.pt` checkpoints.

Consequently, the checkout cannot rerun dataset auditing, training, inversion,
or raw-shard report generation. Stored hashes establish the identities recorded
by retained summaries but do not replace the missing bytes. The report labels
commands requiring these artifacts as conditional rebuild commands.

The full, deep, supervised, and hybrid aggregate numbers remain inspectable in
the retained JSON files. A new same-sample single-start comparison for the
400-model deep subset and a direct aligned-cell surrogate-versus-physical
prediction error cannot be computed from the retained summaries alone.

## Code verification

With the project environment activated, run from the repository root:

```bash
python -m pytest -q
ruff check src tests scripts
```

The 2026-08-17 verification completed 307 tests and passed Ruff.
