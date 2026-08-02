# Final Review Fix Report

## Outcome

The coordinated final-review fix wave is complete on `main`.

- Base: `e21add3da3c5a67bf8916b22cbcd084e6e1a2842`
- Implementation: `dd90c9fa6c4424617c4bfc59bfe6981c0a1b8c8c`
  (`fix: harden inversion experiment integrity`)
- Scope: one Critical, four Important, and one Minor final-review finding
- Independent re-review: approved with no remaining Critical or Important
  findings; all four high-risk probes passed and the diff was clean

The formal million-sample generation, production training, 100,000-row full
inversion, and 400-row/100-start deep experiment were not run. This wave used
unit, integration, real-spawn, corruption, production-topology, and packaging
checks only.

## Findings Resolved

### 1. Bounded deep scheduling and recovery

Deep work is no longer grouped into one large job per source shard. The selected
population is globally ordered by `sample_id` and split into deterministic,
bounded chunks. The default is 10 samples per job, and each job name binds its
first ID, last ID, and the first 12 hexadecimal characters of its ordered-ID
digest:

```text
deep-{noise}-samples-{first:020d}-{last:020d}-{digest[:12]}
```

For the formal 400-row, two-noise deep topology, the production-scale test
proves 80 jobs, at most 10 rows and 1,000 starts per job, exactly 20 jobs for
each of four task indices, disjoint assignment, complete coverage, and exact
ordered sample coverage in both noise scenarios.

CPU submission now maintains at most one in-flight job per worker instead of
submitting the whole run. Spawned workers receive explicit Torch, OpenMP, MKL,
OpenBLAS, NumExpr, and vecLib thread limits. A surrogate is still loaded once
per job. `deep_samples_per_job` and `threads_per_worker` are validated config
and CLI controls. They are operational rather than scientific hash inputs, but
changing chunk size changes the manifest's expected job/population identity and
therefore rejects an incompatible resume.

### 2. Separate inversion and physical-reconstruction outcomes

For deep results, `success`, `status`, and `failure_code` now describe the
inversion ensemble. Physical reconstruction has separate `physical_success`,
`physical_status`, and `physical_failure_code` fields.

A sufficient ensemble remains an inversion success when the physical solver
fails. Its median model, surrogate curve, P10/P90 interval, and optimization
diagnostics remain available. A total physical failure publishes no physical
cells, so it contributes its full observed-cell count to physical missingness
instead of disappearing from the denominator. Partial physical masks remain
canonical: valid cells are positive and finite, while every invalid cell is
`NaN`.

Reports expose separate inversion and physical outcome counts, statuses, and
failure codes. Paired clean/noisy results also include physical-success-fraction
change.

### 3. Auditable per-start diagnostics

Result schema v3 adds per-start:

- iterations and evaluations;
- initial and final objective;
- success, optimizer status, stable failure code, and bounded printable UTF-8
  message;
- recovered model and IQR inlier flag.

Schema validation enforces shapes, dtypes, nonnegative effort, finite-or-`NaN`
objective semantics, failure-code consistency, valid messages, bounded models,
and the relationship among start success, IQR inliers, sample success, and the
configured minimum valid-solution count. The runner's message encoder also
avoids splitting a multibyte character at the 512-byte storage boundary.

Reports now distinguish start convergence from sample sufficiency and physical
reconstruction. They include status/failure/message counts, IQR rejection,
iterations/evaluations, initial/final objectives, and aligned clean/noisy start
convergence and effort deltas. The optimization figure uses the actual
per-start diagnostics for deep results.

### 4. Exact zero-IQR behavior

`iqr_inlier_mask()` always applies the inclusive 1.5-IQR fences. There is no
zero-IQR keep-all branch. Consequently, nine objective values at 1 and one at
100 retain the nine values at the zero-width fence and reject 100. The design,
plan, README, implementation, and regression test now state the same rule.

### 5. Immutable dataset, population, and software identity

Result schema is now version 3. The manifest and every shard bind:

- the canonical dataset manifest digest, including source-shard checksums;
- the existing dataset configuration, checkpoint, split-policy, and scientific
  inversion identities;
- an installed-software digest over package version plus sorted `swave` Python
  relative paths and source bytes;
- each expected job's exact sample count and ordered `uint64` sample-ID digest.

Initialization, publication, recovery, completion, and report validation enforce
these identities. Complete-result validation independently rejects duplicate
IDs within a scenario, while per-job identities reject a clean/noisy omission
even when both scenarios omit the same rows.

Report construction validates the complete result identity and all result
shards first, then checksum-validates the dataset and its bound manifest digest,
then validates result alignment and configuration. Only after those gates does
it read the exact requested truth rows.

Schema-v2 result directories are deliberately not migrated: attempting to load
one produces an explicit instruction to create a new schema-v3 result directory.
This prevents partially completed old and new scientific contracts from being
combined.

### 6. Strict pre-execution configuration validation

`InversionConfig` now rejects malformed types, booleans in integer fields,
non-finite or negative weights and regularization, invalid tolerance, negative
seeds, duplicate noise scenarios, unsupported Vs bounds outside `[0.3, 2.6]`,
invalid ensemble counts, and invalid worker/thread/chunk/task controls. TOML
arrays are canonicalized to tuples before this validation. The public CLI uses
`dataclasses.replace()`, so overrides revalidate the complete object before any
dataset scan or result-manifest creation.

## TDD Evidence

The first focused run after adding the review regressions failed at the intended
missing behaviors:

```text
28 failed, 135 passed
```

Those failures covered production deep topology, rolling queue/thread controls,
physical-outcome separation, physical missingness, per-start storage/reporting,
zero-IQR fences, dataset/population/software identity, v2 rejection, report
truth-access order, and strict config validation.

A final corruption edge was also observed red before its fix:

```text
FAILED test_encoded_optimizer_message_is_bounded_printable_utf8
E   UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 510-511
1 failed
```

After truncating only at a valid UTF-8 boundary and normalizing nonprintable
characters, that test and the explicit software/chunk-resume identity checks
passed:

```text
3 passed in 1.39s
```

Intermediate focused suites passed independently: configuration/inversion/
dataset 50 tests, result schema 54 tests, runner 19 tests, report 28 tests, and
CLI 12 tests. The real spawned-process integration test also passed on its own:

```text
1 passed in 2.97s
```

## Final Verification

Fresh full repository suite after the last source and test change:

```text
$ ../.venv/bin/pytest -q
226 passed in 83.73s (0:01:23)
```

Static, touched-file formatting, and diff-integrity checks:

```text
$ ../.venv/bin/ruff check src tests scripts
All checks passed!

$ ../.venv/bin/ruff format --check \
    src/swave/config.py src/swave/dataset.py src/swave/inversion.py \
    src/swave/inversion_results.py src/swave/inversion_runner.py \
    src/swave/inversion_report.py src/swave/cli.py \
    tests/test_config.py tests/test_dataset.py tests/test_inversion.py \
    tests/test_inversion_results.py tests/test_inversion_runner.py \
    tests/test_inversion_report.py tests/test_cli.py
14 files already formatted

$ git diff --check
# no output
```

A repository-wide format check additionally identified 20 pre-existing,
unrelated files that Ruff would reformat. They were intentionally not changed in
this focused wave; every touched Python file is format-clean as shown above.

The isolated build initially could not download its build requirements inside
the network-restricted sandbox. After the required approved network retry, both
artifacts built successfully:

```text
$ ../.venv/bin/python -m build
Successfully built swave_forward-0.1.0.tar.gz and
swave_forward-0.1.0-py3-none-any.whl
```

## Independent Re-review

The original independent final review supplied the six findings addressed in
this wave. The same reviewer then inspected the complete worktree diff against
`e21add3`, including report-delta and concurrent-publication edge cases. The
verdict was approval with no remaining Critical or Important findings; all four
high-risk checks passed and the diff was clean.

This report is committed after the implementation commit above. Its own commit
SHA is recorded in the final handoff because a commit cannot embed its own hash.
