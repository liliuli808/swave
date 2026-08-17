# Formal Report High-Priority Fixes Design

Date: 2026-08-17

## Objective

Repair the three high-priority validity and handoff problems identified in the
formal technical report without inventing results or changing any archived
experiment output.

The revised report must:

1. avoid attributing a cross-population MAE difference to multi-start inversion;
2. avoid treating two observation-residual metrics as a direct surrogate-versus-physical error measurement; and
3. distinguish archived-summary verification from raw-artifact reproducibility,
   while placing the report and its retained supporting evidence under version
   control.

## Scientific Narrative Changes

### Multi-start experiment

The full single-start experiment contains 100,000 unique models in the natural
production mixture. The deep multi-start experiment contains 400 unique models,
with exactly 100 from each model family. Their aggregate MAEs therefore have
different population weights and cannot estimate a multi-start treatment effect.

The report will retain the deep experiment's absolute MAE, interval coverage,
start diagnostics, and physical revalidation results. It will remove the claimed
11% improvement and state that a same-sample single-start comparator is required
to quantify improvement. The abstract, main result section, discussion, and
conclusion will use the same limitation.

### Surrogate and physical residuals

The archived summary reports the surrogate-to-observation residual and the
physical-solver-to-observation residual for recovered models. It does not report
the direct discrepancy between surrogate and physical predictions, and the two
residual calculations do not have an identical valid-cell population.

The report will retain both residual values but interpret them only as evidence
that the assessed fit depends on the evaluation operator. It will explicitly say
that direct surrogate error on recovered models remains unquantified and requires
an aligned-cell metric such as
`MAE(F_surrogate(v_hat), F_physical(v_hat))`.

## Archive and Reproducibility Changes

The appendix will separate three levels of evidence:

1. **Versioned and directly inspectable:** report source/PDF, selected figures,
   summary JSON files, manifests, tuning JSON, supervised histories/evaluation,
   and small audit evidence files.
2. **Numerically verifiable from retained summaries:** published aggregate
   tables, identities, checksums, and derived percentages.
3. **Not reproducible from the current Git checkout:** production dataset,
   forward checkpoints, legacy inversion HDF5 shards, hybrid HDF5 shards, and
   supervised best checkpoints.

Commands that require missing raw artifacts will be labelled as conditional
rebuild commands rather than currently runnable reproduction commands. The report
will not claim that the current checkout can regenerate the experiments.

## Versioned Delivery Layout

Add a self-contained report delivery under:

```text
docs/final-report/
  README.md
  report.tex
  report.pdf
  figures/
  evidence/
```

The directory will contain only the figures actually referenced by the report
and the three currently retained small evidence files. `README.md` will document
the compile command, source-of-truth JSON locations, verification scope, and
missing raw artifacts. Existing untracked files under the repository-level
`result/` directory will not be added, deleted, or modified.

The working formal copy at `/home/jichi/zuoye/result/` will remain synchronized
with the versioned source and PDF.

## Verification

Before handoff:

1. recompute the high-impact retained metrics from the archived JSON summaries;
2. verify that no 11% multi-start improvement claim remains;
3. verify that no direct surrogate-error claim is made without a direct metric;
4. confirm every listed present/missing artifact against the checkout;
5. compile the versioned LaTeX source with Tectonic;
6. inspect the rendered pages containing the abstract, deep experiment,
   discussion, conclusion, and archive appendix;
7. run the full test suite and Ruff; and
8. commit only the design and final-report delivery files, leaving unrelated
   untracked outputs untouched.

## Non-goals

- No new inversion, training, or dataset-generation run.
- No reconstructed or estimated same-sample multi-start effect.
- No fabricated surrogate-versus-physical discrepancy metric.
- No changes to archived JSON, manifests, model checkpoints, or experiment code.
- No attempt to commit missing large raw artifacts that are not available in the
  current workspace.
