# Formal Report High-Priority Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove two unsupported scientific inferences and deliver an archive-honest, version-controlled formal report.

**Architecture:** Treat the archived JSON summaries as immutable evidence and revise only the reader-facing interpretation. Build a self-contained `docs/final-report/` delivery from the corrected LaTeX source, selected figures, small evidence files, and an explicit archive-boundary README; keep the external working copy synchronized.

**Tech Stack:** LaTeX/ctex, Tectonic 0.16.9, JSON, Git, Pytest, Ruff, pypdfium2 for PDF inspection.

## Global Constraints

- Do not change archived JSON, manifests, checkpoints, experiment code, or numerical results.
- Do not estimate a same-sample multi-start effect from different populations.
- Do not infer direct surrogate error from two observation-residual metrics.
- Do not claim raw reproducibility when required raw artifacts are absent.
- Leave the existing untracked repository-level `result/inversion-report/` files untouched.
- Commit only the design, plan, and `docs/final-report/` delivery files.

---

### Task 1: Correct the scientific interpretation

**Files:**
- Modify: `/home/jichi/zuoye/result/report.tex:30-56`
- Modify: `/home/jichi/zuoye/result/report.tex:386-404`
- Modify: `/home/jichi/zuoye/result/report.tex:725-751`
- Modify: `/home/jichi/zuoye/result/report.tex:782-800`

**Interfaces:**
- Consumes: immutable metrics from `results/inversion-report/summary.json`
- Produces: scientifically conservative report text with unchanged numbers

- [ ] **Step 1: Record the invalid claims before editing**

Run:

```bash
rg -n '改善约 11|外推误差已超过|代理精度成为|多起点实验仅将' /home/jichi/zuoye/result/report.tex
```

Expected: matches in the abstract, deep-result section, and discussion.

- [ ] **Step 2: Recompute the population and residual facts**

Run:

```bash
python3 - <<'PY'
import json
s = json.load(open('results/inversion-report/summary.json'))['experiment_scopes']
for scope in ('full', 'deep'):
    overall = s[scope]['groups']['overall']
    print(scope, overall['sample_count'])
    print({k: v['vs']['row_count'] for k, v in s[scope]['groups']['model_kind'].items()})
deep = s['deep']['groups']['overall']
print(deep['surrogate_frequency']['overall']['mae_km_s'])
print(deep['physical_frequency']['overall']['mae_km_s'])
print(deep['physical_frequency']['overall']['missing_fraction'])
PY
```

Expected: full has 200,000 rows in the natural family mix; deep has 800 rows with 200 per family; residuals are 0.0155229 and 0.0115596 km/s; the physical comparison has nonzero missingness.

- [ ] **Step 3: Replace the multi-start treatment-effect language**

Keep the absolute deep metrics and state that the equal-family 400-model subset cannot be compared causally with the naturally weighted 100,000-model full population. State that a single-start evaluation on the same 400 IDs is required to quantify a multi-start effect.

- [ ] **Step 4: Replace the direct surrogate-error inference**

Keep both observation residuals, state that they show evaluation-operator dependence, and explicitly identify aligned-cell `MAE(F_surrogate(v_hat), F_physical(v_hat))` as the missing metric.

- [ ] **Step 5: Verify the invalid claims are absent**

Run:

```bash
rg -n '改善约 11|外推误差已超过|代理精度成为' /home/jichi/zuoye/result/report.tex
```

Expected: no matches.

### Task 2: Make the archive boundary explicit

**Files:**
- Modify: `/home/jichi/zuoye/result/report.tex:1-5`
- Modify: `/home/jichi/zuoye/result/report.tex:803-850`
- Create: `docs/final-report/README.md`

**Interfaces:**
- Consumes: checkout artifact inventory and report source paths
- Produces: accurate present/missing artifact inventory and conditional rebuild instructions

- [ ] **Step 1: Inventory every report dependency**

Run:

```bash
for path in data/production runs/production-48g runs/supervised-inversion-v2 \
  results/inversion results/inversion-report results/hybrid-inversion \
  results/hybrid-inversion-report; do
  test -e "$path" && echo "PRESENT $path" || echo "MISSING $path"
done
git ls-files data runs results
find /home/jichi/zuoye/result/evidence -maxdepth 1 -type f -printf '%f\n' | sort
```

- [ ] **Step 2: Rewrite the appendix into three evidence levels**

List versioned inspectable artifacts, metrics verifiable from retained summaries, and raw experiments not reproducible from the checkout. Explicitly include all missing production data, forward checkpoints, legacy inversion HDF5, hybrid HDF5, and supervised best checkpoints.

- [ ] **Step 3: Label rebuild commands as conditional**

Keep the commands only with a prerequisite statement requiring restored raw artifacts. Preserve `pytest` and Ruff as currently runnable code-quality checks.

- [ ] **Step 4: Create the delivery README**

Document:

```text
compile: cd docs/final-report && /tmp/tectonic-0.16.9/tectonic -X compile report.tex
source summaries: ../../results/inversion-report/summary.json and ../../results/hybrid-inversion-report/summary.json
current scope: summary-level verification, not raw experiment reproduction
```

- [ ] **Step 5: Check the inventory text against the filesystem**

Run:

```bash
rg -n 'data/production|runs/production-48g|results/inversion|hybrid.*HDF5|seed-.*best.pt' \
  /home/jichi/zuoye/result/report.tex docs/final-report/README.md
```

Expected: every path described as present is tracked or copied into `docs/final-report/`; every absent raw artifact is explicitly named.

### Task 3: Build the versioned report delivery

**Files:**
- Create: `docs/final-report/report.tex`
- Create: `docs/final-report/report.pdf`
- Create: `docs/final-report/figures/*.png`
- Create: `docs/final-report/evidence/dataset-audit.json`
- Create: `docs/final-report/evidence/forward-test-evaluation.json`
- Create: `docs/final-report/evidence/plot-supervised-comparison.py`

**Interfaces:**
- Consumes: corrected external working report and retained selected assets
- Produces: self-contained, version-controlled report delivery

- [ ] **Step 1: Copy only the nine referenced figures**

Run:

```bash
rg -o '\\includegraphics\[[^]]*\]\{figures/[^}]+\}' /home/jichi/zuoye/result/report.tex
```

Copy exactly the eight resulting PNG names. Do not copy unreferenced result figures.

- [ ] **Step 2: Copy the three small evidence files**

Copy only:

```text
/home/jichi/zuoye/result/evidence/dataset-audit.json
/home/jichi/zuoye/result/evidence/forward-test-evaluation.json
/home/jichi/zuoye/result/evidence/plot-supervised-comparison.py
```

- [ ] **Step 3: Place the corrected LaTeX source in the delivery directory**

Update report-local evidence paths to `evidence/...` and preserve repository source-summary paths in the appendix as provenance.

- [ ] **Step 4: Confirm self-contained report inputs**

Run:

```bash
python3 - <<'PY'
import pathlib, re
root = pathlib.Path('docs/final-report')
text = (root / 'report.tex').read_text()
paths = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text)
missing = [path for path in paths if not (root / path).is_file()]
assert not missing, missing
print(f'{len(paths)} referenced graphics present')
PY
```

### Task 4: Compile and inspect the final PDF

**Files:**
- Generate: `docs/final-report/report.pdf`
- Synchronize: `/home/jichi/zuoye/result/report.tex`
- Synchronize: `/home/jichi/zuoye/result/report.pdf`

**Interfaces:**
- Consumes: `docs/final-report/report.tex`, figures, and evidence
- Produces: validated PDF in both versioned and working locations

- [ ] **Step 1: Compile with Tectonic**

Run:

```bash
cd docs/final-report
/tmp/tectonic-0.16.9/tectonic -X compile report.tex --keep-logs --keep-intermediates
```

Expected: exit code 0; only known underfull-box warnings; no undefined references, missing figures, or LaTeX errors.

- [ ] **Step 2: Render and inspect affected pages**

Run pypdfium2 from `/tmp/pdf-render` to render all PDF pages to `/tmp/final-report-audit-page-N.png`, then assemble a contact sheet with Pillow. Verify readable tables, no clipping, and adjacent caveats.

- [ ] **Step 3: Synchronize the working formal copy**

Copy the validated `report.tex`, `report.pdf`, selected figures, and evidence back to `/home/jichi/zuoye/result/` without modifying unrelated files.

- [ ] **Step 4: Verify source/PDF identity and freshness**

Record SHA-256 values and confirm the versioned and working copies are byte-identical.

### Task 5: Final verification and commit

**Files:**
- Verify: `docs/final-report/**`
- Verify: repository tracked source and tests

**Interfaces:**
- Consumes: completed versioned report delivery
- Produces: committed, reviewable report fix

- [ ] **Step 1: Re-run numeric spot checks**

Run read-only Python assertions against:

```text
results/inversion-report/summary.json
runs/supervised-inversion-v2/evaluation.json
results/hybrid-inversion/tuning.json
results/hybrid-inversion-report/summary.json
```

Assert the rounded values printed in the report: full MAE 0.3047, deep MAE 0.2713, supervised test/inversion MAE 0.02629/0.02633, hybrid clean/noise MAE 0.02739/0.05491, and control reductions 89.34%/78.64%.

- [ ] **Step 2: Run report integrity checks**

Check missing graphics, trailing whitespace, undefined references, forbidden invalid-claim phrases, and the present/missing artifact inventory.

- [ ] **Step 3: Run the full code verification**

Run:

```bash
/home/jichi/zuoye/.venv/bin/python -m pytest -q
/home/jichi/zuoye/.venv/bin/ruff check src tests scripts
git diff --check
```

Expected: 307 tests pass, Ruff passes, and no whitespace errors.

- [ ] **Step 4: Review the exact Git payload**

Expected tracked additions: the implementation plan and `docs/final-report/**`. Expected unrelated state: repository-level `result/` remains untracked and untouched.

- [ ] **Step 5: Commit and push**

```bash
git add docs/superpowers/plans/2026-08-17-report-high-priority-fixes.md docs/final-report
git commit -m "docs: fix formal report validity and archive scope"
git push
```
