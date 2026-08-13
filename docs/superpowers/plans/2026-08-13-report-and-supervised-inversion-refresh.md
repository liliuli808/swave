# Report and Supervised Inversion Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the production dataset, prepare a leakage-safe three-seed supervised inversion workflow for the 80/5/5/10 protocol, and revise the technical report using only current-protocol evidence.

**Architecture:** Keep deterministic splitting in `swave.splits`, add a streaming audit module and a separate supervised-inversion module with explicit checkpoint identities, and expose both through focused CLI commands. Treat `/home/jichi/swave/swave` as immutable evidence, archive derived JSON under `/home/jichi/zuoye/result/evidence`, and leave supervised numerical tables explicitly unpopulated until the GPU run produces `evaluation.json`.

**Tech Stack:** Python 3.12, NumPy, HDF5/h5py, PyTorch, pytest, Ruff, XeTeX/tectonic-compatible LaTeX.

## Global Constraints

- The only split policy is `mod100-v2-80-5-5-10`: train 0--79, validation 80--84, test 85--89, inversion 90--99.
- Test and inversion rows must never affect training statistics, checkpoint selection, or hyperparameters; inversion rows may be read only after all checkpoints are fixed for the requested same-sample comparison with Section 5.
- Formal supervised training uses seeds `0`, `1`, and `2`; the final prediction is their equal-weight ensemble.
- `/home/jichi/swave/swave` is read-only evidence and must not be changed.
- Old 90/5/5 and all iNETT evidence must be absent from the revised report.
- Missing GPU results must be labeled as not yet run; no values may be inferred from old checkpoints.

---

### Task 1: Streaming Production Dataset Audit

**Files:**
- Create: `src/swave/dataset_audit.py`
- Create: `tests/test_dataset_audit.py`
- Modify: `src/swave/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `DatasetConfig`, `validate_dataset_files()`, `generate_model()`, `mask_for_split()`.
- Produces: `audit_dataset(dataset_dir: Path, dataset_config: DatasetConfig) -> dict[str, object]` and `swave audit-dataset --dataset-dir ... --dataset-config ... --output ...`.

- [ ] **Step 1: Write failing identity, duplicate, and geology-rule tests**

```python
def test_audit_reports_exact_duplicates_and_cross_split_leakage(tmp_path):
    dataset, config = write_audit_fixture(tmp_path, duplicate_ids=(0, 85))
    report = audit_dataset(dataset, config, validate_checksums=False)
    assert report["duplicates"]["vs_duplicate_rows"] == 2
    assert report["duplicates"]["vs_cross_split_groups"] == 1


def test_audit_accepts_reconstructed_four_family_models(tmp_path):
    dataset, config = write_generated_family_fixture(tmp_path)
    report = audit_dataset(dataset, config, validate_checksums=False)
    assert report["geology"]["violations"] == 0
    assert set(report["geology"]["by_kind"]) == {
        "NORMAL", "LOW_VELOCITY", "HIGH_VELOCITY", "COUPLED"
    }
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `../.venv/bin/pytest tests/test_dataset_audit.py -q`  
Expected: collection fails because `swave.dataset_audit` does not exist.

- [ ] **Step 3: Implement streaming digest and geology checks**

```python
def audit_dataset(dataset_dir: Path, dataset_config: DatasetConfig, *,
                  validate_checksums: bool = True) -> dict[str, object]:
    manifest = (validate_dataset_files(dataset_dir) if validate_checksums
                else load_manifest(dataset_dir / "manifest.json"))
    # Read one shard at a time, retain only IDs and 32-byte row digests.
    # Reconstruct each stored model from seed, ID, and retry count and compare
    # float32 Vs/Vp/density and the declared family rule.
    return {
        "schema_version": 1,
        "dataset_manifest_sha256": dataset_manifest_sha256(manifest),
        "identity": identity_summary,
        "duplicates": duplicate_summary,
        "geology": geology_summary,
    }
```

Use SHA-256 row digests, sort the fixed-width digest arrays, and re-read only candidate groups to distinguish a true byte-for-byte duplicate from a digest collision. Record at most ten example ID groups while preserving full counts.

- [ ] **Step 4: Add the CLI command and JSON output path**

```python
audit = subparsers.add_parser("audit-dataset")
audit.add_argument("--dataset-config", default="configs/dataset.toml")
audit.add_argument("--dataset-dir", required=True)
audit.add_argument("--output", required=True)
audit.set_defaults(handler=_audit_dataset)
```

- [ ] **Step 5: Run focused tests**

Run: `../.venv/bin/pytest tests/test_dataset_audit.py tests/test_cli.py -q`  
Expected: all focused tests pass.

- [ ] **Step 6: Commit the audit implementation**

```bash
git add src/swave/dataset_audit.py src/swave/cli.py \
  tests/test_dataset_audit.py tests/test_cli.py
git commit -m "feat: audit production dataset geology and duplicates"
```

### Task 2: Supervised Inversion Data and Network

**Files:**
- Create: `src/swave/supervised_inversion.py`
- Create: `tests/test_supervised_inversion.py`

**Interfaces:**
- Consumes: `SPLIT_POLICY`, `mask_for_split()`, `validate_dataset_files()`, `dataset_manifest_sha256()`.
- Produces: `SupervisedConfig`, `SupervisedNormalization`, `InverseNet`, `SupervisedHDF5Dataset`, `compute_supervised_normalization()`.

- [ ] **Step 1: Write failing split-isolation and normalization tests**

```python
def test_supervised_rows_obey_four_way_policy(four_split_dataset):
    assert len(SupervisedHDF5Dataset(four_split_dataset, "train")) == 80
    assert len(SupervisedHDF5Dataset(four_split_dataset, "validation")) == 5
    assert len(SupervisedHDF5Dataset(four_split_dataset, "test")) == 5
    assert len(SupervisedHDF5Dataset(four_split_dataset, "inversion")) == 10


def test_normalization_uses_train_rows_only(four_split_dataset):
    stats = compute_supervised_normalization(four_split_dataset)
    assert stats.target_mean[0] == pytest.approx(expected_train_mean)
    assert stats.fill_values[1, 0] == pytest.approx(expected_train_fill)
```

- [ ] **Step 2: Run tests and confirm the missing-module failure**

Run: `../.venv/bin/pytest tests/test_supervised_inversion.py -q`  
Expected: collection fails because `swave.supervised_inversion` does not exist.

- [ ] **Step 3: Implement validated configuration and Pre-LN network**

```python
@dataclass(frozen=True)
class SupervisedConfig:
    dataset_dir: Path = Path("data/production")
    output_dir: Path = Path("runs/supervised-inversion-v2")
    seeds: tuple[int, ...] = (0, 1, 2)
    width: int = 1024
    blocks: int = 4
    batch_size: int = 8192
    epochs: int = 150
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 16
    device: str = "cuda"
    resume: bool = True
```

`InverseNet.forward()` maps `(batch, 476)` normalized dispersion values to `(batch, 20)` normalized Vs values using a GELU stem, four Pre-LN residual blocks, final LayerNorm, and linear head.

- [ ] **Step 4: Implement train-only streaming statistics and row loading**

Read HDF5 in shard batches. Drop only frequency column zero, calculate fill, input mean/std, and target mean/std from `mask_for_split(ids, "train")`, and make `SupervisedHDF5Dataset` fill and normalize each row without materializing the million-row dataset on GPU.

- [ ] **Step 5: Run focused tests**

Run: `../.venv/bin/pytest tests/test_supervised_inversion.py -q`  
Expected: split, fill, normalization, and network-shape tests pass.

- [ ] **Step 6: Commit the data and network layer**

```bash
git add src/swave/supervised_inversion.py tests/test_supervised_inversion.py
git commit -m "feat: add leakage-safe supervised inversion data path"
```

### Task 3: Three-Seed Training, Resume, Ensemble Evaluation, and CLI

**Files:**
- Modify: `src/swave/supervised_inversion.py`
- Modify: `tests/test_supervised_inversion.py`
- Modify: `src/swave/cli.py`
- Modify: `tests/test_cli.py`
- Create: `configs/supervised-inversion-48g.toml`
- Modify: `README.md`

**Interfaces:**
- Produces: `train_supervised(config: SupervisedConfig) -> Path`, `evaluate_supervised_ensemble(...) -> dict[str, object]`, and `swave train-inverse --config ...`.

- [ ] **Step 1: Write failing checkpoint identity, validation selection, resume, and ensemble tests**

```python
def test_checkpoint_binds_current_dataset_and_split(tiny_supervised_run):
    payload = torch.load(tiny_supervised_run / "seed-0-best.pt", weights_only=False)
    assert payload["split_policy"] == SPLIT_POLICY
    assert payload["dataset_config_hash"] == "tiny-complete-fixture"
    assert len(payload["dataset_manifest_sha256"]) == 64


def test_ensemble_evaluation_reads_holdouts_only_after_training(tiny_supervised_run):
    report = json.loads((tiny_supervised_run / "evaluation.json").read_text())
    assert report["test"]["sample_count"] == 5
    assert report["inversion_comparison"]["sample_count"] == 10
```

- [ ] **Step 2: Run tests and confirm failures at the unimplemented training interfaces**

Run: `../.venv/bin/pytest tests/test_supervised_inversion.py tests/test_cli.py -q`  
Expected: new training and evaluation tests fail while Task 2 tests stay green.

- [ ] **Step 3: Implement atomic last/best checkpoints and JSON histories**

Each seed uses `seed-{seed}-last.pt`, `seed-{seed}-best.pt`, and `seed-{seed}-history.json`. Checkpoints include model and optimizer state, epoch, best validation physical MAE, normalization arrays, model/training config, split policy, dataset identities, and ordered train-ID SHA-256. Resume rejects any mismatch.

- [ ] **Step 4: Implement validation-only selection and final ensemble evaluation**

At every epoch, select by physical validation MAE. After all three best checkpoints exist, evaluate their individual and equal-weight ensemble validation predictions, then read the test split once for final metrics. Finally, read the inversion holdout once for a same-sample comparison with Section 5; this comparison cannot influence any checkpoint or hyperparameter. Write `evaluation.json` atomically with overall, family, and per-layer metrics in km/s plus sample counts and identities.

- [ ] **Step 5: Add config, CLI, and production command documentation**

```toml
[supervised]
dataset_dir = "data/production"
output_dir = "runs/supervised-inversion-v2"
seeds = [0, 1, 2]
width = 1024
blocks = 4
batch_size = 8192
epochs = 150
patience = 15
learning_rate = 0.001
weight_decay = 0.0001
num_workers = 16
device = "cuda"
resume = true
```

- [ ] **Step 6: Run focused tests and lint**

Run: `../.venv/bin/pytest tests/test_supervised_inversion.py tests/test_cli.py -q`  
Run: `../.venv/bin/ruff check src/swave/supervised_inversion.py src/swave/cli.py tests/test_supervised_inversion.py tests/test_cli.py`  
Expected: tests pass and Ruff reports no errors.

- [ ] **Step 7: Commit the training workflow**

```bash
git add src/swave/supervised_inversion.py src/swave/cli.py \
  tests/test_supervised_inversion.py tests/test_cli.py \
  configs/supervised-inversion-48g.toml README.md
git commit -m "feat: train supervised inversion on isolated four-way split"
```

### Task 4: Produce CPU Evidence from the Formal Artifacts

**Files:**
- Create: `/home/jichi/zuoye/result/evidence/dataset-audit.json`
- Create: `/home/jichi/zuoye/result/evidence/forward-test-evaluation.json`

**Interfaces:**
- Consumes: `/home/jichi/swave/swave/data/production`, `/home/jichi/swave/swave/runs/production-48g/best.pt`.
- Produces: immutable JSON evidence consumed manually by the report.

- [ ] **Step 1: Run the production dataset audit**

Run:

```bash
NUMBA_CACHE_DIR=/tmp/swave-numba-cache PYTHONPATH=src ../.venv/bin/swave \
  audit-dataset \
  --dataset-config configs/dataset.toml \
  --dataset-dir /home/jichi/swave/swave/data/production \
  --output /home/jichi/zuoye/result/evidence/dataset-audit.json
```

Expected: one million rows, zero ID duplicates, zero exact Vs/full-record duplicates, zero cross-split duplicates, and zero geology violations. If the result differs, report the measured finding and do not alter the audit to force the expected result.

- [ ] **Step 2: Run current-checkpoint test evaluation**

Run a small Python wrapper around `swave.training.evaluate()` and write its returned object as strict JSON to `forward-test-evaluation.json` using atomic replacement. Expected mode counts are approximately six million cells each and every metric is finite.

- [ ] **Step 3: Independently inspect evidence consistency**

Check the audit manifest/config hashes against the checkpoint and inversion summary. Confirm the test split count is 50,000 and the inversion summary remains bound to the same current checkpoint.

### Task 5: Rewrite and Compile the Technical Report

**Files:**
- Modify: `/home/jichi/zuoye/result/report.tex`
- Modify: `/home/jichi/zuoye/result/report.pdf`
- Delete: `/home/jichi/zuoye/result/figures/inett-vs-comparison-all.png`

**Interfaces:**
- Consumes: the two Task 4 evidence JSON files, current inversion `summary.json`, current training `history.json`.
- Produces: a self-contained intermediate report that contains no old-split or iNETT claims.

- [ ] **Step 1: Add regression scans before editing**

Record the current matches for `90/5/5`, `90--94`, `>=95`, `old-split`, `iNETT`, `inett`, and the old supervised values `26.87`, `26.08`, and `26.9`. The post-edit scan must return no matches.

- [ ] **Step 2: Replace title metadata and move citations**

Set an empty author, replace introductory footnotes with `\cite{pan-mode-kissing,pan-qedispinv}`, and add a `thebibliography` section immediately before the appendix with both papers and GitHub repositories.

- [ ] **Step 3: Replace data and forward-evaluation claims with measured evidence**

Use `dataset-audit.json` for family counts, anomaly rules/ranges, exact duplicate counts, and the explicit synthetic-geology limitation. Use `forward-test-evaluation.json` for the four-mode 85--89 test table and remove the old table and limitation paragraph.

- [ ] **Step 4: Remove iNETT and old supervised claims, retain only the new method and pending status**

Delete the external-reference subsection, iNETT figure and section text, and every abstract/discussion/conclusion statement that compares against old supervised results. Rewrite Section 6 to document the new split-safe three-seed method and state that its numerical table will be inserted only from `runs/supervised-inversion-v2/evaluation.json` after the GPU run.

- [ ] **Step 5: Build the PDF and inspect compilation diagnostics**

Run XeTeX/tectonic twice as required for references and table of contents. Reject undefined references, missing figures, overfull content that clips, and compilation errors. Confirm `report.pdf` is newer than `report.tex`.

### Task 6: Final Verification and GPU Handoff

**Files:**
- Modify if needed: touched code/tests/docs only.

- [ ] **Step 1: Run the complete repository suite**

Run: `../.venv/bin/pytest -q`  
Expected: all tests pass.

- [ ] **Step 2: Run static and diff checks**

Run: `../.venv/bin/ruff check src tests scripts`  
Run: `git diff --check`  
Expected: both complete without diagnostics.

- [ ] **Step 3: Scan the final report source**

Run:

```bash
rg -n '90/5/5|90--94|>=95|old-split|iNETT|inett|26\.87|26\.08|26\.9' \
  /home/jichi/zuoye/result/report.tex
```

Expected: no output. Also confirm the new split policy, audit evidence, and 85--89 test table are present.

- [ ] **Step 4: Deliver the resumable GPU command**

```bash
NUMBA_CACHE_DIR=/tmp/swave-numba-cache swave train-inverse \
  --config configs/supervised-inversion-48g.toml \
  --dataset-dir data/production \
  --output-dir runs/supervised-inversion-v2 \
  --device cuda
```

Expected outputs: three best/last checkpoints, three histories, and one strict `evaluation.json`. Re-running the same command resumes compatible incomplete seeds and validates completed outputs.
