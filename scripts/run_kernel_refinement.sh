#!/usr/bin/env bash
# One-command pipeline: physical kernel labels -> Sobolev fine-tuning -> evaluation.
# See docs/runbooks/kernel-refinement.md. Every stage is resumable; rerunning the
# script skips finished work. Override any variable from the environment, e.g.
#   DEVICE=cpu EPOCHS=30 BATCH_SIZE=1024 KERNEL_BATCH_SIZE=512 \
#     bash scripts/run_kernel_refinement.sh
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET_DIR=${DATASET_DIR:-data/production}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-runs/production-48g/best.pt}
KERNEL_DIR=${KERNEL_DIR:-data/kernels}
CACHE_DIR=${CACHE_DIR:-data/cache}
CORRECTIONS=${CORRECTIONS:-results/kissing-repair/corrections.npz}
OUTPUT_DIR=${OUTPUT_DIR:-runs/kernel-finetune}
RESULTS_DIR=${RESULTS_DIR:-results/forward-kernel}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-4096}
KERNEL_BATCH_SIZE=${KERNEL_BATCH_SIZE:-2048}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
WARMUP_STEPS=${WARMUP_STEPS:-300}
KERNEL_WEIGHT=${KERNEL_WEIGHT:-1.0}
# >0 resamples kernel-labelled models in proportion to their worst kernel row.
KERNEL_HARD_POWER=${KERNEL_HARD_POWER:-0}
# Random Jacobian-vector directions per kernel sample; 0 uses the exact full
# Jacobian (one JVP per layer, ~10x the kernel cost of the default 2).
KERNEL_DIRECTIONS=${KERNEL_DIRECTIONS:-2}
# 1 derives anomaly-localizing profile features inside the network (fresh
# training; cannot fine-tune the 256x4 base checkpoint with this on).
PROFILE_FEATURES=${PROFILE_FEATURES:-0}
# 256/4 fine-tunes the base checkpoint; any other size trains from scratch.
WIDTH=${WIDTH:-256}
BLOCKS=${BLOCKS:-4}
THREADS=${THREADS:-$(nproc)}
PYTHON=${PYTHON:-python}
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-$THREADS}

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { log "ERROR: $*"; exit 1; }

log "stage 0: preflight"
[ -d "$DATASET_DIR" ] || fail "dataset directory $DATASET_DIR not found"
shards=$(find "$DATASET_DIR" -maxdepth 1 -name 'shard-*.h5' | wc -l)
[ "$shards" -eq 100 ] || fail "expected 100 shards in $DATASET_DIR, found $shards"
[ -f "$BASE_CHECKPOINT" ] || fail "base checkpoint $BASE_CHECKPOINT not found"
[ -f "$CORRECTIONS" ] || fail "corrections file $CORRECTIONS not found"
"$PYTHON" -c "import swave" 2>/dev/null || fail "swave is not installed: run python -m pip install -e '.[dev]'"
if [ "$DEVICE" = cuda ]; then
  "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || fail "DEVICE=cuda but torch.cuda.is_available() is False (set DEVICE=cpu to run on CPU)"
fi
"$PYTHON" -m pytest -q tests/test_kernels.py tests/test_solver.py \
  || fail "kernel/solver tests failed"

log "stage 1: physical sensitivity-kernel labels -> $KERNEL_DIR"
"$PYTHON" scripts/build_kernel_dataset.py "$DATASET_DIR" "$KERNEL_DIR" --split validation --stride 5
"$PYTHON" scripts/build_kernel_dataset.py "$DATASET_DIR" "$KERNEL_DIR" --split test --stride 5
"$PYTHON" scripts/build_kernel_dataset.py "$DATASET_DIR" "$KERNEL_DIR" --split train --stride 4
for split in validation test train; do
  count=$(find "$KERNEL_DIR/$split" -maxdepth 1 -name 'kernels-*.h5' | wc -l)
  [ "$count" -eq 100 ] || fail "expected 100 kernel files for $split, found $count"
done

log "stage 2: Sobolev fine-tuning on $DEVICE -> $OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")"
extra_args=()
if [ "${PROFILE_FEATURES:-0}" = "1" ]; then
  extra_args+=(--profile-features)
fi
"$PYTHON" scripts/finetune_forward_kernels.py \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --dataset-dir "$DATASET_DIR" --kernel-dir "$KERNEL_DIR" \
  --cache-dir "$CACHE_DIR" --corrections "$CORRECTIONS" \
  --output-dir "$OUTPUT_DIR" --device "$DEVICE" --threads "$THREADS" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --kernel-batch-size "$KERNEL_BATCH_SIZE" --learning-rate "$LEARNING_RATE" \
  --warmup-steps "$WARMUP_STEPS" --kernel-weight "$KERNEL_WEIGHT" \
  --width "$WIDTH" --blocks "$BLOCKS" \
  --kernel-hard-example-power "$KERNEL_HARD_POWER" \
  --kernel-directions "$KERNEL_DIRECTIONS" \
  "${extra_args[@]}" \
  2>&1 | tee -a "$OUTPUT_DIR.log"
[ -f "$OUTPUT_DIR/best.pt" ] || [ -f "$OUTPUT_DIR/last.pt" ] || fail "fine-tuning produced no checkpoint"
final="$OUTPUT_DIR/best.pt"
[ -f "$final" ] || final="$OUTPUT_DIR/last.pt"

log "stage 3: evaluation and figures -> $RESULTS_DIR"
"$PYTHON" scripts/evaluate_forward_kernels.py \
  --checkpoint base="$BASE_CHECKPOINT" \
  --checkpoint finetuned="$final" \
  --dataset-dir "$DATASET_DIR" --kernel-dir "$KERNEL_DIR" \
  --cache-dir "$CACHE_DIR" --corrections "$CORRECTIONS" \
  --output-dir "$RESULTS_DIR" --threads "$THREADS"
cp "$OUTPUT_DIR/history.json" "$RESULTS_DIR/finetune-history.json"

log "stage 4: acceptance check"
"$PYTHON" - "$RESULTS_DIR/summary.json" <<'EOF'
import json, sys
summary = json.load(open(sys.argv[1]))
rows = []
for name, item in summary.items():
    value, kernel = item["value"], item["kernel"]
    rows.append((name, value["samples_all_within_1pct"], value["points_within_1pct"],
                 value["max_relative_error"], kernel["rows_within_5pct"],
                 kernel["median_relative_l2"]))
print(f"{'model':<10} {'curves<=1%':>11} {'points<=1%':>11} {'max rel':>8} {'kernel<=5%':>11} {'kernel med':>11}")
for name, a, b, c, d, e in rows:
    print(f"{name:<10} {a:11.4%} {b:11.4%} {c:8.3%} {d:11.4%} {e:11.3%}")
final = summary["finetuned"]
passed = (final["value"]["samples_all_within_1pct"] >= 0.99
          and final["kernel"]["rows_within_5pct"] >= 0.99)
print("ACCEPTANCE:", "PASS" if passed else "NOT YET (see runbook: tuning)")
EOF
log "done. Report back: $OUTPUT_DIR.log and $RESULTS_DIR/summary.json"
