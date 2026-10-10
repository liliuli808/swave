#!/usr/bin/env bash
# Controlled training-row experiment; evaluates validation only during training.
set -euo pipefail
cd "$(dirname "$0")/.."
SWAVE_REPO_ROOT=$(pwd -P)
export PYTHONPATH="$SWAVE_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
if [ "$DEVICE" = cuda ]; then
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
fi
"$PYTHON" -m pytest -q tests/test_kernel_mining.py tests/test_kernels.py \
  tests/test_kernel_training.py

exec "$PYTHON" scripts/finetune_forward_kernels.py \
  --base-checkpoint "${BASE_CHECKPOINT:-runs/kernel-wide-kw3/best.pt}" \
  --dataset-dir "${DATASET_DIR:-data/production}" \
  --kernel-dir "${KERNEL_DIR:-data/kernels}" --cache-dir "${CACHE_DIR:-data/cache}" \
  --corrections "${CORRECTIONS:-results/kissing-repair/corrections.npz}" \
  --output-dir "${OUTPUT_DIR:-runs/kernel-corrected-rows-v4}" \
  --device "$DEVICE" --threads "${THREADS:-$(nproc)}" --seed 20261007 \
  --width 512 --blocks 6 --require-warm-start --max-initial-score 0.05 \
  --epochs "${EPOCHS:-50}" --batch-size 4096 --kernel-batch-size 2048 \
  --learning-rate 2e-5 --final-learning-rate 1e-6 \
  --warmup-steps 300 --max-grad-norm 1 \
  --kernel-weight 1 --kernel-directions 2 \
  --hard-example-power 0 --kernel-hard-example-power 0 \
  --kernel-row-weight "${KERNEL_ROW_WEIGHT:-0.1}" \
  --kernel-row-batch-size "${KERNEL_ROW_BATCH_SIZE:-256}" \
  --kernel-mining-samples "${KERNEL_MINING_SAMPLES:-8192}" \
  --kernel-mining-interval "${KERNEL_MINING_INTERVAL:-5}"
