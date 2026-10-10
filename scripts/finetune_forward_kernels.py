"""Fine-tune the forward surrogate on values plus physical sensitivity kernels.

    python scripts/finetune_forward_kernels.py \
        --base-checkpoint runs/production-48g/best.pt \
        --dataset-dir data/production --kernel-dir data/kernels \
        --output-dir runs/kernel-finetune
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

from swave.kernel_training import KernelTrainingConfig, train_with_kernels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for item in fields(KernelTrainingConfig):
        name = "--" + item.name.replace("_", "-")
        if item.type in {"Path", Path}:
            parser.add_argument(name, type=Path, required=True)
        elif item.name == "require_warm_start":
            parser.add_argument(
                name, action="store_true",
                help="Require a fresh fine-tune from matching base weights; "
                     "reject random initialization and automatic resume.",
            )
        elif item.type in {bool, "bool"}:
            parser.add_argument(name, action="store_true")
        elif item.name == "corrections":
            parser.add_argument(name, type=Path, default=None)
        elif item.name == "steps_per_epoch":
            parser.add_argument(name, type=int, default=None)
        elif item.name == "kernel_train_split":
            parser.add_argument(name, default="train")
        else:
            caster = {"int": int, "float": float}.get(str(item.type), str)
            parser.add_argument(name, type=caster, default=item.default)
    args = parser.parse_args()
    config = KernelTrainingConfig(**vars(args))
    print(train_with_kernels(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
