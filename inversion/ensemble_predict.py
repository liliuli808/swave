#!/usr/bin/env python3
"""多种子集成预测：对全部可用 inverse_net_seed*_best.pt 做集成。

- val 折上对比 单种子(seed0) vs 集成 的 MAE（决定是否采用集成）；
- 测试折集成预测 -> results/predictions_invnet.npy（物理 km/s）；
- 种子间标准差 -> results/invnet_std.npy（不确定性诊断，仅种子方差）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SWAVE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SWAVE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inverse_net import load_inverse_net, predict_physical  # noqa: E402

RESULTS_DIR = SWAVE_ROOT / "inversion/results"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = sorted(RESULTS_DIR.glob("inverse_net_seed*_best.pt"))
    print(f"集成种子: {[c.stem for c in ckpts]}")

    z = np.load(RESULTS_DIR / "fold_data_cache.npz")
    X, vs, fold = z["X"], z["vs"], z["fold"]
    val_sel = (fold >= 90) & (fold < 95)
    test_sel = fold >= 95

    preds_val, preds_test = [], []
    for ckpt in ckpts:
        model, norm = load_inverse_net(ckpt, device=device)
        preds_val.append(predict_physical(model, norm, X[val_sel],
                                          device=device))
        preds_test.append(predict_physical(model, norm, X[test_sel],
                                           device=device))

    # val: 单种子 vs 集成
    mae_single = np.abs(preds_val[0] - vs[val_sel]).mean() * 1000
    mae_ens = np.abs(np.mean(preds_val, axis=0) - vs[val_sel]).mean() * 1000
    print(f"val MAE: 单种子 {mae_single:.2f} -> {len(ckpts)} 种子集成 "
          f"{mae_ens:.2f} m/s")

    ens_test = np.mean(preds_test, axis=0)
    std_test = np.std(preds_test, axis=0)
    mae_test = np.abs(ens_test - vs[test_sel]).mean() * 1000
    print(f"test MAE（集成）: {mae_test:.2f} m/s "
          f"(仅汇报，选择依据是 val)")

    np.save(RESULTS_DIR / "predictions_invnet.npy", ens_test)
    np.save(RESULTS_DIR / "invnet_std.npy", std_test)
    print(f"saved predictions_invnet.npy {ens_test.shape}, invnet_std.npy")


if __name__ == "__main__":
    main()
