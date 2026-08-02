#!/usr/bin/env python3
"""诊断2（多进程版）: 独立物理引擎验证 InvNet 预测。

预测的 Vs 经 Brocher(2005) 生成 Vp/ρ，送入物理求解器（不经任何 NN）
计算频散，与数据集观测频散对比。对照：真值 Vs（应≈0）与训练均值剖面。
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

SWAVE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SWAVE_ROOT / "src"))

from swave.config import load_dataset_config  # noqa: E402
from swave.empirical import material_properties  # noqa: E402
from swave.secular import LayeredModel  # noqa: E402
from swave.solver import DispersionSolver  # noqa: E402

_STATE = {}


def _init():
    config = load_dataset_config(SWAVE_ROOT / "configs/dataset.toml")
    _STATE["config"] = config
    _STATE["depth"] = np.arange(20) * config.geology.thickness_km


def _forward(vs_km):
    vp, rho = material_properties(vs_km, "brocher05")
    model = LayeredModel(depth=_STATE["depth"], density=rho, vs=vs_km, vp=vp)
    result = DispersionSolver(model, _STATE["config"].physics).solve_grid(
        strategy="quadratic")
    return result.phase_velocity


def _worker(task):
    which, vs_km, obs_i, mask_i = task
    curves = _forward(vs_km)
    m = mask_i & np.isfinite(curves)
    return which, float(np.abs(curves[m] - obs_i[m]).mean() * 1000)


def main() -> None:
    n_samples = 60
    z = np.load(SWAVE_ROOT / "inversion/results/fold_data_cache.npz")
    vs_true, fold, kind = z["vs"], z["fold"], z["kind"]
    pred = np.load(SWAVE_ROOT / "inversion/results/predictions_invnet.npy")

    obs_list, mask_list = [], []
    for sp in sorted(SWAVE_ROOT.glob("data/production-w64/shard-*.h5")):
        with h5py.File(sp) as f:
            ids = f["sample_id"][:]
            sel = (ids % 100) >= 95
            obs_list.append(f["phase_velocity"][sel])
            mask_list.append(f["valid_mask"][sel])
    obs = np.concatenate(obs_list)
    obs_mask = np.concatenate(mask_list)

    test_vs = vs_true[fold >= 95]
    test_kind = kind[fold >= 95]
    mean_profile = vs_true[fold < 90].mean(axis=0)

    rng = np.random.default_rng(1)
    idx = rng.choice(len(pred), n_samples, replace=False)

    tasks = []
    for i in idx:
        tasks.append(("InvNet预测", pred[i], obs[i], obs_mask[i]))
        tasks.append(("真值Vs", test_vs[i], obs[i], obs_mask[i]))
        tasks.append(("训练均值", mean_profile, obs[i], obs_mask[i]))

    stats: dict[str, list] = {"InvNet预测": [], "真值Vs": [], "训练均值": []}
    with ProcessPoolExecutor(max_workers=48, initializer=_init) as pool:
        for which, mae in pool.map(_worker, tasks, chunksize=2):
            stats[which].append(mae)

    print("===== 物理求解器（无 NN 参与）频散残差 =====")
    for name, arr in stats.items():
        arr = np.asarray(arr)
        print(f"{name}: MAE {arr.mean():.1f} m/s | 中位 {np.median(arr):.1f} | "
              f"p95 {np.percentile(arr, 95):.1f}")

    # 分族（InvNet）
    per_kind: dict[int, list] = {0: [], 1: [], 2: [], 3: []}
    for k, i in enumerate(idx):
        per_kind[test_kind[i]].append(stats["InvNet预测"][k])
    names = {0: "NORMAL", 1: "LOW", 2: "HIGH", 3: "COUPLED"}
    print("\nInvNet 分族:")
    for kid, arr in per_kind.items():
        if arr:
            print(f"  {names[kid]}: n={len(arr)}, MAE {np.mean(arr):.1f} m/s")


if __name__ == "__main__":
    main()
