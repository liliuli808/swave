#!/usr/bin/env python3
"""由训练折 valid_mask 统计生成频率权重 (476,)。

数据准备（iNETT 侧）曾把无效频点（高阶模态低频段物理缺失）线性插值
填满；这些频点携带的信息量低于真实有效频点。本脚本按训练折逐频点
有效率构造降权权重，供 Adam 精修的数据残差使用：

  w = clip(valid_frac, 0.2, 1.0) × (1/σ_mode)，归一化使 mean(w)=1

σ_mode 取正演代理的逐模态测试 MAE（runs/production-48g/evaluation.json），
让拟合不把残差压到代理自身噪声地板以下。

输出：inversion/results/freq_weights.npy (476,) float32，mode-major。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SWAVE_ROOT = Path(__file__).resolve().parent.parent
DROP_FREQ_COLS = 1
FLOOR = 0.2  # 完全缺失频点降权到 0.2 而非归零（插值仍是光滑信号）


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir",
                        default=str(SWAVE_ROOT / "data/production-w64"))
    parser.add_argument("--evaluation",
                        default=str(SWAVE_ROOT
                                    / "runs/production-48g/evaluation.json"))
    parser.add_argument("--output",
                        default=str(SWAVE_ROOT
                                    / "inversion/results/freq_weights.npy"))
    args = parser.parse_args()

    # 训练折逐频点有效率（去 0.5 Hz 列后 (4,119)）
    valid_count = np.zeros((4, 120), dtype=np.int64)
    total = 0
    for shard_path in sorted(Path(args.dataset_dir).glob("shard-*.h5")):
        with h5py.File(shard_path, "r") as handle:
            sample_ids = handle["sample_id"][:]
            train = (sample_ids % 100) < 90
            valid_count += handle["valid_mask"][train].sum(axis=0)
            total += int(train.sum())
    valid_frac = (valid_count / total)[:, DROP_FREQ_COLS:]
    print(f"训练折样本 {total}；逐模态平均有效率: "
          + ", ".join(f"M{m}={valid_frac[m].mean():.4f}" for m in range(4)))

    # 逐模态代理噪声（km/s -> 相对权重 1/σ，归一化到 M0=1）
    with open(args.evaluation) as f:
        evaluation = json.load(f)
    sigma = np.array([evaluation[f"mode_{m}"]["mae_km_s"] for m in range(4)])
    mode_weight = sigma[0] / sigma  # M0 权重为 1，高模态降权
    print(f"代理逐模态 MAE: {sigma * 1000} m/s -> 权重 {mode_weight}")

    w = np.clip(valid_frac, FLOOR, 1.0) * mode_weight[:, None]
    w = (w / w.mean()).astype(np.float32).reshape(-1)  # mode-major (476,)
    print(f"权重范围 [{w.min():.3f}, {w.max():.3f}]，均值 {w.mean():.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, w)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
