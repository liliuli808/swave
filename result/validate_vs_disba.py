# -*- coding: utf-8 -*-
"""交叉验证:swave 自实现 Dunkin δ 求解器(数据集标签) vs disba(独立 δ 矩阵实现)。

用 anaconda python3 运行(disba 只装在 anaconda):
    python3 /home/smbu/swave/result/validate_vs_disba.py

对比内容:
1. 代表样本 90-99(文档中用过的保留样本):逐模态逐频点差值统计 + 最差值
2. 随机 200 个模型的总体统计
3. 模态存在性一致性:swave valid_mask=False 处 disba 是否也找不到根
"""
import h5py
import numpy as np
from disba import PhaseDispersion

FREQS = np.arange(0.5, 60.25, 0.5)
PERIODS = 1.0 / FREQS  # disba 用周期
THICKNESS_FINITE = 0.1  # km,19 个有限层


def disba_dispersion(vs, vp, rho, n_modes=4):
    """返回 (4,120) 数组,模态不存在处为 NaN。"""
    thickness = np.array([THICKNESS_FINITE] * 19 + [0.0])
    pd = PhaseDispersion(thickness, np.asarray(vp, np.float64),
                         np.asarray(vs, np.float64), np.asarray(rho, np.float64))
    out = np.full((n_modes, len(PERIODS)), np.nan)
    order = np.argsort(PERIODS)  # disba 要求周期升序
    inv = np.argsort(order)
    for mode in range(n_modes):
        try:
            res = pd(PERIODS[order], mode=mode, wave="rayleigh")
            out[mode] = res.velocity[inv]
        except Exception:
            # 部分频点模态不存在会导致整次调用失败,退化为逐频点
            for j, t in enumerate(PERIODS[order]):
                try:
                    out[mode, order[j]] = pd(np.array([t]), mode=mode, wave="rayleigh").velocity[0]
                except Exception:
                    pass
    return out


def compare(vs, vp, rho, ref_vel, ref_mask):
    """返回 (diff_values, n_exist_mismatch)。"""
    mine = disba_dispersion(vs, vp, rho)
    diffs = []
    mismatch = 0
    for mode in range(4):
        swave_exist = ref_mask[mode]
        disba_exist = np.isfinite(mine[mode])
        mismatch += int(np.sum(swave_exist != disba_exist))
        both = swave_exist & disba_exist
        diffs.append(np.abs(mine[mode][both] - ref_vel[mode][both]))
    return np.concatenate(diffs), mismatch


def load_samples(shard_path, indices):
    f = h5py.File(shard_path, "r")
    data = {k: f[k][:] for k in ["vs", "vp", "density", "phase_velocity", "valid_mask", "sample_id"]}
    f.close()
    order = {int(s): i for i, s in enumerate(data["sample_id"])}
    return [{k: data[k][order[i]] for k in data} for i in indices]


def main():
    shard0 = "/home/smbu/swave/data/production/shard-00000.h5"

    print("== 代表样本 90-99(逐样本统计,单位 km/s)==")
    all_diffs, total_mismatch, total_points = [], 0, 0
    for s in load_samples(shard0, range(90, 100)):
        diffs, mismatch = compare(s["vs"], s["vp"], s["density"], s["phase_velocity"], s["valid_mask"])
        all_diffs.append(diffs)
        total_mismatch += mismatch
        total_points += len(diffs)
        print(f"sample {int(s['sample_id']):3d}: n={len(diffs):4d}  MAE={diffs.mean():.2e}  "
              f"max={diffs.max():.2e}  存在性不一致点数={mismatch}")
    all_diffs = np.concatenate(all_diffs)
    print(f"合计: n={total_points}  MAE={all_diffs.mean():.3e}  RMSE={np.sqrt((all_diffs**2).mean()):.3e}  "
          f"P95={np.percentile(all_diffs, 95):.3e}  max={all_diffs.max():.3e}  存在性不一致={total_mismatch}")

    print("\n== 随机 200 模型(shard 90,反演保留分片)==")
    rng = np.random.default_rng(0)
    shard90 = "/home/smbu/swave/data/production/shard-00090.h5"
    idx = rng.choice(10000, 200, replace=False)
    f = h5py.File(shard90, "r")
    data = {k: f[k][:] for k in ["vs", "vp", "density", "phase_velocity", "valid_mask"]}
    f.close()
    all_diffs, total_mismatch = [], 0
    for i in idx:
        diffs, mismatch = compare(data["vs"][i], data["vp"][i], data["density"][i],
                                  data["phase_velocity"][i], data["valid_mask"][i])
        all_diffs.append(diffs)
        total_mismatch += mismatch
    all_diffs = np.concatenate(all_diffs)
    print(f"合计: n={len(all_diffs)}  MAE={all_diffs.mean():.3e}  RMSE={np.sqrt((all_diffs**2).mean()):.3e}  "
          f"P95={np.percentile(all_diffs, 95):.3e}  max={all_diffs.max():.3e}  存在性不一致={total_mismatch}")


if __name__ == "__main__":
    main()
