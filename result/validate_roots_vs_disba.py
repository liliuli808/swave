# -*- coding: utf-8 -*-
"""根集合对比:swave 求解器 vs disba(忽略模态编号,只比根的位置)。

结论预期:swave 的每个根都应在 disba 的根集合中找到匹配(反之 disba 的前几个根亦然),
从而证明两个独立 δ 矩阵实现求出的频散根一致;差异只来自 disba 的模态编号跟踪。

用法: python3 result/validate_roots_vs_disba.py
"""
import h5py
import numpy as np
from disba import PhaseDispersion

FREQS = np.arange(0.5, 60.25, 0.5)
PERIODS = np.sort(1.0 / FREQS)


def disba_roots(vs, vp, rho, max_mode=6):
    """每个频点的根集合(模态 0..max_mode 合并去重,频率升序对齐)。"""
    pd = PhaseDispersion(np.array([0.1] * 19 + [0.0]),
                         np.asarray(vp, np.float64), np.asarray(vs, np.float64),
                         np.asarray(rho, np.float64))
    roots = [[] for _ in FREQS]
    for mode in range(max_mode + 1):
        try:
            r = pd(PERIODS, mode=mode, wave="rayleigh")
        except Exception:
            continue
        for t, v in zip(r.period, r.velocity):
            k = int(np.argmin(np.abs(FREQS - 1.0 / t)))
            roots[k].append(v)
    return [np.sort(np.array(rk)) for rk in roots]


def main():
    rng = np.random.default_rng(0)
    f = h5py.File("/home/smbu/swave/data/production/shard-00090.h5", "r")
    data = {k: f[k][:] for k in ["vs", "vp", "density", "phase_velocity", "valid_mask"]}
    f.close()

    idx = rng.choice(10000, 100, replace=False)
    match_err, missing = [], 0  # swave 根在 disba 集合中的最近距离;找不到(<0.5 km/s 内)的个数
    for i in idx:
        vs_hs = data["vs"][i][-1]
        droots = disba_roots(data["vs"][i], data["vp"][i], data["density"][i])
        ref, mask = data["phase_velocity"][i], data["valid_mask"][i]
        for k in range(len(FREQS)):
            sroots = ref[mask[:, k], k]
            dset = droots[k][droots[k] < vs_hs]
            for s in sroots:
                if dset.size == 0:
                    missing += 1
                    continue
                d = np.min(np.abs(dset - s))
                if d < 0.5:
                    match_err.append(d)
                else:
                    missing += 1
    match_err = np.array(match_err)
    print(f"swave 根总数(有效点): {len(match_err) + missing}")
    print(f"在 disba 根集合中匹配到: {len(match_err)}  未匹配: {missing}")
    print(f"匹配误差: MAE={match_err.mean():.3e}  P95={np.percentile(match_err, 95):.3e}  "
          f"max={match_err.max():.3e} km/s")


if __name__ == "__main__":
    main()
