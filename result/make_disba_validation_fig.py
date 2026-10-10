# -*- coding: utf-8 -*-
"""生成交叉验证图:样本 98 上 swave 各阶模态(平滑曲线) vs disba 各阶模态(散点)。

disba 的模态编号在 mode-kissing 后跳到更高的分支,其 "M1" 散点实际落在
swave 的 M3(甚至 M4)分支上——直观展示两个实现的根一致、编号语义不同。

用法(在 swave 根目录,需 anaconda python3 的 disba):
    python3 result/make_disba_validation_fig.py
"""
import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from disba import PhaseDispersion

zh = [f.name for f in font_manager.fontManager.ttflist
      if any(k in f.name for k in ["Noto Sans CJK", "WenQuanYi", "SimHei"])]
if zh:
    plt.rcParams["font.sans-serif"] = [zh[0], "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FREQS = np.arange(0.5, 60.25, 0.5)
PERIODS = np.sort(1.0 / FREQS)

f = h5py.File("data/production/shard-00000.h5", "r")
vs = f["vs"][98].astype(np.float64)
vp = f["vp"][98].astype(np.float64)
rho = f["density"][98].astype(np.float64)
ref = f["phase_velocity"][98]
mask = f["valid_mask"][98]
f.close()

pd = PhaseDispersion(np.array([0.1] * 19 + [0.0]), vp, vs, rho)
disba = {}
for mode in range(4):
    try:
        r = pd(PERIODS, mode=mode, wave="rayleigh")
        freqs_d = 1.0 / np.asarray(r.period)
        disba[mode] = (freqs_d, np.asarray(r.velocity))
    except Exception:
        pass

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
labels = ["M0(基阶)", "M1(一阶)", "M2(二阶)", "M3(三阶)"]

fig, ax = plt.subplots(figsize=(9, 5.5))
for m in range(4):
    ax.plot(FREQS[mask[m]], ref[m][mask[m]], color=colors[m], lw=1.8,
            label=f"swave {labels[m]}")
for m in [1, 2, 3]:
    if m not in disba:
        continue
    fd, vd = disba[m]
    sel = fd <= 15
    ax.plot(fd[sel], vd[sel], "x--", color="black", ms=6, lw=0.8,
            alpha=0.35 + 0.2 * m, label=f'disba "M{m}"(编号跳支)' if m == 1 else f'disba "M{m}"')

ax.annotate('disba "M1" 在 1 Hz 跳到 2.28 km/s,\n该值恰为 swave M3 的根(久期函数已验证)',
            xy=(1.0, 2.2827), xytext=(3.2, 2.05), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.set_xlim(0.4, 15)
ax.set_xlabel("频率 (Hz)")
ax.set_ylabel("相速度 (km/s)")
ax.set_title('样本 98:swave 求解器 vs 独立实现 disba——根值一致,disba 模态编号跳支')
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("result/figures/disba-cross-validation-sample98.png", dpi=200)
print("saved result/figures/disba-cross-validation-sample98.png")
