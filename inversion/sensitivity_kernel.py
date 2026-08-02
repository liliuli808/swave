#!/usr/bin/env python3
"""计算并绘制正演代理的敏感核（Jacobian ∂频散/∂Vs）。

对每族中位代表样本，用 autograd 精确计算 J (476×20)，
可视化为 频率×深度 热图（每模态一幅），并给出逐层敏感度占比
——定量解释反演误差随深度增长的物理根源。
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SWAVE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SWAVE_ROOT / "src"))
sys.path.insert(0, str(SWAVE_ROOT / "inversion"))

from forward_torch import TorchForward  # noqa: E402

KIND_TITLES = {0: "Normal", 1: "Low-velocity", 2: "High-velocity",
               3: "Coupled high+low"}
FREQUENCIES = np.arange(1.0, 60.0 + 0.25, 0.5)  # 去 0.5 Hz 列后 119 点
DEPTHS = (np.arange(20) + 0.5) * 100.0  # 层中心深度 (m)


def pick_median(kinds, vs, kind):
    rows = np.where(kinds == kind)[0]
    order = rows[np.argsort(vs[rows].mean(axis=1))]
    return int(order[len(order) // 2])


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    forward = TorchForward(device=device)

    with h5py.File(SWAVE_ROOT / "data/production-w64/shard-00000.h5") as f:
        kinds = f["model_kind"][:]
        vs_all = f["vs"][:]
        ids = f["sample_id"][:]

    # 每族中位样本的 Jacobian (4,119,20)
    jacobians = {}
    for kind in KIND_TITLES:
        row = pick_median(kinds, vs_all, kind)
        vs_t = torch.as_tensor(vs_all[row:row + 1], dtype=torch.float32,
                               device=device)
        J = torch.autograd.functional.jacobian(
            lambda v: forward.forward_flat(v), vs_t)
        jacobians[kind] = J.reshape(476, 20).cpu().numpy()
        print(f"{KIND_TITLES[kind]}: sample_id={ids[row]}")

    # ---------- 图 1：敏感核热图（行=模态，列=族） ----------
    from matplotlib import colors as mcolors
    fig, axes = plt.subplots(4, 4, figsize=(19, 13), sharex=True, sharey=True)
    vmax = np.percentile(np.abs(np.stack(list(jacobians.values()))), 99)
    norm = mcolors.PowerNorm(gamma=0.35, vmin=0, vmax=vmax)  # 拉开弱信号
    for col, (kind, title) in enumerate(KIND_TITLES.items()):
        J = jacobians[kind].reshape(4, 119, 20)
        for mode in range(4):
            ax = axes[mode, col]
            im = ax.pcolormesh(DEPTHS, FREQUENCIES, np.abs(J[mode]),
                               cmap="Blues", norm=norm, shading="auto")
            ax.set_ylim(60, 1)
            if mode == 0:
                ax.set_title(title, fontsize=11)
            if mode == 3:
                ax.set_xlabel("Depth (m)")
            if col == 0:
                ax.set_ylabel(f"M{mode}  Frequency (Hz)")
    fig.suptitle("Sensitivity kernel |∂c(mode,f) / ∂Vs(layer)| "
                 "(median sample per family)", y=0.98)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    # 独立的 colorbar 轴，避免与热图重叠
    cbar_ax = fig.add_axes([0.945, 0.08, 0.012, 0.86])
    fig.colorbar(im, cax=cbar_ax, label="dimensionless")
    out1 = SWAVE_ROOT / "results/sensitivity-kernels.png"
    fig.savefig(out1, dpi=300, facecolor="#fcfcfb")
    print(f"saved {out1}")

    # ---------- 图 2 + 表：逐层敏感度占比 ----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
    print("\n逐层敏感度占比（%，Jacobian 列范数 / 总范数）：")
    header = "层    " + "".join(f"{d + 1:>6}" for d in range(20))
    print(header)
    for (kind, title), color in zip(KIND_TITLES.items(), colors):
        J = jacobians[kind]
        share = np.linalg.norm(J, axis=0)
        share = share / share.sum() * 100
        ax.plot(np.arange(1, 21), share, "o-", color=color, label=title,
                linewidth=1.8, markersize=4)
        print(f"{title:<16}" + "".join(f"{v:6.1f}" for v in share))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sensitivity share (%)")
    ax.set_xticks(range(1, 21))
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Per-layer share of total dispersion sensitivity")
    fig.tight_layout()
    out2 = SWAVE_ROOT / "results/sensitivity-per-layer.png"
    fig.savefig(out2, dpi=300, facecolor="#fcfcfb")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
