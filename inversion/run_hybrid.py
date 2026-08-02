#!/usr/bin/env python3
"""混合反演主流程：反演网络初值 + 批量投影 Adam 精修。

方法 C（混合法）：
  1) InverseNet 给出含训练分布先验的初值 x0（深层非唯一区间被先验约束）；
  2) 在可微正演代理上做批量投影 Adam 精修，直接最小化
       数据残差 0.5*mean((F(x)-d)^2 * w) + λ·Σ diff(Vs)^2
     每步 clamp 到 NN 定义域 [0.3, 2.6] km/s；
  3) 逐样本跟踪历史最优（含第 0 步=初值）——结果绝不差于初值。

用法：
  # 正式混合反演（测试折前 2000 个，与 iNETT 同子集）
  .venv/bin/python inversion/run_hybrid.py --warmstart invnet --name hybrid --samples 2000
  # 对照：ICNN 初值（iNETT 的热启动）+ Adam 精修
  .venv/bin/python inversion/run_hybrid.py --warmstart icnn-file --name adam --samples 2000
  # 调参（只用 val 折！）
  .venv/bin/python inversion/run_hybrid.py --split val --samples 500 --lr 5e-2 --lambda-smooth 1e-3
  # 评估对比（读取各方法结果 + iNETT 基线）
  .venv/bin/python inversion/run_hybrid.py --evaluate --samples 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SWAVE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SWAVE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forward_torch import TorchForward, VS_MIN, VS_MAX  # noqa: E402
from inverse_net import load_inverse_net, predict_physical  # noqa: E402
from train_inverse_net import build_inputs, load_split  # noqa: E402

RESULTS_DIR = SWAVE_ROOT / "inversion/results"
INETT_RESULTS = Path("/home/smbu/iNETT-iterated-network-Tikhonov-method"
                     "/sw_inversion_20x100m/results/swave_inversion")
KIND_NAMES = {0: "NORMAL", 1: "LOW_VELOCITY", 2: "HIGH_VELOCITY",
              3: "COUPLED"}


# ---------------- Adam 精修 ----------------

def adam_invert(forward: TorchForward, d_obs: torch.Tensor, x0: torch.Tensor,
                *, steps: int = 300, lr: float = 5e-2,
                lambda_smooth: float = 1e-3, mu_prox: float = 0.0,
                weights: torch.Tensor | None = None,
                restarts: int = 1, restart_sigma: float = 0.05,
                select: str = "best", seed: int = 0):
    """批量投影 Adam 反演。

    损失 = 数据残差 + λ·Σ diff(Vs)^2 + μ·‖x-x0‖^2（信赖域，
    防止沿非唯一方向远离含先验的初值）。
    d_obs: (N,476) 物理 km/s；x0: (N,20) 物理 km/s。
    返回 (best_x (N,20), best_loss (N,), init_loss (N,), clamped_frac (N,))。
    逐样本跟踪含初值在内的历史最优（select="best"），或返回末步（"last"）。
    """
    n = len(x0)
    if restarts > 1:
        # restart 沿 batch 维堆叠一次算完
        gen = torch.Generator(device=x0.device).manual_seed(seed)
        noise = torch.randn((restarts, n, 20), generator=gen,
                            device=x0.device) * restart_sigma
        x_all = (x0.unsqueeze(0) + noise).reshape(-1, 20)
        d_all = d_obs.repeat(restarts, 1)
    else:
        x_all, d_all = x0.clone(), d_obs

    x_anchor = x_all.clone()  # 信赖域锚点
    x = x_all.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.02)

    def total_loss(values):
        misfit = forward.data_misfit(values, d_all, weights)
        smooth = (values[:, 1:] - values[:, :-1]).pow(2).sum(dim=1)
        loss = misfit + lambda_smooth * smooth
        if mu_prox > 0:
            loss = loss + mu_prox * (values - x_anchor).pow(2).sum(dim=1)
        return loss, misfit

    with torch.no_grad():
        init_loss, init_misfit = total_loss(x_all)
    best_x = x_all.clone()
    best_loss = init_loss.clone()

    clamp_count = torch.zeros_like(best_loss)
    for step in range(steps):
        loss, _ = total_loss(x)
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            clamped = (x < VS_MIN) | (x > VS_MAX)
            clamp_count += clamped.any(dim=1).float()
            x.clamp_(VS_MIN, VS_MAX)
            new_loss, _ = total_loss(x)
            better = new_loss < best_loss
            best_x[better] = x[better]
            best_loss[better] = new_loss[better]

    if select == "last":
        best_x = x.detach().clone()
        best_loss = total_loss(best_x)[0]

    final_x = best_x[:n] if restarts > 1 else best_x
    if restarts > 1:
        # 逐样本在 restart 间取最优（按数据残差+平滑项）
        cand = best_x.reshape(restarts, n, 20)
        scores = []
        with torch.no_grad():
            for r in range(restarts):
                s, _ = (lambda v: (forward.data_misfit(v, d_obs, weights)
                                   + lambda_smooth
                                   * (v[:, 1:] - v[:, :-1]).pow(2).sum(dim=1))
                        )(cand[r])
                scores.append(s)
        pick = torch.stack(scores).argmin(dim=0)
        final_x = cand[pick, torch.arange(n, device=x0.device)]
        final_loss = torch.stack(scores)[pick, torch.arange(n, device=x0.device)]
    else:
        final_loss = best_loss

    clamped_frac = (clamp_count[:n] / steps).detach()
    return final_x.detach(), final_loss.detach(), init_loss[:n].detach(), clamped_frac


# ---------------- 数据与初值 ----------------

def load_fold_data(dataset_dir: Path):
    """加载数据集并构建填充后的 (476,) 输入。缓存到 npz 加速重复运行。"""
    cache = RESULTS_DIR / "fold_data_cache.npz"
    if cache.exists():
        z = np.load(cache)
        return z["vs"], z["X"], z["fold"], z["kind"]
    data = load_split(dataset_dir)
    fill_payload = torch.load(RESULTS_DIR / "inverse_net_seed0_best.pt",
                              map_location="cpu", weights_only=False)
    fill = np.asarray(fill_payload["fill_values"], dtype=np.float64)
    X = build_inputs(data["pv"], data["mask"], fill)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, vs=data["vs"], X=X, fold=data["fold"],
                        kind=data["kind"])
    return data["vs"], X, data["fold"], data["kind"]


def warm_start(name: str, X: np.ndarray, device: torch.device) -> np.ndarray:
    """返回 (N,20) 物理 km/s 初值。"""
    if name == "invnet":
        ckpts = sorted(RESULTS_DIR.glob("inverse_net_seed*_best.pt"))
        if not ckpts:
            raise FileNotFoundError("未找到 inverse_net checkpoint，先训练")
        preds = []
        for ckpt in ckpts:
            model, norm = load_inverse_net(ckpt, device=str(device))
            preds.append(predict_physical(model, norm, X,
                                          device=str(device)))
        return np.mean(preds, axis=0)
    if name == "icnn-file":
        # iNETT 的 ICNN 校正结果（测试折顺序一致），仅用于对照
        icnn = np.load(INETT_RESULTS / "predictions_icnn.npy")
        return icnn[:len(X)]
    if name == "mean":
        # 训练折逐层均值（不含任何测试信息），用于验证精修机制本身
        cache = RESULTS_DIR / "fold_data_cache.npz"
        z = np.load(cache)
        train_mean = z["vs"][z["fold"] < 90].mean(axis=0)
        return np.broadcast_to(train_mean, (len(X), 20)).copy()
    raise ValueError(f"未知初值: {name}")


# ---------------- 评估 ----------------

def compute_metrics(y_true, y_pred):
    """MAE/RMSE 全局；R2 按层计算后平均（对齐 sklearn r2_score 默认
    multioutput='uniform_average'，与 iNETT 评估口径一致）。"""
    err = y_pred - y_true
    per_layer_r2 = []
    for j in range(y_true.shape[1]):
        ss_res = float(np.sum(err[:, j] ** 2))
        ss_tot = float(np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2))
        per_layer_r2.append(1.0 - ss_res / ss_tot)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "R2": float(np.mean(per_layer_r2))}


def evaluate(samples: int, names: list[str]) -> None:
    """同子集（测试折前 samples 个）对比各方法 + iNETT 基线。"""
    true_vs = np.load(INETT_RESULTS / "true_vs.npy")[:samples]
    kinds = np.load(Path(INETT_RESULTS).parent.parent
                    / "data/swave_inversion/test_model_kind.npy")[:samples]

    methods = {}
    for label, path in [
            ("icnn", INETT_RESULTS / "predictions_icnn.npy"),
            ("inett", INETT_RESULTS / "predictions_inett.npy")]:
        if path.exists():
            methods[label] = np.load(path)[:samples]
    for name in names:
        path = RESULTS_DIR / f"predictions_{name}.npy"
        if path.exists():
            methods[name] = np.load(path)[:samples]

    summary = {}
    print(f"\n===== 同子集对比（测试折前 {samples} 个样本）=====")
    print(f"{'方法':<10} {'MAE(m/s)':>9} {'RMSE(m/s)':>10} {'R2':>8}")
    for label, pred in methods.items():
        entry = {"overall": compute_metrics(true_vs, pred),
                 "by_model_kind": {}}
        for kind_id, kind_name in KIND_NAMES.items():
            m = kinds == kind_id
            if m.sum():
                entry["by_model_kind"][kind_name] = compute_metrics(
                    true_vs[m], pred[m])
        summary[label] = entry
        o = entry["overall"]
        print(f"{label:<10} {o['MAE'] * 1000:>9.1f} {o['RMSE'] * 1000:>10.1f} "
              f"{o['R2']:>8.4f}")

    # 逐层 MAE
    layers = {}
    for label, pred in methods.items():
        layers[label] = (np.abs(pred - true_vs).mean(axis=0) * 1000).tolist()
    summary["per_layer_MAE_m_s"] = layers

    with open(RESULTS_DIR / "metrics_comparison.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nsaved {RESULTS_DIR / 'metrics_comparison.json'}")

    # 剖面对比图：前 5 样本 + 前 5 耦合样本
    styles = {"true": ("k-", "True"), "icnn": ("g--", "ICNN"),
              "inett": ("r-.", "iNETT"), "invnet": ("m--", "InvNet"),
              "adam": ("c--", "Adam"), "hybrid": ("b-", "Hybrid")}
    for tag, idx in [("first5", np.arange(5)),
                     ("coupled", np.where(kinds == 3)[0][:5])]:
        if len(idx) == 0:
            continue
        fig, axes = plt.subplots(1, len(idx),
                                 figsize=(4.4 * len(idx), 5), sharey=True)
        depth_edges = np.arange(21) * 100.0
        for col, i in enumerate(idx):
            ax = axes[col] if len(idx) > 1 else axes
            for label in ["true", *methods.keys()]:
                style, legend = styles.get(label, ("-", label))
                values = true_vs[i] if label == "true" else methods[label][i]
                px = np.repeat(values, 2)  # km/s
                py = np.column_stack(
                    [depth_edges[:-1], depth_edges[1:]]).reshape(-1)
                lw = 2.2 if label == "true" else 1.6
                ax.plot(px, py, style, linewidth=lw, label=legend)
            ax.axhspan(200.0, 1200.0, color="#eda100", alpha=0.08)
            ax.set_ylim(2500.0, 0.0)
            ax.set_xlim(-0.1, 3.0)
            ax.set_xlabel("Vs (km/s)")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"sample #{i}")
            if col == 0:
                ax.set_ylabel("Depth (m)")
                ax.legend(fontsize=8)
        fig.tight_layout()
        out = RESULTS_DIR / f"vs_comparison_hybrid_{tag}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"saved {out}")


# ---------------- 主流程 ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir",
                        default=str(SWAVE_ROOT / "data/production-w64"))
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--warmstart", default="invnet",
                        choices=["invnet", "icnn-file", "mean"])
    parser.add_argument("--name", default="hybrid")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--lambda-smooth", type=float, default=1e-3)
    parser.add_argument("--mu-prox", type=float, default=0.0,
                        help="信赖域系数 mu*||x-x0||^2（防止远离初值）")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--restart-sigma", type=float, default=0.05)
    parser.add_argument("--select", choices=["best", "last"], default="best")
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument("--evaluate", action="store_true",
                        help="只做评估对比（--name 可传多个，逗号分隔）")
    args = parser.parse_args()

    if args.evaluate:
        evaluate(args.samples, args.name.split(","))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}; split={args.split}, 样本数={args.samples}, "
          f"初值={args.warmstart}")

    vs_all, X_all, fold, kind = load_fold_data(Path(args.dataset_dir))
    sel = (fold >= 95) if args.split == "test" else (fold >= 90) & (fold < 95)
    X = X_all[sel][:args.samples]
    true_vs = vs_all[sel][:args.samples]
    print(f"数据: X {X.shape}, 真值 {true_vs.shape}")

    x0 = warm_start(args.warmstart, X, device)
    x0 = np.clip(x0, VS_MIN, VS_MAX)
    mae0 = np.abs(x0 - true_vs).mean() * 1000
    print(f"初值 MAE: {mae0:.2f} m/s")

    forward = TorchForward(device=str(device))
    weights = None
    weights_path = RESULTS_DIR / "freq_weights.npy"
    if not args.no_weights and weights_path.exists():
        weights = torch.as_tensor(np.load(weights_path), device=device)
        print(f"使用频率权重: {weights_path}")

    d_obs = torch.as_tensor(X, device=device)
    x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)

    t0 = time.time()
    best_x, final_loss, init_loss, clamped_frac = adam_invert(
        forward, d_obs, x0_t, steps=args.steps, lr=args.lr,
        lambda_smooth=args.lambda_smooth, mu_prox=args.mu_prox,
        weights=weights,
        restarts=args.restarts, restart_sigma=args.restart_sigma,
        select=args.select)
    elapsed = time.time() - t0

    pred = best_x.cpu().numpy()
    mae = np.abs(pred - true_vs).mean() * 1000
    init_res = init_loss.cpu().numpy()
    final_res = final_loss.cpu().numpy()
    improved = final_res < init_res - 1e-12
    guard_ok = bool(np.all(final_res <= init_res + 1e-12))
    print(f"\n===== {args.name} 结果 =====")
    print(f"MAE: {mae0:.2f} -> {mae:.2f} m/s；改善样本 {improved.mean() * 100:.1f}%")
    print(f"损失（残差+平滑）: {init_res.mean():.3e} -> {final_res.mean():.3e}")
    print(f"守卫审计: {'通过' if guard_ok else '失败！'} "
          f"(100% 样本 final <= init 应为真)")
    print(f"clamp 样本比例: {(clamped_frac.cpu().numpy() > 0).mean() * 100:.1f}%, "
          f"耗时 {elapsed:.1f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / f"predictions_{args.name}.npy", pred)
    np.savez(RESULTS_DIR / f"{args.name}_residuals.npz",
             init=init_res, final=final_res)
    np.save(RESULTS_DIR / f"{args.name}_status.npy",
            improved.astype(np.int8))
    np.save(RESULTS_DIR / f"{args.name}_clamped_frac.npy",
            clamped_frac.cpu().numpy())
    print(f"saved {RESULTS_DIR}/predictions_{args.name}.npy 等")


if __name__ == "__main__":
    main()
