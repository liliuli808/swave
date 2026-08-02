#!/usr/bin/env python3
"""训练监督反演网络：物理频散 (476,) -> 物理 Vs (20,)。

数据直接读 data/production-w64 的 HDF5 分片，按 sample_id % 100 分折
（<90 训练 / 90-94 验证 / >=95 测试，与正演代理及 iNETT 管线一致）。

要点：
- 无效频点（valid_mask=False，高阶模态低频物理缺失）用训练折的
  逐(模态,频点)均值填充，填充向量随 checkpoint 保存，预测时一致复用；
- 输入/输出各自 z-score 归一化（训练折统计），存进 checkpoint；
- 全量数据驻留 GPU（900000×476×4B ≈ 1.7GB），batch 8192，无 DataLoader；
- AdamW + CosineAnnealing + bf16 autocast + val MSE 早停。

用法：
    .venv/bin/python inversion/train_inverse_net.py --seeds 0
    .venv/bin/python inversion/train_inverse_net.py --seeds 0 1 2 --epochs 150
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

import sys
SWAVE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SWAVE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inverse_net import InverseNet  # noqa: E402

DROP_FREQ_COLS = 1
RESULTS_DIR = SWAVE_ROOT / "inversion/results"


def load_split(dataset_dir: Path):
    """读全部样本及其 fold。返回 dict of arrays（物理单位）。"""
    vs_all, pv_all, mask_all, ids_all, kind_all = [], [], [], [], []
    for shard_path in sorted(dataset_dir.glob("shard-*.h5")):
        with h5py.File(shard_path, "r") as handle:
            vs_all.append(handle["vs"][:])
            pv_all.append(handle["phase_velocity"][:])
            mask_all.append(handle["valid_mask"][:])
            ids_all.append(handle["sample_id"][:])
            kind_all.append(handle["model_kind"][:])
    ids = np.concatenate(ids_all)
    return {
        "vs": np.concatenate(vs_all),
        "pv": np.concatenate(pv_all)[:, :, DROP_FREQ_COLS:],   # (N,4,119)
        "mask": np.concatenate(mask_all)[:, :, DROP_FREQ_COLS:],
        "ids": ids,
        "kind": np.concatenate(kind_all),
        "fold": ids % 100,
    }


def build_inputs(pv, mask, fill):
    """无效频点填充 + 展平 (N,476)。"""
    x = np.where(mask, pv, np.nan).astype(np.float32)
    nan_cells = ~np.isfinite(x)
    x[nan_cells] = np.broadcast_to(fill[None], x.shape)[nan_cells]
    return x.reshape(len(x), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir",
                        default=str(SWAVE_ROOT / "data/production-w64"))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"设备: {device}")

    # ---------- 数据 ----------
    print("读取数据集 ...")
    t0 = time.time()
    data = load_split(Path(args.dataset_dir))
    vs, pv, mask, fold = data["vs"], data["pv"], data["mask"], data["fold"]
    train_sel, val_sel, test_sel = fold < 90, (fold >= 90) & (fold < 95), fold >= 95

    # 逐(模态,频点)训练折均值作填充向量 (4,119)
    pv_train = pv[train_sel]
    mask_train = mask[train_sel]
    fill = np.full((4, pv.shape[2]), np.nan, dtype=np.float64)
    for m in range(4):
        for fi in range(pv.shape[2]):
            valid = mask_train[:, m, fi]
            if valid.any():
                fill[m, fi] = pv_train[valid, m, fi].mean()
    # 训练折全无效的格子退化为模态均值
    for m in range(4):
        bad = ~np.isfinite(fill[m])
        if bad.any():
            mode_mean = pv_train[mask_train[:, m], m].mean()
            fill[m, bad] = mode_mean
    print(f"填充向量范围 [{fill.min():.3f}, {fill.max():.3f}] km/s, "
          f"耗时 {time.time() - t0:.0f}s")

    X_all = build_inputs(pv, mask, fill)            # (N,476) 物理 km/s
    Y_all = vs.astype(np.float32)                   # (N,20)

    # 训练折归一化统计
    X_train = X_all[train_sel]
    in_mean = X_train.mean(axis=0).astype(np.float32)
    in_std = X_train.std(axis=0).astype(np.float32)
    in_std[in_std < 1e-8] = 1.0
    t_mean = Y_all[train_sel].mean(axis=0).astype(np.float32)
    t_std = Y_all[train_sel].std(axis=0).astype(np.float32)
    t_std[t_std < 1e-8] = 1.0

    def to_gpu(array, sel):
        return torch.as_tensor(array[sel], device=device)

    Xtr = (to_gpu(X_all, train_sel) - torch.as_tensor(in_mean, device=device)) \
        / torch.as_tensor(in_std, device=device)
    Ytr = (to_gpu(Y_all, train_sel) - torch.as_tensor(t_mean, device=device)) \
        / torch.as_tensor(t_std, device=device)
    Xva = (to_gpu(X_all, val_sel) - torch.as_tensor(in_mean, device=device)) \
        / torch.as_tensor(in_std, device=device)
    Yva = to_gpu(Y_all, val_sel)  # 物理单位，用于汇报 val MAE (km/s)
    print(f"train {len(Xtr)}, val {len(Xva)}, 已驻留 {device}")

    in_mean_t = torch.as_tensor(in_mean, device=device)
    in_std_t = torch.as_tensor(in_std, device=device)
    t_mean_t = torch.as_tensor(t_mean, device=device)
    t_std_t = torch.as_tensor(t_std, device=device)

    histories = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = InverseNet(width=args.width, blocks=args.blocks,
                           dropout=args.dropout).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n===== seed {seed}: {n_params / 1e6:.1f}M 参数 =====")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)
        loss_fn = (nn.MSELoss() if args.loss == "mse"
                   else nn.HuberLoss(delta=1.0))

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        history = []
        n_train = len(Xtr)
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_train, device=device)
            total_loss = 0.0
            for i in range(0, n_train, args.batch_size):
                idx = perm[i:i + args.batch_size]
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    pred = model(Xtr[idx])
                    loss = loss_fn(pred, Ytr[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
            scheduler.step()

            # 验证（物理 km/s MAE + 标准化 MSE）
            model.eval()
            with torch.no_grad():
                val_mse_scaled, val_mae_phys, n_val = 0.0, 0.0, 0
                for i in range(0, len(Xva), args.batch_size):
                    xb = Xva[i:i + args.batch_size]
                    pred = model(xb) * t_std_t + t_mean_t
                    val_mae_phys += (pred - Yva[i:i + len(xb)]).abs() \
                        .sum().item()
                    val_mse_scaled += ((pred - t_mean_t) / t_std_t
                                       - (Yva[i:i + len(xb)] - t_mean_t)
                                       / t_std_t).pow(2).sum().item()
                    n_val += len(xb) * 20
                val_mae_phys /= n_val
                val_mse_scaled /= n_val
            history.append(dict(epoch=epoch,
                                train_loss=total_loss / n_train,
                                val_mse=val_mse_scaled,
                                val_mae_km_s=val_mae_phys,
                                lr=scheduler.get_last_lr()[0]))
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch:3d}: train {total_loss / n_train:.5f}, "
                      f"val MSE {val_mse_scaled:.5f}, "
                      f"val MAE {val_mae_phys * 1000:.2f} m/s")

            if val_mse_scaled < best_val - 1e-6:
                best_val = val_mse_scaled
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"  早停 @ epoch {epoch}")
                    break

        ckpt_path = results_dir / f"inverse_net_seed{seed}_best.pt"
        torch.save({
            "model_state_dict": best_state,
            "config": dict(model.config, batch_size=args.batch_size,
                           learning_rate=args.lr,
                           weight_decay=args.weight_decay, loss=args.loss,
                           seed=seed, dataset="swave production-w64"),
            "input_mean": in_mean, "input_std": in_std,
            "target_mean": t_mean, "target_std": t_std,
            "fill_values": fill.astype(np.float32),
            "val_mse": best_val,
            "val_mae_km_s": min(h["val_mae_km_s"] for h in history),
        }, ckpt_path)
        histories[seed] = history
        print(f"  最佳 val MSE {best_val:.5f}, "
              f"val MAE {min(h['val_mae_km_s'] for h in history) * 1000:.2f} m/s "
              f"-> {ckpt_path}")

    with open(results_dir / "inverse_net_history.json", "w") as f:
        json.dump({str(s): h for s, h in histories.items()}, f)

    # ---------- 测试集预测（物理 km/s） ----------
    print("\n测试集预测 ...")
    models = []
    for seed in args.seeds:
        payload = torch.load(results_dir / f"inverse_net_seed{seed}_best.pt",
                             map_location=device, weights_only=False)
        net = InverseNet(**{k: payload["config"][k] for k in
                            ("input_dim", "output_dim", "width", "blocks",
                             "dropout")}).to(device)
        net.load_state_dict(payload["model_state_dict"])
        net.eval()
        models.append(net)

    Xte = torch.as_tensor(X_all[test_sel], device=device)
    preds = []
    with torch.no_grad():
        for net in models:
            out = []
            for i in range(0, len(Xte), args.batch_size):
                xb = (Xte[i:i + args.batch_size] - in_mean_t) / in_std_t
                out.append(net(xb) * t_std_t + t_mean_t)
            preds.append(torch.cat(out).cpu().numpy())
    pred_mean = np.mean(preds, axis=0)
    np.save(results_dir / "predictions_invnet.npy", pred_mean)
    if len(preds) > 1:
        np.save(results_dir / "invnet_std.npy", np.std(preds, axis=0))
    mae_test = np.abs(pred_mean - Y_all[test_sel]).mean() * 1000
    print(f"saved predictions_invnet.npy ({pred_mean.shape}), "
          f"集成种子数 {len(preds)}, test MAE {mae_test:.2f} m/s")


if __name__ == "__main__":
    main()
