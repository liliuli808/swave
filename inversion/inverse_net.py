"""反演网络定义：标准化频散 (476,) -> 标准化 Vs (20,)。

InverseNet 学习条件均值 E[Vs|d]：在非唯一的深度区间输出训练分布的
先验均值而非外推，作为混合反演（初值 + Adam 精修）的初值来源。
归一化数组随 checkpoint 保存，推理时物理单位进/出。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

INPUT_DIM = 476   # 4 模态 × 119 频点（1.0:0.5:60 Hz，mode-major）
OUTPUT_DIM = 20   # 20 层 Vs


class ResidualBlock(nn.Module):
    """Pre-LN 残差块: x + Linear(GELU(Linear(LN(x))))。"""

    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        return x + h


class InverseNet(nn.Module):
    """残差 MLP：标准化频散 -> 标准化 Vs。"""

    def __init__(self, input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM,
                 width: int = 1024, blocks: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.config = dict(input_dim=input_dim, output_dim=output_dim,
                           width=width, blocks=blocks, dropout=dropout)
        self.stem = nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
        self.backbone = nn.Sequential(
            *[ResidualBlock(width, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(width),
                                  nn.Linear(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(self.stem(x)))


def load_inverse_net(ckpt_path: Path | str, device: str = "cpu"):
    """加载 checkpoint，返回 (model, norm) ；norm 含输入/输出归一化数组。

    checkpoint 格式：
      model_state_dict, config,
      input_mean/input_std (476,), target_mean/target_std (20,)
    """
    payload = torch.load(Path(ckpt_path), map_location=device,
                         weights_only=False)
    config = payload["config"]
    model = InverseNet(**{k: config[k] for k in
                          ("input_dim", "output_dim", "width", "blocks",
                           "dropout")})
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    norm = {k: np.asarray(payload[k], dtype=np.float32)
            for k in ("input_mean", "input_std", "target_mean", "target_std")}
    if "fill_values" in payload:  # 无效频点填充向量 (476,) 或 (4,119)
        norm["fill_values"] = np.asarray(
            payload["fill_values"], dtype=np.float32).reshape(-1)
    return model, norm


def predict_physical(model: InverseNet, norm: dict, d_obs: np.ndarray,
                     device: str = "cpu", batch_size: int = 8192) -> np.ndarray:
    """物理频散 (N,476) km/s -> 物理 Vs (N,20) km/s。

    含 NaN 的无效频点用 checkpoint 的 fill_values 填充（与训练一致）。
    """
    d_obs = np.asarray(d_obs, dtype=np.float32)
    if "fill_values" in norm:
        nan_cells = ~np.isfinite(d_obs)
        if nan_cells.any():
            d_obs = d_obs.copy()
            d_obs[nan_cells] = np.broadcast_to(
                norm["fill_values"][None], d_obs.shape)[nan_cells]
    mean = torch.as_tensor(norm["input_mean"], device=device)
    std = torch.as_tensor(norm["input_std"], device=device)
    t_mean = torch.as_tensor(norm["target_mean"], device=device)
    t_std = torch.as_tensor(norm["target_std"], device=device)
    outputs = []
    with torch.no_grad():
        for i in range(0, len(d_obs), batch_size):
            batch = torch.as_tensor(
                d_obs[i:i + batch_size], dtype=torch.float32, device=device)
            pred = model((batch - mean) / std) * t_std + t_mean
            outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs)
