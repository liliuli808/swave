#!/usr/bin/env python3
"""可微分批量正演算子（物理单位，纯 torch）。

包装 runs/production-48g/best.pt 的四头正演代理，供反演使用：
- 输入/输出均为物理单位（km/s），归一化参数直接取自 checkpoint，
  不依赖 iNETT 数据准备的 sklearn scaler；
- 原生批量：(N,20) -> (N,4,120)，支持 autograd；
- forward_flat 展平为 (N,476)（去 0.5 Hz 列，mode-major），与
  iNETT 数据布局一致，便于与既有结果对比；
- data_misfit 返回逐样本数据残差 0.5*mean((F-d)^2 * w)，恒带梯度图。

直接运行本文件执行自检：
1) 与 swave.inference.ForwardPredictor 的批量输出一致性（<1e-5 km/s）；
2) autograd 梯度 vs 中心有限差分（相对误差 <1e-3）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SWAVE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SWAVE_ROOT / "src"))

from swave.network import FourHeadForwardModel  # noqa: E402

DEFAULT_CHECKPOINT = SWAVE_ROOT / "runs/production-48g/best.pt"
DROP_FREQ_COLS = 1  # 丢 0.5 Hz 列，对齐 iNETT 的 1.0:0.5:60 网格
N_MODES = 4
N_FREQ_OUT = 120 - DROP_FREQ_COLS  # 119
N_DATA = N_MODES * N_FREQ_OUT  # 476
VS_MIN, VS_MAX = 0.3, 2.6  # NN 定义域（km/s）


class TorchForward:
    """物理单位的可微批量正演：vs_km (N,20) -> 频散 (N,4,120) km/s。

    dtype 默认 float32（生产精度）；自检等需要高精度梯度的场景
    可传 torch.float64。
    """

    def __init__(self, checkpoint: Path | str = DEFAULT_CHECKPOINT,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        payload = torch.load(Path(checkpoint), map_location=self.device,
                             weights_only=False)
        model = FourHeadForwardModel()
        model.load_state_dict(payload["model"])
        model.to(device=self.device, dtype=dtype)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model
        self.in_mean = torch.as_tensor(
            np.asarray(payload["input_mean"], dtype=np.float32),
            device=self.device, dtype=dtype)
        self.in_std = torch.as_tensor(
            np.asarray(payload["input_std"], dtype=np.float32),
            device=self.device, dtype=dtype)
        # (4,1) -> (1,4,1) 便于广播
        self.tgt_mean = torch.as_tensor(
            np.asarray(payload["target_mean"], dtype=np.float32),
            device=self.device, dtype=dtype).unsqueeze(0)
        self.tgt_std = torch.as_tensor(
            np.asarray(payload["target_std"], dtype=np.float32),
            device=self.device, dtype=dtype).unsqueeze(0)

    def forward(self, vs_km: torch.Tensor) -> torch.Tensor:
        """vs_km (N,20) km/s -> (N,4,120) km/s，保留梯度图。"""
        normalized = (vs_km - self.in_mean) / self.in_std
        prediction = self.model(normalized)
        return prediction * self.tgt_std + self.tgt_mean

    def forward_flat(self, vs_km: torch.Tensor) -> torch.Tensor:
        """vs_km (N,20) -> (N,476)：去 0.5 Hz 列后 mode-major 展平。"""
        curves = self.forward(vs_km)
        return curves[:, :, DROP_FREQ_COLS:].reshape(vs_km.shape[0], -1)

    def data_misfit(self, vs_km: torch.Tensor, d_obs: torch.Tensor,
                    weights: torch.Tensor | None = None) -> torch.Tensor:
        """逐样本数据残差 0.5*mean((F(vs)-d)^2 * w) -> (N,)，恒带梯度图。

        d_obs: (N,476) 物理 km/s（与 forward_flat 同布局）。
        weights: (476,) 可选频率权重（广播）。
        """
        residual = self.forward_flat(vs_km) - d_obs
        if weights is not None:
            residual = residual * weights
        return 0.5 * residual.pow(2).mean(dim=1)


def _self_check() -> None:
    from swave.inference import ForwardPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_forward = TorchForward(device=device)
    predictor = ForwardPredictor.load(DEFAULT_CHECKPOINT, device=device)

    rng = np.random.default_rng(20260729)
    # 光滑递增的 Vs 剖面（符合数据集形态）
    vs = np.sort(rng.uniform(0.3, 2.6, size=(32, 20)).astype(np.float32),
                 axis=1)

    # 1) 与 ForwardPredictor 批量输出一致性
    reference = predictor.predict(vs)  # (32,4,120)
    with torch.no_grad():
        mine = torch_forward.forward(
            torch.as_tensor(vs, device=torch_forward.device)
        ).cpu().numpy()
    max_diff = float(np.max(np.abs(mine - reference)))
    print(f"[1] 与 ForwardPredictor 最大差: {max_diff:.3e} km/s")
    assert max_diff < 1e-5, "批量正演与 ForwardPredictor 不一致"

    # forward_flat 布局检查
    flat = torch_forward.forward_flat(
        torch.as_tensor(vs, device=torch_forward.device))
    assert flat.shape == (32, N_DATA)
    ref_flat = reference[:, :, DROP_FREQ_COLS:].reshape(32, -1)
    assert float((flat.cpu().numpy() - ref_flat).__abs__().max()) < 1e-5

    # 2) autograd 梯度 vs 中心有限差分（float64 实例，排除精度干扰）
    check = TorchForward(device=device, dtype=torch.float64)
    n_check = 4
    offset = rng.normal(0.0, 0.05, size=(n_check, N_DATA))
    vs64 = torch.as_tensor(vs[:n_check], dtype=torch.float64,
                           device=check.device)
    vs_t = vs64.clone().requires_grad_(True)
    d_obs = check.forward_flat(vs_t).detach()
    offset_t = torch.as_tensor(offset, dtype=torch.float64,
                               device=check.device)
    misfit = check.data_misfit(vs_t, d_obs + offset_t)
    grad_auto = torch.autograd.grad(misfit.sum(), vs_t)[0].detach().cpu().numpy()

    eps = 1e-4
    grad_fd = np.zeros_like(grad_auto)
    for i in range(n_check):
        for j in range(20):
            v_plus = vs64.clone(); v_plus[i, j] += eps
            v_minus = vs64.clone(); v_minus[i, j] -= eps
            f_plus = check.data_misfit(v_plus, d_obs + offset_t)[i].item()
            f_minus = check.data_misfit(v_minus, d_obs + offset_t)[i].item()
            grad_fd[i, j] = (f_plus - f_minus) / (2 * eps)
    rel_err = float(np.linalg.norm(grad_auto - grad_fd)
                    / (np.linalg.norm(grad_fd) + 1e-12))
    print(f"[2] autograd vs 有限差分 范数相对误差: {rel_err:.3e}")
    assert rel_err < 1e-4, "批量梯度与有限差分不符"
    print("自检通过")


if __name__ == "__main__":
    _self_check()
