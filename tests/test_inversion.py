"""inversion/ 混合反演模块的测试。

遵循 tests/ 现有模式：临时 checkpoint + CPU 小尺度验证。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SWAVE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SWAVE_ROOT / "src"))

from swave.inference import ForwardPredictor  # noqa: E402
from swave.network import FourHeadForwardModel  # noqa: E402


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


forward_torch = _load_module(
    "forward_torch", SWAVE_ROOT / "inversion/forward_torch.py")
inverse_net = _load_module(
    "inverse_net", SWAVE_ROOT / "inversion/inverse_net.py")
run_hybrid = _load_module(
    "run_hybrid", SWAVE_ROOT / "inversion/run_hybrid.py")


def _tiny_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = FourHeadForwardModel()
    payload = {
        "model": model.state_dict(),
        "input_mean": np.full(20, 1.0, dtype=np.float32),
        "input_std": np.full(20, 0.5, dtype=np.float32),
        "target_mean": np.full((4, 1), 1.0, dtype=np.float32),
        "target_std": np.full((4, 1), 0.2, dtype=np.float32),
    }
    path = tmp_path / "forward.pt"
    torch.save(payload, path)
    return path


def _random_vs(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.sort(rng.uniform(0.3, 2.6, size=(n, 20)).astype(np.float32),
                   axis=1)


def test_torch_forward_matches_predictor(tmp_path: Path) -> None:
    """批量正演与 ForwardPredictor 逐点一致。"""
    ckpt = _tiny_checkpoint(tmp_path)
    forward = forward_torch.TorchForward(ckpt, device="cpu")
    predictor = ForwardPredictor.load(ckpt, device="cpu")
    vs = _random_vs(8)

    with torch.no_grad():
        mine = forward.forward(torch.as_tensor(vs)).numpy()
    reference = predictor.predict(vs)
    assert mine.shape == (8, 4, 120)
    np.testing.assert_allclose(mine, reference, atol=1e-5)

    flat = forward.forward_flat(torch.as_tensor(vs))
    assert flat.shape == (8, 476)
    np.testing.assert_allclose(
        flat.numpy(), reference[:, :, 1:].reshape(8, -1), atol=1e-5)


def test_torch_forward_gradient_vs_finite_difference(tmp_path: Path) -> None:
    """float64 下 autograd 梯度与中心差分一致。"""
    ckpt = _tiny_checkpoint(tmp_path)
    forward = forward_torch.TorchForward(ckpt, device="cpu",
                                         dtype=torch.float64)
    vs = torch.as_tensor(_random_vs(2), dtype=torch.float64)
    d_obs = forward.forward_flat(vs).detach() + 0.01
    vs_t = vs.clone().requires_grad_(True)
    misfit = forward.data_misfit(vs_t, d_obs)
    grad_auto = torch.autograd.grad(misfit.sum(), vs_t)[0].detach().numpy()

    eps = 1e-5
    grad_fd = np.zeros_like(grad_auto)
    for i in range(2):
        for j in range(20):
            plus = vs.clone(); plus[i, j] += eps
            minus = vs.clone(); minus[i, j] -= eps
            grad_fd[i, j] = (
                forward.data_misfit(plus, d_obs)[i].item()
                - forward.data_misfit(minus, d_obs)[i].item()) / (2 * eps)
    rel = np.linalg.norm(grad_auto - grad_fd) / np.linalg.norm(grad_fd)
    assert rel < 1e-4


def test_inverse_net_output_shape() -> None:
    model = inverse_net.InverseNet(width=64, blocks=2)
    out = model(torch.randn(7, 476))
    assert out.shape == (7, 20)


def test_predict_physical_fills_nan(tmp_path: Path) -> None:
    """predict_physical 用 fill_values 填充 NaN，不报错且形状正确。"""
    model = inverse_net.InverseNet(width=64, blocks=2)
    norm = {
        "input_mean": np.zeros(476, dtype=np.float32),
        "input_std": np.ones(476, dtype=np.float32),
        "target_mean": np.ones(20, dtype=np.float32),
        "target_std": np.ones(20, dtype=np.float32),
        "fill_values": np.full(476, 0.8, dtype=np.float32),
    }
    d_obs = np.random.default_rng(0).normal(1.0, 0.1, size=(5, 476))
    d_obs[0, :10] = np.nan
    pred = inverse_net.predict_physical(model, norm, d_obs, device="cpu")
    assert pred.shape == (5, 20)
    assert np.all(np.isfinite(pred))


class _LinearForward:
    """玩具正演 F(x) = A x，接口与 TorchForward.data_misfit 相同。"""

    def __init__(self, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.A = torch.randn(64, 20, generator=gen) * 0.05

    def data_misfit(self, vs, d_obs, weights=None):
        residual = vs @ self.A.T - d_obs
        if weights is not None:
            residual = residual * weights
        return 0.5 * residual.pow(2).mean(dim=1)


def test_adam_invert_recovers_toy_problem() -> None:
    """线性玩具问题：残差下降、约束满足、守卫（不差于初值）成立。"""
    forward = _LinearForward()
    x_true = torch.full((6, 20), 1.0)
    x_true += torch.linspace(0, 0.5, 20)
    d_obs = x_true @ forward.A.T
    x0 = torch.full((6, 20), 0.8)

    best_x, final_loss, init_loss, clamped = run_hybrid.adam_invert(
        forward, d_obs, x0, steps=200, lr=5e-2, lambda_smooth=0.0)

    assert best_x.shape == (6, 20)
    assert torch.all(final_loss <= init_loss + 1e-12)
    assert final_loss.mean() < init_loss.mean() * 0.1
    assert torch.all(best_x >= run_hybrid.VS_MIN)
    assert torch.all(best_x <= run_hybrid.VS_MAX)
    err0 = (x0 - x_true).abs().mean()
    err1 = (best_x - x_true).abs().mean()
    assert err1 < err0


def test_adam_invert_never_worse_than_warmstart() -> None:
    """初值即真解时，精修不得使其变差（守卫含第 0 步）。"""
    forward = _LinearForward()
    x_true = torch.full((4, 20), 1.2)
    d_obs = x_true @ forward.A.T

    best_x, final_loss, init_loss, _ = run_hybrid.adam_invert(
        forward, d_obs, x_true.clone(), steps=50, lr=5e-2,
        lambda_smooth=1e-3)
    assert torch.all(final_loss <= init_loss + 1e-12)
