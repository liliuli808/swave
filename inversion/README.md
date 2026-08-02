# 混合反演（方法 C）：监督反演网络 + 批量 Adam 精修

基于 swave 可微正演代理的面波 Vs 反演，独立于 iNETT 管线构建。
全部在 swave `.venv` 中运行，物理单位，纯 torch。

## 结论先行（测试折前 2000 样本，与 iNETT 同子集）

| 方法 | MAE (m/s) | RMSE (m/s) | R² |
|---|---|---|---|
| ICNN 校正（iNETT 热启动） | 75.6 | 113.8 | 0.740 |
| iNETT（李老师方法 + swave 正演） | 63.5 | 105.1 | 0.795 |
| **InvNet（3 种子集成反演网络）** | **26.0** | **48.5** | **0.962** |
| Hybrid（InvNet + Adam 精修） | 26.0 | 48.6 | 0.962 |
| Adam 精修（ICNN 初值，对照） | 74.2 | 112.5 | 0.748 |

分族 MAE（m/s）：InvNet NORMAL 18.7 / LOW_VELOCITY 29.5 /
HIGH_VELOCITY 28.9 / COUPLED 28.0；iNETT 42.0 / 67.7 / 60.6 / 73.4。
InvNet 在所有族上一致地好 2.3–2.6 倍，且族间差异小得多
（对异常模型无系统性变差）。单种子 val MAE 26.74 → 3 种子集成 25.82 m/s。

**两个核心发现：**

1. **监督反演网络远超迭代反演**：直接从 90 万样本学习 频散→Vs 映射
   （条件均值 E[Vs|d]），MAE 26.9 m/s，比 iNETT 的 63.5 m/s 好 2.4 倍。
   原因：反问题非唯一性强（深层 Vs 对频散几乎不可见），迭代法靠数据
   拟合在这些方向上没有约束，而网络学到的先验分布恰好填补了信息缺口。

2. **数据拟合精修在强初值下无效**：从 InvNet 初值（26.9 m/s）出发，
   Adam 精修在任何超参配置下都不能改善 Vs 精度（最好持平）；
   从均值初值出发，数据残差可降 67 倍但 Vs 误差反而增大——
   残差减少对应的是零空间（null space）内的移动，不是向真解收敛。
   这不是优化器问题（玩具线性问题验证通过），而是问题本身的性质。
   注意 iNETT 从 ICNN 初值能改善 16%（63.5 vs 75.6），靠的是其
   学习型 ICNN 正则项，而非纯数据拟合。

## 文件说明

| 文件 | 作用 |
|---|---|
| `forward_torch.py` | 可微批量正演算子（物理单位）。`TorchForward.forward/forward_flat/data_misfit`；`python forward_torch.py` 自检（与 ForwardPredictor 一致性 + float64 梯度校验） |
| `inverse_net.py` | `InverseNet`：残差 MLP（476→1024×4块→20，约 8.9M 参数）；`load_inverse_net` / `predict_physical`（自动用 fill_values 填充无效频点） |
| `freq_weights.py` | 由训练折 valid_mask 有效率 × 逐模态代理噪声生成 (476,) 频率权重 |
| `train_inverse_net.py` | 训练反演网络（全量驻留 GPU，batch 8192，AdamW+Cosine，bf16，早停）；支持 `--seeds` 多种子集成；训练后自动预测测试集 |
| `run_hybrid.py` | 混合反演主流程：`adam_invert`（投影 Adam + 平滑正则 + μ·‖x−x0‖² 信赖域 + 逐样本历史最优守卫）+ `--evaluate` 同子集对比 |
| `ensemble_predict.py` | 多种子集成预测：val 上对比单种子 vs 集成，输出 predictions_invnet.npy 与 invnet_std.npy |
| `results/` | checkpoint、预测、指标、图 |

## 复现

```bash
# 1. 频率权重
.venv/bin/python inversion/freq_weights.py

# 2. 训练反演网络（~35 分钟/种子，单 A6000）
.venv/bin/python inversion/train_inverse_net.py --seeds 0 1 2 --epochs 150

# 2b. 集成预测（val 验证集成收益后采用）
.venv/bin/python inversion/ensemble_predict.py

# 3. 混合反演（测试折前 2000 个，几秒）
.venv/bin/python inversion/run_hybrid.py --warmstart invnet --name hybrid \
    --steps 200 --lr 5e-3 --lambda-smooth 1e-3 --mu-prox 1e-1

# 4. 评估对比（含 iNETT 基线，只读其结果文件）
.venv/bin/python inversion/run_hybrid.py --evaluate --samples 2000 \
    --name invnet,hybrid,adam
```

## 设计要点

- **折间卫生**：`sample_id % 100` 分折（<90 训练 / 90–94 验证 / ≥95 测试），
  与正演代理、iNETT 完全一致；所有超参（lr、λ、μ、steps）只在 val 折调，
  test 每种最终配置只碰一次。
- **守卫**：`adam_invert` 逐样本跟踪含初值在内的历史最优，结果在损失意义下
  绝不差于初值（玩具问题与正式运行均审计通过）。
- **信赖域**：μ·‖x−x0‖² 防止精修沿非唯一方向远离含先验的初值；
  val 上调得 μ=0.1、λ=1e-3、lr=5e-3、steps=200。
- **频率权重**：插值频点与高噪声模态降权（`results/freq_weights.npy`）。

## 验证（回应"过拟合或数据泄漏"质疑）

1. **训练折 vs 测试折**：MAE 25.19 vs 25.94 m/s（差距 3%）——无过拟合特征；
2. **独立物理引擎验证**（`verify_physics.py`）：预测 Vs 经 Brocher(2005)
   生成 Vp/ρ 后送入物理求解器（Dunkin + Pan 模态恢复，全程无 NN），
   与观测频散残差仅 **0.9 m/s**（真值自洽性 0.0 m/s，训练均值 94.2 m/s）
   ——预测在物理上自洽，无法靠"作弊"通过；
3. **误差形态**：RMSE/MAE=1.89（重尾），逐层误差随深度增长（1→47 m/s，
   与敏感核衰减镜像），最差样本 297 m/s——记忆拟合应呈均匀近零误差。

## 局限性

- **深层非唯一性是物理极限**：13–20 层 Vs 对频散几乎不可见，
  InvNet 在这些层输出的是先验均值；逐层指标见 metrics_comparison.json。
- **正演代理自身误差**（M0–M3: 0.08–0.26 m/s）是任何基于它的反演的
  数据拟合下限；InvNet 不经过代理（直接学数据），不受此限。
- **集成不确定性**：多种子标准差（invnet_std.npy）只是种子方差，
  不是校准的后验不确定性；严格 UQ 需要贝叶斯方法（MCMC 等）。

## 与 iNETT 仓库的关系

唯一接口是只读其 `results/swave_inversion/` 下的
`true_vs.npy / predictions_icnn.npy / predictions_inett.npy` 作同子集对比
（两边同为 sample_id%100 分折、shard 顺序拼接，前 2000 测试样本天然对齐）。
反演网络与精修代码完全独立，不依赖 iNETT 的任何数据准备。
