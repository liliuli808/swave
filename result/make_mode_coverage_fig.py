# -*- coding: utf-8 -*-
"""生成图1:10 万个模型中各阶模态的存在覆盖率随频率变化。

用法(在 swave 仓库根目录):
    .venv/bin/python result/make_mode_coverage_fig.py
"""
import h5py
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体(DejaVu Sans 无 CJK 字形,需自动探测)
zh = [f.name for f in font_manager.fontManager.ttflist
      if any(k in f.name for k in ['Noto Sans CJK', 'WenQuanYi', 'SimHei', 'Microsoft YaHei'])]
if zh:
    plt.rcParams['font.sans-serif'] = [zh[0], 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

freqs = np.arange(0.5, 60.25, 0.5)
cov = np.zeros((4, 120))
for s in range(10):  # 前 10 个分片 = 10 万样本
    with h5py.File(f'data/production/shard-{s:05d}.h5', 'r') as f:
        cov += f['valid_mask'][:].sum(axis=0)
cov /= 100000

fig, ax = plt.subplots(figsize=(9, 5))
labels = ['M0(基阶)', 'M1(一阶)', 'M2(二阶)', 'M3(三阶)']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for m in range(4):
    ax.plot(freqs, cov[m] * 100, color=colors[m], lw=1.8, label=labels[m])
ax.set_xlim(0, 15)
ax.set_ylim(0, 105)
ax.set_xlabel('频率 (Hz)')
ax.set_ylabel('该模态存在的模型比例 (%)')
ax.set_title('10 万个模型中各阶模态的存在覆盖率随频率变化')
ax.grid(alpha=0.3)
ax.legend(loc='lower right')
ax.axvspan(0.5, 1.5, color='gray', alpha=0.12)
ax.annotate('截止频率区:\n高阶模态仅在此处缺失', xy=(1.0, 32), ha='center', fontsize=10)
fig.tight_layout()
fig.savefig('result/figures/mode-coverage-vs-frequency.png', dpi=200)
print('saved result/figures/mode-coverage-vs-frequency.png')
