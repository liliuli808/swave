#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把聚类标注的数字字形组装成置信度，输出 ref_confs.json。"""
import json
from pathlib import Path

LABELS = {
    0: '0', 1: '7', 2: '?', 3: '4', 4: '9', 5: '9', 6: '0', 7: '6', 8: '9', 9: '3',
    10: '8', 11: '5', 12: '8', 13: '1', 14: '6', 15: '8', 16: '2', 17: '5', 18: '0', 19: '6',
    20: '8', 21: '5', 22: '0', 23: '6', 24: '8', 25: '6', 26: '7', 27: '9', 28: '0', 29: '2',
    30: '5', 31: '0', 32: '8', 33: '0', 34: '2', 35: '3', 36: '0', 37: '6', 38: '5', 39: '9',
    40: '2', 41: '9', 42: '1', 43: '8', 44: '9', 45: '?', 46: '6', 47: '4',
}

base = Path.home() / "swave/results/viz-compare"
idx = json.loads((base / "glyphs_index.json").read_text())
clusters = json.loads((base / "glyph_clusters.json").read_text())

# 12 个字形右半缺失（簇 2/45）的条目，已从原图人工读出置信度
MANUAL = {
    ("baseline", "air-hole7-000.jpg", 0): 0.70,
    ("baseline", "air-hole7-071.jpg", 1): 0.80,
    ("baseline", "unfused30.jpg", 0): 0.80,
    ("final", "air-hole7-001.jpg", 0): 0.40,
    ("final", "air-hole7-027.jpg", 0): 0.80,
    ("final", "air-hole7-045.jpg", 0): 0.70,
    ("final", "air-hole7-064.jpg", 1): 0.80,
    ("final", "air-hole7-116.jpg", 1): 0.40,
    ("final", "air-hole7-193.jpg", 0): 0.30,
    ("final", "air-hole7-386.jpg", 1): 0.50,
    ("final", "air-hole7-391.jpg", 3): 0.60,
    ("final", "unfused44.jpg", 0): 0.80,
}

confs = {"baseline": {}, "final": {}}
bad = 0
for e in idx:
    if len(e["glyph_ids"]) != 3:
        bad += 1
        continue
    d1, d2, d3 = (LABELS[clusters[g]] for g in e["glyph_ids"])
    if "?" in (d1, d2, d3):
        key = (e["model"], e["image"], e["strip"])
        if key in MANUAL:
            v = MANUAL[key]
        else:
            bad += 1
            continue
    else:
        v = int(d1) + 0.1 * int(d2) + 0.01 * int(d3)
    m = confs[e["model"]].setdefault(e["image"], {})
    m[e["class"]] = max(m.get(e["class"], 0.0), round(v, 2))

(base / "ref_confs.json").write_text(json.dumps(confs, ensure_ascii=False))
print("丢弃(噪声簇):", bad)
print("baseline 覆盖", len(confs["baseline"]), "final 覆盖", len(confs["final"]))
print("air-hole7-000 baseline 应0.70:", confs["baseline"].get("air-hole7-000.jpg"))
print("air-hole7-010 baseline 应0.59:", confs["baseline"].get("air-hole7-010.jpg"))
print("bite-edge2-02 final 应0.80:", confs["final"].get("bite-edge2-02.jpg"))
