#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""8 个类别各选 1 张图：final 置信度在 7 模型中排名前 3 且 > baseline。

合并 ref_confs.json（baseline/final，图上读回）与 local_confs.json
（5 个本地模型，精确值），缺检测记 0。rank = 1 + 严格大于 final 的模型数。
候选按 (rank 升序, final 置信度降序, baseline 有检测优先) 排序取第一。
"""
import json
from pathlib import Path

base = Path.home() / "swave/results/viz-compare"
ref = json.loads((base / "ref_confs.json").read_text())
local = json.loads((base / "local_confs.json").read_text())

MODELS = ["baseline", "final", "yolo26n", "RT-DETR-L", "yolov9t", "yolov10n", "yolov5s"]
CLASS_GROUP = {
    "air-hole": "air-hole7", "bite-edge": "bite-edge2", "broken-arc": "broken-arc2",
    "crack": "crack", "hollow-bead": "air-hole4(hollow-bead)", "overlap": "overlap",
    "slag-inclusion": "slag-inclusion2", "unfused": "unfused",
}
GROUP_EXP = {
    "baseline": {"air-hole7": "exp40", "bite-edge2": "exp41", "broken-arc2": "exp42", "crack": "exp43",
                 "air-hole4(hollow-bead)": "exp44", "overlap": "exp45", "slag-inclusion2": "exp46", "unfused": "exp47"},
    "final": {"air-hole7": "exp48", "bite-edge2": "exp49", "broken-arc2": "exp50", "crack": "exp51",
              "air-hole4(hollow-bead)": "exp52", "overlap": "exp53", "slag-inclusion2": "exp54", "unfused": "exp55"},
}
TODESK = Path.home() / "Downloads/ToDesk"

def conf_of(model, img, cls):
    if model in ref:
        return ref[model].get(img, {}).get(cls, 0.0)
    return local[model].get(img, {}).get(cls, 0.0)

selection = {}
for cls, group in CLASS_GROUP.items():
    # 候选图 = final 对应 exp 文件夹里的全部图
    imgs = sorted(p.name for p in (TODESK / "final (2)/final" / GROUP_EXP["final"][group]).glob("*.jpg"))
    cands = []
    for img in imgs:
        fc = conf_of("final", img, cls)
        if fc <= 0:
            continue
        bc = conf_of("baseline", img, cls)
        if fc <= bc:
            continue
        others = [conf_of(m, img, cls) for m in MODELS if m != "final"]
        rank = 1 + sum(1 for c in others if c > fc)
        if rank > 3:
            continue
        cands.append((rank, -fc, -bc, img, fc, bc,
                      {m: round(conf_of(m, img, cls), 2) for m in MODELS}))
    cands.sort()
    print(f"== {cls}（{group}）候选 {len(cands)} 张 ==")
    for c in cands[:5]:
        print(f"  rank{c[0]} final={c[4]:.2f} baseline={c[5]:.2f} {c[3]}  {c[6]}")
    if not cands:
        print("  !! 无满足条件的候选")
        continue
    selection[cls] = {"image": cands[0][3], "group": group, "final_conf": cands[0][4],
                      "baseline_conf": cands[0][5], "rank": cands[0][0], "confs": cands[0][6]}

(base / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2))
print("\n选中:", json.dumps({k: v["image"] for k, v in selection.items()}, ensure_ascii=False, indent=2))
