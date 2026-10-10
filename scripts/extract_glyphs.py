#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 baseline/final 可视化图中提取标签条尾部的置信度字形。

流程：类别色填充矩形 -> 极性文字掩码 -> 右侧最大空隙切出置信度尾部
-> 投影切分字形（数字/小数点）-> 归一化保存，供聚类标注。
输出：
  results/viz-compare/glyphs/*.png        字形小图
  results/viz-compare/glyphs_index.json   每个字形的来源信息
"""
import json
from pathlib import Path

import cv2
import numpy as np

TODESK = Path.home() / "Downloads/ToDesk"
VIZ_ROOTS = {
    "baseline": TODESK / "baseline (2)/baseline",
    "final": TODESK / "final (2)/final",
}
OUT_DIR = Path.home() / "swave/results/viz-compare/glyphs"
INDEX = Path.home() / "swave/results/viz-compare/glyphs_index.json"

CLASS_COLORS = {  # BGR
    "air-hole": (255, 42, 4), "bite-edge": (235, 219, 11),
    "broken-arc": (243, 243, 243), "crack": (183, 223, 0),
    "hollow-bead": (104, 31, 17), "overlap": (221, 111, 255),
    "slag-inclusion": (79, 68, 255), "unfused": (0, 237, 204),
}
WHITE_TEXT = {"air-hole", "hollow-bead", "slag-inclusion"}
WHITE_BG = {CLASS_COLORS[c] for c in WHITE_TEXT}  # 深色底 -> 白字
GROUP_TO_CLASS = {
    "exp40": ("air-hole7", "air-hole"), "exp48": ("air-hole7", "air-hole"),
    "exp41": ("bite-edge2", "bite-edge"), "exp49": ("bite-edge2", "bite-edge"),
    "exp42": ("broken-arc2", "broken-arc"), "exp50": ("broken-arc2", "broken-arc"),
    "exp43": ("crack", "crack"), "exp51": ("crack", "crack"),
    "exp44": ("air-hole4(hollow-bead)", "hollow-bead"), "exp52": ("air-hole4(hollow-bead)", "hollow-bead"),
    "exp45": ("overlap", "overlap"), "exp53": ("overlap", "overlap"),
    "exp46": ("slag-inclusion2", "slag-inclusion"), "exp54": ("slag-inclusion2", "slag-inclusion"),
    "exp47": ("unfused", "unfused"), "exp55": ("unfused", "unfused"),
}


def dmap(img, color):
    d = np.abs(img.astype(np.int16) - np.array(color, np.int16)).max(axis=2)
    return d


def find_strips(img, color):
    """找填充标签条，返回 [(x0,y0,x1,y1)]。

    闭运算连成候选块后，在块内找「高占比连续行段」作为标签条
    （标签是实心矩形，行占比接近 1；框线合并进来的行很稀疏）。
    """
    mask = (dmap(img, color) <= 16).astype(np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    H, W = mask.shape
    strips = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 60 or h < 14:
            continue
        sub = closed[y:y + h, x:x + w]
        rfrac = sub.mean(axis=1)
        dense = np.where(rfrac > 0.6)[0]
        if not len(dense):
            continue
        runs = np.split(dense, np.where(np.diff(dense) > 1)[0] + 1)
        for run in runs:
            if len(run) < 14 or len(run) > 48:
                continue
            ry0, ry1 = run[0], run[-1] + 1
            band = sub[ry0:ry1]
            cfrac = band.mean(axis=0)
            dcols = np.where(cfrac > 0.6)[0]
            if not len(dcols):
                continue
            cruns = np.split(dcols, np.where(np.diff(dcols) > 1)[0] + 1)
            cols = max(cruns, key=len)
            cx0, cx1 = cols[0], cols[-1] + 1
            bw, bh = cx1 - cx0, ry1 - ry0
            if not (60 <= bw <= 420 and 2.5 <= bw / bh <= 16):
                continue
            # 文字存在性检验（实心标签有文字，纯框线带没有）
            band_img = img[y + ry0:y + ry1, x + cx0:x + cx1]
            tm = text_mask(band_img, color in WHITE_BG, color)
            tcols = np.where(tm.sum(axis=0) > 0)[0]
            if tm.sum() < 30 or len(tcols) < 10:
                continue
            strips.append((x + cx0, y + ry0, min(x + cx1, W), min(y + ry1, H)))
    return strips


def text_mask(crop, white, bg_color=None):
    """极性文字掩码。白字用高阈值取笔画核心（避免光晕搭桥）；
    深色字用「离文字色比离底色近」判定（细笔画晕染严重）。"""
    if white:
        return (crop.min(axis=2) > 218).astype(np.uint8)
    txt_color = np.array((104, 31, 17), np.int16)
    p = crop.astype(np.int16)
    d_txt = np.abs(p - txt_color).max(axis=2)
    if bg_color is not None:
        d_bg = np.abs(p - np.array(bg_color, np.int16)).max(axis=2)
        return (d_txt < d_bg).astype(np.uint8)
    return (d_txt < 90).astype(np.uint8)


def split_tail(tmask):
    """在右半部分找最宽空隙（类名与置信度之间的空格），切出置信度尾部。"""
    h, w = tmask.shape
    colsum = tmask.sum(axis=0)
    zero = np.where(colsum == 0)[0]
    gaps = []
    if len(zero):
        runs = np.split(zero, np.where(np.diff(zero) > 1)[0] + 1)
        gaps = [(r[0], r[-1] + 1) for r in runs if r[-1] + 1 < w]
    right = [g for g in gaps if g[0] > w * 0.4]
    if not right:
        return None
    g = max(right, key=lambda g: g[1] - g[0])
    if g[1] - g[0] < 4:  # 空格显著宽于字母间隙
        return None
    tail = tmask[:, g[1]:]
    if tail.shape[1] < 15:
        return None
    # 去掉尾部上下空白边
    rows = np.where(tail.sum(axis=1) > 0)[0]
    if not len(rows):
        return None
    return tail[rows[0]:rows[-1] + 1, :], g[1]


def segment_glyphs(tail):
    """把尾部切成字形：返回 [('digit'|'dot', bitmap), ...] 从左到右。"""
    t = cv2.morphologyEx(tail, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(t, connectivity=8)
    H = tail.shape[0]
    items = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 4:
            continue
        bm = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        kind = "dot" if (h <= H * 0.45 and w <= H * 0.45 and y > H * 0.4) else "digit"
        items.append((x, kind, bm))
    items.sort(key=lambda t: t[0])
    # 过宽的组件可能是两个数字粘连，从最弱投影谷劈开
    out = []
    for x, kind, bm in items:
        if kind == "digit" and bm.shape[1] > bm.shape[0] * 1.1 and bm.shape[0] > 8:
            proj = bm.sum(axis=0).astype(float)
            mid = len(proj) // 2
            lo = max(2, mid - int(len(proj) * 0.25))
            hi = min(len(proj) - 2, mid + int(len(proj) * 0.25))
            cut = lo + int(np.argmin(proj[lo:hi]))
            if proj[cut] <= proj.max() * 0.35:
                out.append((x, "digit", bm[:, :cut]))
                out.append((x + cut, "digit", bm[:, cut:]))
                continue
        out.append((x, kind, bm))
    return [(k, b) for _, k, b in out]


def normalize(bm, size=32):
    """保纵横比缩放到 size 见方画布。"""
    h, w = bm.shape
    scale = (size - 6) / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    r = cv2.resize(bm, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    gid = 0
    for model_name, root in VIZ_ROOTS.items():
        for exp_dir in sorted(root.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name not in GROUP_TO_CLASS:
                continue
            group, cls_name = GROUP_TO_CLASS[exp_dir.name]
            color = CLASS_COLORS[cls_name]
            white = cls_name in WHITE_TEXT
            for img_path in sorted(exp_dir.glob("*.jpg")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                for si, (x0, y0, x1, y1) in enumerate(find_strips(img, color)):
                    crop = img[y0:y1, x0:x1]
                    tm = text_mask(crop, white, color)
                    sp = split_tail(tm)
                    if sp is None:
                        continue
                    tail, tail_x = sp
                    glyphs = segment_glyphs(tail)
                    kinds = [k for k, _ in glyphs]
                    digits = [g for k, g in glyphs if k == "digit"]
                    # 0.XX 结构：小数点可能小到丢失，3 或 4 个组件都接受
                    if not (len(digits) == 3 and len(glyphs) in (3, 4)):
                        continue
                    entry = {
                        "model": model_name, "group": group, "class": cls_name,
                        "image": img_path.name, "strip": si,
                        "strip_xy": [int(x0), int(y0), int(x1), int(y1)],
                        "tail_x": int(x0 + tail_x),
                        "kinds": kinds, "glyph_ids": [],
                    }
                    for pos, (kind, bm) in enumerate(glyphs):
                        if kind != "digit":
                            continue  # 只保存数字字形，下游按 个位/十分位/百分位 读取
                        name = f"g{gid:06d}.png"
                        cv2.imwrite(str(OUT_DIR / name), normalize(bm) * 255)
                        entry["glyph_ids"].append(name)
                        gid += 1
                    index.append(entry)
        print(f"{model_name} 完成, 累计 strip {len(index)}, 字形 {gid}", flush=True)
    INDEX.write_text(json.dumps(index, ensure_ascii=False))
    print("保存 ->", INDEX)


if __name__ == "__main__":
    main()
