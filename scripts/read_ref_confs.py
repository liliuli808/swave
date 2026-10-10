#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 ultralytics 风格的预测可视化图里逆向读出指定类别的置信度。

原理：标签是「类别色填充矩形 + 白/黑文字」，按 ultralytics Annotator 的
cv2 分支参数 (sf=lw/3, tf=lw-1, h+=3, 基线距底 2px) 把所有候选置信度
(0.25~1.00) 重渲染成模板，与图中色块逐一比对取 MAD 最小者。

先用本地 5 模型的可视化图（真实置信度已知）校准正确率，
再读 baseline/final，输出 ref_confs.json（结构同 local_confs.json）。
"""
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, Colors

TODESK = Path.home() / "Downloads/ToDesk"
VIZ_ROOTS = {
    "baseline": TODESK / "baseline (2)/baseline",
    "final": TODESK / "final (2)/final",
}
LOCAL_VIZ_ROOT = TODESK / "其他模型预测"
LOCAL_CONFS = Path.home() / "swave/results/viz-compare/local_confs.json"
OUT = Path.home() / "swave/results/viz-compare/ref_confs.json"

CLASS_NAMES = ["air-hole", "bite-edge", "broken-arc", "crack",
               "hollow-bead", "overlap", "slag-inclusion", "unfused"]
GROUP_TO_CLASS = {
    "air-hole7": "air-hole",
    "bite-edge2": "bite-edge",
    "broken-arc2": "broken-arc",
    "crack": "crack",
    "air-hole4(hollow-bead)": "hollow-bead",
    "overlap": "overlap",
    "slag-inclusion2": "slag-inclusion",
    "unfused": "unfused",
}
CONF_CANDIDATES = [f"{v / 100:.2f}" for v in range(25, 100)] + ["1.00"]
COLOR_TOL = 24  # JPEG 压缩容忍度


def annotator_params(shape):
    """按 Annotator.__init__ 的 cv2 分支复算 lw/sf/tf。"""
    lw = max(round(sum(shape) / 2 * 0.003), 2)
    return lw, lw / 3, max(lw - 1, 1)


@lru_cache(maxsize=512)
def render_templates(cls_name, color, sf_x3, tf, txt_color):
    """渲染该类全部候选置信度的标签条模板；sf_x3 = round(sf*3) 以便缓存。

    返回 (templates: list[np.ndarray], union_mask: np.ndarray, size: (w, h))
    """
    sf = sf_x3 / 3
    label0 = f"{cls_name} 0.00"
    (w, h), _ = cv2.getTextSize(label0, 0, fontScale=sf, thickness=tf)
    h += 3  # Annotator: h += 3
    bg = np.full((h, w, 3), color, dtype=np.uint8)
    temps = []
    for conf in CONF_CANDIDATES:
        im = bg.copy()
        cv2.putText(im, f"{cls_name} {conf}", (0, h - 2), 0, sf, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
        temps.append(im)
    union = np.zeros((h, w), dtype=bool)
    for t in temps:
        union |= np.any(t.astype(np.int16) != bg.astype(np.int16), axis=2)
    return temps, union, (w, h)


def find_label_strips(img, color):
    """在图中找该类颜色的填充标签条，返回 [(x0, y0, x1, y1), ...]（右下开区间）。

    横向膨胀把一行上的文字间隙与色块连成候选区域，再用行/列占比精修出
    填充矩形的精确边界（文字空洞最多覆盖行内一半像素，框线则稀疏得多）。
    """
    diff = np.abs(img.astype(np.int16) - np.array(color, dtype=np.int16)).max(axis=2)
    mask = (diff <= COLOR_TOL).astype(np.uint8)
    dil = cv2.dilate(mask, np.ones((3, 40), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    strips = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < 60 or h < 12:
            continue
        sub = mask[y:y + h, x:x + w]
        row_ok = np.where(sub.mean(axis=1) > 0.4)[0]
        if not len(row_ok):
            continue
        # 最长的连续高占比行段 = 标签条纵向范围
        runs = np.split(row_ok, np.where(np.diff(row_ok) > 1)[0] + 1)
        rows = max(runs, key=len)
        if len(rows) < 10:
            continue
        ry0, ry1 = rows[0], rows[-1] + 1
        col_ok = np.where(sub[ry0:ry1].mean(axis=0) > 0.25)[0]
        if not len(col_ok):
            continue
        cruns = np.split(col_ok, np.where(np.diff(col_ok) > 1)[0] + 1)
        cols = max(cruns, key=len)
        if len(cols) < 30:
            continue
        strips.append((x + cols[0], y + ry0, x + cols[-1] + 1, y + ry1))
    return strips


def match_strip(strip, cls_name, color, sf, tf, txt_color):
    """对一个标签条做模板匹配，返回 (conf_str, mad) 或 None。"""
    temps, union, (w, h) = render_templates(
        cls_name, tuple(int(v) for v in color), round(sf * 3), tf, txt_color)
    sh, sw = strip.shape[:2]
    if abs(sh - h) > 2 or abs(sw - w) > 2:
        return None
    best, best_mad = None, 1e9
    s16 = strip.astype(np.int16)
    for conf, t in zip(CONF_CANDIDATES, temps):
        t16 = t.astype(np.int16)
        # 允许 ±2px 对齐误差
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ys0, ys1 = max(0, dy), min(sh, sh + dy)
                xs0, xs1 = max(0, dx), min(sw, sw + dx)
                yt0, yt1 = max(0, -dy), min(h, h - dy)
                xt0, xt1 = max(0, -dx), min(w, w - dx)
                if ys1 - ys0 < h - 4 or xs1 - xs0 < w - 4:
                    continue
                m = union[yt0:yt1, xt0:xt1]
                if not m.any():
                    continue
                mad = np.abs(s16[ys0:ys1, xs0:xs1][m] - t16[yt0:yt1, xt0:xt1][m]).mean()
                if mad < best_mad:
                    best, best_mad = conf, mad
    return best, best_mad


def read_image_confs(img_path):
    """读出图中全部 8 类标签的置信度，返回 {class_name: max_conf}。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return {}
    lw, sf, tf = annotator_params(img.shape)
    ann = Annotator(np.zeros((4, 4, 3), np.uint8))
    out = {}
    colors = Colors()
    for ci, cls_name in enumerate(CLASS_NAMES):
        color = tuple(int(v) for v in colors(ci, bgr=True))
        strips = find_label_strips(img, color)
        if not strips:
            continue
        txt_color = tuple(int(v) for v in ann.get_txt_color(color))
        for (x0, y0, x1, y1) in strips:
            strip = img[y0:y1, x0:x1]
            r = match_strip(strip, cls_name, color, sf, tf, txt_color)
            if r is None:
                continue
            conf, mad = r
            if mad > 25:  # 匹配太差，丢弃
                continue
            out[cls_name] = max(out.get(cls_name, 0.0), float(conf))
    return out


def calibrate():
    """用本地模型的图（真实置信度已知）验证读数正确率。"""
    local = json.loads(LOCAL_CONFS.read_text())
    n_tot = n_ok = 0
    per_model = {}
    for model_name in ["yolo26n", "yolov9t", "yolov10n", "yolov5s", "RT-DETR-L"]:
        ok = tot = 0
        for group, cls_name in GROUP_TO_CLASS.items():
            gdir = LOCAL_VIZ_ROOT / model_name / group
            for img_path in sorted(gdir.glob("*.jpg")):
                truth = local[model_name].get(img_path.name, {}).get(cls_name)
                if truth is None:
                    continue
                got = read_image_confs(img_path).get(cls_name)
                tot += 1
                if got is not None and abs(got - truth) <= 0.011:
                    ok += 1
        per_model[model_name] = f"{ok}/{tot}"
        n_tot += tot
        n_ok += ok
        print(f"校准 {model_name}: {ok}/{tot} = {ok / max(tot, 1):.1%}", flush=True)
    print(f"校准总计: {n_ok}/{n_tot} = {n_ok / max(n_tot, 1):.1%}")


def read_refs():
    all_confs = {}
    for model_name, root in VIZ_ROOTS.items():
        confs = {}
        for exp_dir in sorted(root.iterdir()):
            if not exp_dir.is_dir():
                continue
            for img_path in sorted(exp_dir.glob("*.jpg")):
                confs[img_path.name] = read_image_confs(img_path)
            print(f"{model_name}/{exp_dir.name} 完成", flush=True)
        all_confs[model_name] = confs
        print(f"{model_name}: {len(confs)} 张", flush=True)
    OUT.write_text(json.dumps(all_confs, ensure_ascii=False))
    print("保存 ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "read"])
    args = ap.parse_args()
    if args.mode == "calibrate":
        calibrate()
    else:
        read_refs()
