#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 5 个已训练模型在 SteelPipeWeld 测试集上跑预测并保存可视化。

可视化风格与 baseline/final 参考图一致：ultralytics 默认 Annotator
（按类别索引取默认调色板颜色，标签为「类别名 置信度」，位于框左上角上方）。
输出按缺陷组归档到子文件夹，命名与参考文件夹一致。
"""
import re
import shutil
from pathlib import Path

from ultralytics import YOLO, RTDETR

RUNS = Path.home() / "Documents/SteelPipeWeld-runs"
TEST_IMAGES = Path.home() / "datasets/SteelPipeWeld-yolo/test/images"
OUT_ROOT = Path.home() / "Downloads/ToDesk/其他模型预测"

MODELS = {
    "yolo26n": (YOLO, RUNS / "yolo26n-30e/weights/best.pt"),
    "RT-DETR-L": (RTDETR, RUNS / "rtdetr-l-30e-lr1e4/weights/best.pt"),
    "yolov9t": (YOLO, RUNS / "yolov9t-30e/weights/best.pt"),
    "yolov10n": (YOLO, RUNS / "yolov10n-30e/weights/best.pt"),
    "yolov5s": (YOLO, RUNS / "yolov5s-30e/weights/best.pt"),
}


def group_of(name: str) -> str:
    """把测试集文件名映射到参考文件夹的分组名。"""
    stem = Path(name).stem
    if re.fullmatch(r"crack\d+", stem):
        return "crack"
    if re.fullmatch(r"overlap\d+", stem):
        return "overlap"
    if re.fullmatch(r"unfused\d+", stem):
        return "unfused"
    # air-hole7-000 / air-hole4(hollow-bead)-000 / bite-edge2-000 / broken-arc2-000 / slag-inclusion2-000
    return re.sub(r"-\d+$", "", stem)


def main():
    for model_name, (cls, weights) in MODELS.items():
        assert weights.exists(), f"缺少权重: {weights}"
        print(f"===== {model_name} =====", flush=True)
        model = cls(str(weights))
        tmp = OUT_ROOT / model_name / "_flat"
        # 默认 conf=0.25 / iou=0.7 / imgsz=640，默认 Annotator 画风
        model.predict(
            source=str(TEST_IMAGES),
            save=True,
            project=str(tmp.parent),
            name=tmp.name,
            exist_ok=True,
            verbose=False,
        )
        # 按缺陷组归档
        n = 0
        for img in tmp.glob("*.jpg"):
            gdir = OUT_ROOT / model_name / group_of(img.name)
            gdir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img), gdir / img.name)
            n += 1
        shutil.rmtree(tmp)
        print(f"{model_name}: {n} 张图已归档", flush=True)

    print("全部完成 ->", OUT_ROOT)


if __name__ == "__main__":
    main()
