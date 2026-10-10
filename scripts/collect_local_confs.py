#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""收集 5 个本地模型在 SteelPipeWeld 测试集上每张图、每个类别的最大置信度。

输出 ~/swave/results/viz-compare/local_confs.json:
  {model_name: {image_name: {class_name: max_conf}}}
"""
import json
from pathlib import Path

from ultralytics import YOLO, RTDETR

RUNS = Path.home() / "Documents/SteelPipeWeld-runs"
TEST_IMAGES = Path.home() / "datasets/SteelPipeWeld-yolo/test/images"
OUT = Path.home() / "swave/results/viz-compare/local_confs.json"

MODELS = {
    "yolo26n": (YOLO, RUNS / "yolo26n-30e/weights/best.pt"),
    "RT-DETR-L": (RTDETR, RUNS / "rtdetr-l-30e-lr1e4/weights/best.pt"),
    "yolov9t": (YOLO, RUNS / "yolov9t-30e/weights/best.pt"),
    "yolov10n": (YOLO, RUNS / "yolov10n-30e/weights/best.pt"),
    "yolov5s": (YOLO, RUNS / "yolov5s-30e/weights/best.pt"),
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_confs = {}
    for model_name, (cls, weights) in MODELS.items():
        print(f"===== {model_name} =====", flush=True)
        model = cls(str(weights))
        confs = {}
        for r in model.predict(source=str(TEST_IMAGES), verbose=False, stream=True):
            per_cls = {}
            if r.boxes is not None and len(r.boxes):
                for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    name = r.names[int(c)]
                    per_cls[name] = max(per_cls.get(name, 0.0), cf)
            confs[Path(r.path).name] = per_cls
        all_confs[model_name] = confs
        print(f"{model_name}: {len(confs)} 张", flush=True)
    OUT.write_text(json.dumps(all_confs, ensure_ascii=False))
    print("保存 ->", OUT)


if __name__ == "__main__":
    main()
