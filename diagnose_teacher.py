#!/usr/bin/env python3
"""诊断: teacher(Stage1 best.pt) 在 unlabeled 图上的预测 —— 是"没检出"还是"低置信度"?

用极低 conf=0.001 推理, 统计每张图的检测框数与置信度分布, 并可视化样本。
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # 避开 GPU 0 的 vLLM

ROOT = Path("/home/mich/project/blade-clearance")
TEACHER = ROOT / "train-seg-draft-0916/training/stage1_auto_labeled_scratch/weights/best.pt"
UNLABELED = ROOT / "unlabeled-0915-640"
OUT = ROOT / "train-seg-draft-0916/diagnosis"
OUT.mkdir(parents=True, exist_ok=True)

CONF_SWEEP = [0.001, 0.05, 0.10, 0.25, 0.50, 0.75]

teacher = YOLO(str(TEACHER))
imgs = sorted(UNLABELED.rglob("*.jpg"))
print(f"teacher: {TEACHER}")
print(f"unlabeled images: {len(imgs)}")

# ── 统计: 极低 conf 下每张图的检测数 ──
results = teacher.predict(
    source=[str(p) for p in imgs],
    imgsz=640, conf=0.001, iou=0.7, device=1,
    stream=True, verbose=False,
)

rows = []          # (path, n_boxes, max_conf, mean_conf)
all_confs = []     # 所有检测框的置信度
for r in results:
    n = len(r.boxes) if r.boxes is not None else 0
    if n > 0:
        c = r.boxes.conf.detach().cpu().numpy().astype(float)
        all_confs.extend(c.tolist())
        rows.append((r.path, n, float(c.max()), float(c.mean())))
    else:
        rows.append((r.path, 0, 0.0, 0.0))

n_total = len(rows)
n_det = sum(1 for r in rows if r[1] > 0)
print(f"\n=== 极低 conf=0.001 下 ===")
print(f"总图数: {n_total}")
print(f"有任何检测框的图数: {n_det}  ({100*n_det/n_total:.1f}%)")
print(f"检测框总数: {len(all_confs)}")

if all_confs:
    a = np.array(all_confs)
    print(f"\n置信度分布 (n={len(a)}):")
    for lo, hi in [(0,0.05),(0.05,0.1),(0.1,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.0)]:
        cnt = int(((a >= lo) & (a < hi)).sum())
        print(f"  conf [{lo:.2f},{hi:.2f}): {cnt}")
    print(f"  conf >= 0.75: {int((a >= 0.75).sum())}")
    print(f"  max_conf overall = {a.max():.4f}, mean = {a.mean():.4f}")

print("\n=== 每张图明细 (按 max_conf 降序) ===")
for path, n, mx, mn in sorted(rows, key=lambda x: -x[2]):
    name = Path(path).name
    print(f"  {name:20s}  n={n:2d}  max_conf={mx:.4f}  mean_conf={mn:.4f}")

# ── 各 conf 阈值下"有检出的图数" ──
print("\n=== 不同 conf 阈值下, 有检出的图数 ===")
for conf in CONF_SWEEP:
    res = teacher.predict(
        source=[str(p) for p in imgs], imgsz=640, conf=conf, iou=0.7,
        device=1, stream=True, verbose=False,
    )
    k = sum(1 for r in res if r.boxes is not None and len(r.boxes) > 0)
    print(f"  conf={conf:<5} -> {k}/{n_total} 张有检测")

# ── 可视化: 挑 max_conf 最高的 6 张 + 无检出的 2 张 ──
print("\n=== 可视化 ===")
det_rows = [r for r in rows if r[1] > 0]
no_rows = [r for r in rows if r[1] == 0]
pick = sorted(det_rows, key=lambda x: -x[2])[:6]
pick += no_rows[:2]
for path, n, mx, mn in pick:
    r = next(iter(teacher.predict(source=path, imgsz=640, conf=0.05, iou=0.7,
                                  device=1, verbose=False)))
    arr = r.plot(conf=False)  # 不显示 conf 文本, 避免低置信度标签杂乱
    out = OUT / f"diag_{Path(path).stem}.jpg"
    import cv2
    cv2.imwrite(str(out), arr)
    print(f"  saved {out.name}  (n@0.05={len(r.boxes) if r.boxes is not None else 0})")

print(f"\n可视化输出目录: {OUT}")
