#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yolo26/yolo26-capture.py — YOLO26 视频推理 (无窗口批处理)

逐帧推理 → 手绘绿色框 → 输出标注视频 + 检测结果 CSV。

⚠️ 重要: 默认模型 yolo26n.pt 是 COCO 80 类通用检测, 类别表没有 blade。
  用它直接推理叶片视频会把塔筒/叶片误判为 toilet/surfboard/person。
  要检测叶片, 必须先 train_blade.py 训练出 best.pt (blade 单类),
  再传 best.pt 作 model 参数。

用法:
  # 默认 COCO 模型 (通用检测, 无 blade 类)
  uv run yolo26/yolo26-capture.py --drive data/xxx.mp4 --out blade-box

  # 用训练好的叶片模型 (blade 类)
  uv run yolo26/yolo26-capture.py runs/detect/blade_train/weights/best.pt \
      --drive data/xxx.mp4 --out blade-box-yolo
"""
import argparse
import csv
import os

import cv2
from ultralytics import YOLO

# ── 参数解析 ─────────────────────────────────────────
# DEFAULT_MODEL 用 __file__ 定位脚本所在目录的 yolo26n.pt:
#   无论从哪个目录运行, 都能找到模型(不依赖工作目录)。
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26n.pt")
parser = argparse.ArgumentParser(description="YOLO26 video inference (headless batch)")
parser.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                    help=f"Path to the YOLO26 model file (.pt), default: {DEFAULT_MODEL}")
parser.add_argument("--drive", required=True, help="Path to the input video file")
parser.add_argument("--out", default="blade-box", help="Output directory (default: blade-box)")
args = parser.parse_args()

# ── 加载模型 ─────────────────────────────────────────
model = YOLO(args.model)

# ── 打开视频 (OpenCV) ────────────────────────────────
cap = cv2.VideoCapture(args.drive)
if not cap.isOpened():
    raise SystemExit(f"无法打开视频: {args.drive}")

# 视频元数据: fps / 宽高 / 总帧数 (用于 VideoWriter 和进度打印)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ── 输出: 标注视频 + 检测 CSV ────────────────────────
os.makedirs(args.out, exist_ok=True)
csv_path = os.path.join(args.out, "detections.csv")
video_path = os.path.join(args.out, "annotated.mp4")

# VideoWriter: mp4v 编码, 保持原视频 fps/分辨率
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
csvf = open(csv_path, "w", newline="")
csvw = csv.writer(csvf)
csvw.writerow(["frame", "cls", "cls_name", "conf", "x1", "y1", "x2", "y2"])

print(f"视频: {args.drive}  ({width}x{height}, {fps:.1f}fps, {total} 帧)")
print(f"模型: {args.model}  输出: {args.out}/\n")

# ── 逐帧推理 (流式, O(1) 内存) ───────────────────────
frame_idx = 0
n_detections = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 推理: results[0].boxes 是检测框集合
    #   b.xyxy[0]  → 像素坐标 [x1,y1,x2,y2]
    #   b.cls[0]   → 类别 ID
    #   b.conf[0]  → 置信度 [0,1]
    #   model.names[cls] → 类别名 (如 'toilet', 'blade')
    results = model(frame)
    boxes = results[0].boxes

    # ── 手绘绿色框 (BGR 绿, 替代 results[0].plot() 默认彩色) ──
    # 为什么手绘: plot() 用 ultralytics 默认多色(每类不同色),
    #   这里统一绿色 BGR(0,255,0), 线宽 4, 带类别+置信度标签。
    GREEN = (0, 255, 0)
    annotated_frame = frame.copy()
    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            name = model.names[cls]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), GREEN, 4)
            label = f"{name} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

    writer.write(annotated_frame)

    # ── 写 CSV (第二个循环: 避免与画框循环重复遍历 boxes) ──
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            xyxy = b.xyxy[0].tolist()
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            name = model.names[cls]
            csvw.writerow([frame_idx, cls, name, f"{conf:.4f}", *xyxy])
            n_detections += 1

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"  {frame_idx}/{total} 帧  累计检测 {n_detections} 个", flush=True)

# ── 收尾 ────────────────────────────────────────────
cap.release()
writer.release()
csvf.close()

print(f"\n完成: {frame_idx} 帧, 检测 {n_detections} 个目标")
print(f"标注视频 → {video_path}")
print(f"检测结果 → {csv_path}")
