#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_blade_tips_direct.py — 独立脚本: 对本地 MP4 逐帧推理, 直接把叶尖画成绿点合成视频。

为什么不用 JSONL:
  JSONL 是从「循环推流的 RTSP 流」生成的 (ffmpeg -stream_loop -1),
  reader 连上时视频已经循环到中间, 所以 JSONL 的 frame_id 和本地 MP4 的帧序号对不上。
  本脚本直接对本地 MP4 逐帧推理, frame 序号天然对齐。
  绿点表示模型定位的叶尖候选；无检测时保留最后位置，并非当前帧实测位置。
  连续缺失至少三帧后，连续两帧检测到叶片才更新绿点；--no-hold 可关闭保持。

依赖 blade_rtsp_system/src (YOLODetector / blade_segmenter / blade_locator),
复用 main.py 的检测与定位模块；显示层额外保留上一位置。

用法 (在项目根目录跑):
  python blade_rtsp_system/visualize_blade_tips_direct.py \
      --video data/training-data/test01-video-2026-08-28_134655_607.mp4 \
      --output blade_rtsp_system/output/blade_tips_direct.mp4 \
      --max-frames 600
"""
from __future__ import annotations

import argparse
import os
import sys
import math

import cv2  # OpenCV: 读视频 / resize / 画点 / 写视频

# 把 blade_rtsp_system 目录加入模块搜索路径, 保证 `from src import` 可用
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from src.yolo_detector import YOLODetector          # noqa: E402  模型加载 + 单帧推理
from src import blade_segmenter, blade_locator      # noqa: E402  叶片提取 + 定位 (叶尖候选)


class TipHold:
    """短暂漏检保留位置；离开三帧后，新叶片连续出现两帧才接替。"""

    def __init__(self):
        self.last = []
        self.missing = 0
        self.pending = 0

    def update(self, located):
        tips = []
        for blade in located:
            tip = blade.get("blade_tip_candidate")
            if tip is not None and len(tip) == 2 and all(math.isfinite(v) for v in tip):
                tips.append({"blade_tip_candidate": list(tip)})
        if not tips:
            self.missing += 1
            self.pending = 0
            return self.last
        if self.last and self.missing >= 3:
            self.pending += 1
            if self.pending < 2:
                return self.last
        self.last = tips
        self.missing = 0
        self.pending = 0
        return self.last


def draw_tips(frame, located, sx, sy, radius):
    """把 located 里的 blade_tip_candidate 缩放后画成绿点 (绿圆 + 白描边)。"""
    for b in located:                                  # 遍历每个 blade
        tip = b.get("blade_tip_candidate")             # 叶尖候选 [x, y] (resize 后坐标)
        if not tip:                                    # 无叶尖跳过
            continue
        px = int(tip[0] * sx)                          # x 缩放回原视频
        py = int(tip[1] * sy)                          # y 缩放回原视频
        cv2.circle(frame, (px, py), radius, (0, 255, 0), -1)       # 绿色实心圆 (小点)
        cv2.circle(frame, (px, py), radius + 1, (255, 255, 255), 1)  # 白描边 (细, 等比缩小)


def main() -> int:
    parser = argparse.ArgumentParser(description="对本地 MP4 逐帧推理并叠加叶尖绿点")
    parser.add_argument("--video", default="data/training-data/test01-video-2026-08-28_134655_607.mp4")
    parser.add_argument("--model", default="blade_rtsp_system/models/best.pt")
    parser.add_argument("--output", default="blade_rtsp_system/output/blade_tips_direct.mp4")
    parser.add_argument("--resize", nargs=2, type=int, default=(640, 360),
                        help="推理前缩放到 W H (与原 pipeline 一致)")
    parser.add_argument("--max-frames", type=int, default=600, help="最多处理多少帧")
    parser.add_argument("--radius", type=int, default=4, help="绿点半径 (原视频像素)")
    parser.add_argument("--no-hold", action="store_true",
                        help="关闭位置保持，只标注当前帧检测到的叶尖")
    args = parser.parse_args()

    # 1. 加载模型 (与原 pipeline 一致: imgsz=640, retina_masks=True, device=cpu)
    det = YOLODetector(args.model, imgsz=640, conf=0.5, device="cpu", retina_masks=True)
    det.load()
    print("模型加载完成", flush=True)

    # 2. 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}")
        return 1
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 原视频宽
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))      # 原视频高
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0           # 帧率
    rw, rh = args.resize                              # resize 目标
    sx = vw / rw                                      # x 缩放因子 (resize 空间 → 原视频)
    sy = vh / rh                                      # y 缩放因子
    print(f"视频 {vw}x{vh} @ {fps:.2f}fps, resize {rw}x{rh}, 缩放 sx={sx:.2f} sy={sy:.2f}")

    # 3. 输出 writer
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    if not out.isOpened():
        print("无法创建输出视频")
        cap.release()
        return 1

    # 4. 逐帧: resize → 推理 → 定位叶尖 → 缩放 → 画点 → 写帧
    i = 0
    drew = 0
    hold = TipHold()
    while i < args.max_frames:
        ret, frame = cap.read()                        # 读一帧 (原始 2560x1440)
        if not ret:
            break
        small = cv2.resize(frame, (rw, rh))            # 缩放到推理尺寸 (与原 pipeline 一致)
        result, _ = det.predict(small)                 # YOLO 推理
        det_res = blade_segmenter.extract_blades(result, class_name="blade")  # 提取 blade
        located = blade_locator.locate_blades(         # 定位 (含叶尖候选)
            det_res["blade_instances"], roi_offset=(0.0, 0.0),
            frame_width=rw, frame_height=rh)
        displayed = located if args.no_hold else hold.update(located)
        if displayed:
            draw_tips(frame, displayed, sx, sy, args.radius)
            drew += 1
        out.write(frame)                               # 写帧
        i += 1
        if i % 100 == 0:
            print(f"  已处理 {i} 帧, 其中 {drew} 帧画了叶尖", flush=True)

    cap.release()
    out.release()
    print(f"完成: {i} 帧, {drew} 帧画了叶尖绿点 → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
