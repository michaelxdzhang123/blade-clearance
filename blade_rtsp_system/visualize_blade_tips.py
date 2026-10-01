#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_blade_tips.py — 独立脚本: 把 JSONL 里检测到的叶尖候选画成绿色圆点, 合成视频。

完全独立, 不依赖 blade_rtsp_system/src 下的任何模块 (只用 cv2 + json + argparse + os)。

原理:
  - JSONL 每帧一行, 含 frame_id / frame_width / frame_height / blades[].blade_tip_candidate
  - blade_tip_candidate = [x, y] 是「blade polygon 中 y 最大的顶点」(叶片自由端触底端), 单位是 resize 后坐标
  - 本脚本按 frame_id 对齐原视频帧, 并把坐标从 resize 空间缩放回原视频分辨率, 画绿点

用法 (在项目根目录跑):
  python blade_rtsp_system/visualize_blade_tips.py \
      --jsonl blade_rtsp_system/output/jsonl/blade_positions.jsonl \
      --video data/training-data/test01-video-2026-08-28_134655_607.mp4 \
      --output blade_rtsp_system/output/blade_tips_overlay.mp4
"""
from __future__ import annotations

import argparse
import json
import os

import cv2  # OpenCV: 读视频 / 画圆点 / 写视频


def load_tips(jsonl_path: str):
    """读 JSONL, 返回 (tip_map, frame_w, frame_h)。

    tip_map: dict[int, list[tuple[float, float]]] —— frame_id → 该帧所有叶尖候选 (resize 后坐标)
    frame_w / frame_h: JSONL 里记录的帧尺寸 (即 resize 后的 640x360 等), 用于缩放回原视频
    """
    tip_map = {}
    frame_w = frame_h = 0
    with open(jsonl_path, encoding="utf-8") as f:           # 打开 JSONL
        for line in f:                                       # 逐行读
            line = line.strip()                              # 去空白
            if not line:                                     # 空行跳过
                continue
            d = json.loads(line)                             # 解析这一帧
            frame_w = int(d.get("frame_width", frame_w))     # 记录帧宽 (取最后一个非空值)
            frame_h = int(d.get("frame_height", frame_h))    # 记录帧高
            fid = int(d["frame_id"])                         # 帧序号
            tips = [                                          # 收集本帧所有叶尖候选
                (float(b["blade_tip_candidate"][0]), float(b["blade_tip_candidate"][1]))
                for b in d.get("blades", [])
                if b.get("blade_tip_candidate")              # 只有叶尖候选非空的 blade 才收
            ]
            if tips:                                         # 有叶尖才记录
                tip_map[fid] = tips
    return tip_map, frame_w, frame_h


def main() -> int:
    parser = argparse.ArgumentParser(description="把 JSONL 叶尖候选叠加成绿色圆点视频")
    parser.add_argument("--jsonl", default="blade_rtsp_system/output/jsonl/blade_positions.jsonl",
                        help="JSONL 结果文件路径")
    parser.add_argument("--video", default="data/training-data/test01-video-2026-08-28_134655_607.mp4",
                        help="原始视频路径")
    parser.add_argument("--output", default="blade_rtsp_system/output/blade_tips_overlay.mp4",
                        help="输出视频路径")
    parser.add_argument("--radius", type=int, default=18,
                        help="绿点半径 (像素, 按原视频分辨率计)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理多少帧 (0=全部)")
    args = parser.parse_args()

    # 1. 读 JSONL → 叶尖映射表
    tip_map, jw, jh = load_tips(args.jsonl)                  # tip_map[frame_id] = [(x,y),...]
    print(f"JSONL: {len(tip_map)} 帧含叶尖, 帧尺寸 {jw}x{jh}")

    # 2. 打开原始视频
    cap = cv2.VideoCapture(args.video)                       # 打开视频
    if not cap.isOpened():                                   # 打开失败
        print(f"无法打开视频: {args.video}")
        return 1
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))              # 原视频宽 (2560)
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))             # 原视频高 (1440)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0                  # 原视频帧率
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)      # 总帧数
    print(f"视频: {vw}x{vh} @ {fps:.2f}fps, 共 {total} 帧")

    # 3. 缩放因子: resize 空间 → 原视频空间 (640x360 → 2560x1440 时为 4x)
    sx = vw / jw if jw else 1.0                              # x 缩放因子
    sy = vh / jh if jh else 1.0                              # y 缩放因子
    print(f"缩放因子: sx={sx:.3f} sy={sy:.3f}")

    # 4. 输出视频 writer (mp4v 编码, 与原视频同分辨率/帧率)
    out = cv2.VideoWriter(args.output,                       # 输出路径
                          cv2.VideoWriter_fourcc(*"mp4v"),   # 编码器
                          fps, (vw, vh))                     # 帧率 + 分辨率
    if not out.isOpened():                                   # writer 打开失败
        print("无法创建输出视频 (检查编码器 mp4v 是否可用)")
        cap.release()
        return 1

    # 5. 逐帧处理: 对齐 frame_id 并叠加绿点
    i = 0                                                     # 视频帧序号 (0 起)
    drawn = 0                                                 # 已画点帧数统计
    while True:
        ret, frame = cap.read()                               # 读一帧
        if not ret:                                           # 读完
            break
        fid = i + 1                                           # frame_id 是 1 起 (JSONL 第一帧=1)
        if fid in tip_map:                                    # 这一帧有叶尖候选
            for (x, y) in tip_map[fid]:                       # 遍历该帧所有叶尖
                px = int(x * sx)                              # x 缩放回原视频坐标
                py = int(y * sy)                              # y 缩放回原视频坐标
                cv2.circle(frame, (px, py), args.radius,      # 画绿色实心圆点
                           (0, 255, 0), -1)                   # BGR 绿 (0,255,0), -1=填充
                cv2.circle(frame, (px, py), args.radius + 2,  # 外圈描边 (更醒目)
                           (0, 255, 0), 2)                    # 粗 2px 描边
            drawn += 1
        out.write(frame)                                      # 写这一帧到输出
        i += 1
        if args.max_frames > 0 and i >= args.max_frames:      # 达到帧数上限
            break

    cap.release()                                             # 释放视频源
    out.release()                                             # 关闭 writer (落盘)
    print(f"完成: 处理 {i} 帧, 其中 {drawn} 帧画了叶尖绿点 → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
