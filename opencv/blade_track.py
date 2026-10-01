#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Blade 运动轨迹分析: Blade Pass 窗口内 motion mask 质心轨迹"""
import av, numpy as np, cv2, os

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")

def extract(f0, f1):
    container = av.open(VIDEO); stream = container.streams.video[0]; stream.thread_type = "AUTO"
    want = set(range(f0, f1)); frames = {}
    for idx, frame in enumerate(container.decode(stream)):
        if idx in want:
            frames[idx] = frame.to_ndarray(format="gray")
            want.discard(idx)
        if not want: break
    container.close()
    ks = sorted(frames.keys())
    return [frames[k] for k in ks], ks

# Blade Pass 窗口: frame 1600~1680 (含 peak 1617)
frames, nos = extract(1600, 1680)
print(f"Blade Pass 窗口 {len(frames)} 帧 (frame {nos[0]}~{nos[-1]})")

# 每帧 motion mask (帧间差分) 的质心 + 面积
print("\nframe | motion_area% | centroid_x | centroid_y | 位移")
prev_c = None
for i in range(1, len(frames)):
    d = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
    area = (d > 15).mean()
    ys, xs = np.where(d > 15)
    if len(xs) > 50:
        cx, cy = float(xs.mean()), float(ys.mean())
        disp = f"{np.hypot(cx-prev_c[0], cy-prev_c[1]):.0f}px" if prev_c else "-"
        prev_c = (cx, cy)
        print(f"{nos[i]:5d} | {area*100:12.2f}% | {cx:10.0f} | {cy:10.0f} | {disp}")
    else:
        print(f"{nos[i]:5d} | {area*100:12.2f}% | (散点<50) | - | -")
        prev_c = None
