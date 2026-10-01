#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase1: 25fps 连续帧扫描, 找 W0(IDLE) + Blade Pass 窗口"""
import av, numpy as np, csv, os

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
os.makedirs(OUT, exist_ok=True)

container = av.open(VIDEO)
stream = container.streams.video[0]
stream.thread_type = "AUTO"

prev = None
motion = []  # (frame_idx, timestamp, mean_absdiff)
SCALE = 0.25  # 降采样到 1/4

for idx, frame in enumerate(container.decode(stream)):
    img = frame.to_ndarray(format="gray")
    h, w = img.shape
    small = img[::4, ::4].astype(np.float32)  # 640x360
    if prev is not None:
        d = np.abs(small - prev).mean()
        t = float(frame.time) if frame.time is not None else idx / 25.0
        motion.append((idx, t, d))
    prev = small

container.close()

motion = np.array(motion)
print(f"扫描完成: {len(motion)} 帧间差分, 时间范围 {motion[0,1]:.1f}~{motion[-1,1]:.1f}s")

# 保存运动强度序列
with open(os.path.join(OUT, "motion_timeline.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_idx", "timestamp", "mean_absdiff"])
    w.writerows(motion)

d = motion[:, 2]
print(f"mean_absdiff 分布: min={d.min():.2f} p5={np.percentile(d,5):.2f} p25={np.percentile(d,25):.2f} "
      f"p50={np.percentile(d,50):.2f} p75={np.percentile(d,75):.2f} p95={np.percentile(d,95):.2f} max={d.max():.2f}")

# 找 IDLE 窗口 (连续低运动, diff < p25)
idle_thr = np.percentile(d, 25)
print(f"\nIDLE 阈值 (diff < {idle_thr:.2f}):")
idle_mask = d < idle_thr
# 找连续段
segments = []
start = None
for i, m in enumerate(idle_mask):
    if m and start is None:
        start = i
    elif not m and start is not None:
        if i - start >= 25:  # 至少 1 秒
            segments.append((start, i, d[start:i].mean()))
        start = None
if start is not None and len(idle_mask) - start >= 25:
    segments.append((start, len(idle_mask), d[start:].mean()))

segments.sort(key=lambda x: -x[2])  # 按平均 diff 升序(最干净在前) -> 反转用负数
print(f"找到 {len(segments)} 个 IDLE 段 (>=25帧):")
for s in segments[:10]:
    f0, f1, m = s
    print(f"  frame {f0}~{f1} (t={motion[f0,1]:.1f}~{motion[f1-1,1]:.1f}s, {f1-f0}帧) mean_diff={m:.2f}")

# 找 Blade Pass (高运动段)
pass_thr = np.percentile(d, 90)
print(f"\nBlade Pass 阈值 (diff > {pass_thr:.2f}):")
pass_mask = d > pass_thr
pass_segments = []
start = None
for i, m in enumerate(pass_mask):
    if m and start is None:
        start = i
    elif not m and start is not None:
        if i - start >= 5:
            pass_segments.append((start, i, d[start:i].max()))
        start = None
print(f"找到 {len(pass_segments)} 个 Blade Pass 候选:")
for s in pass_segments[:15]:
    f0, f1, m = s
    print(f"  frame {f0}~{f1} (t={motion[f0,1]:.1f}~{motion[f1-1,1]:.1f}s, {f1-f0}帧) peak_diff={m:.2f}")
