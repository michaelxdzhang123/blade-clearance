#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断: 高阈值帧间差分的 Blade narrow band 可行性"""
import av, numpy as np, cv2, os

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")

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

frames, nos = extract(1618, 1650)
print(f"Blade 主体经过段 {len(frames)} 帧 (frame {nos[0]}~{nos[-1]})")

# 对每帧, 高阈值 diff 的连通域分析
print("\nframe | diff>30 area% | 最大连通域bbox | 最大连通域面积%")
for i in range(1, len(frames)):
    d = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
    m30 = (d > 30).astype(np.uint8)
    area30 = m30.mean()
    # 连通域
    n, labels, stats, cent = cv2.connectedComponentsWithStats(m30, connectivity=8)
    if n > 1:
        # 最大连通域 (排除背景 label 0)
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x0, y0 = stats[largest, cv2.CC_STAT_LEFT], stats[largest, cv2.CC_STAT_TOP]
        w, h = stats[largest, cv2.CC_STAT_WIDTH], stats[largest, cv2.CC_STAT_HEIGHT]
        a = stats[largest, cv2.CC_STAT_AREA]
        if i in (1, 5, 10, 15, 20, 25, 30):
            print(f"{nos[i]:5d} | {area30*100:11.2f}% | ({x0},{y0}) {w}x{h} | {a/(2560*1440)*100:.2f}%")
