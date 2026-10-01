#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""叶尖灰度剖面分析: 确认叶尖是边缘端点而非角点, 找正确亚像素方法。"""
import os
import numpy as np
import cv2
import av

VIDEO = "data/test01-video-2026-08-28_134655_607.mp4"
BG = "outputs/TASK2_GATE_CLOSURE/bg_idle_median.npy"
bg = np.load(BG).astype(np.float32)


def blade_tip_int(img, bg):
    diff = img - bg
    moving = np.abs(diff) > 25
    moving[:800, :] = False
    n, labels, stats, _ = cv2.connectedComponentsWithStats(moving.astype(np.uint8), 8)
    best = None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if a > 5000 and y + h >= 1430:
            if best is None or a > best[0]:
                best = (a, x, y, w, h, i)
    if best is None:
        return None
    _, x, y, w, h, li = best
    mask = (labels == li)
    ys, xs = np.where(mask)
    ymax = ys.max()
    tip_x = xs[ys == ymax].min()
    return (int(tip_x), int(ymax))


c = av.open(VIDEO)
s = c.streams.video[0]
for i, f in enumerate(c.decode(s)):
    if i != 1617:
        continue
    img = f.to_ndarray(format="gray").astype(np.float32)
    tip = blade_tip_int(img, bg)
    tx, ty = tip
    print(f"帧 {i}: 整数叶尖 = ({tx}, {ty})")

    # 沿 x 方向灰度剖面 (y = ty, 从横扫带内部向左到外部)
    print(f"\n灰度剖面 (y={ty}, x 从 {tx-20} 到 {tx+15}, 每像素):")
    profile = img[ty, tx-20:tx+15]
    for k, v in enumerate(profile):
        x = tx-20+k
        bar = '#' * int(v/15)
        print(f"  x={x:>4}: {v:5.1f}  {bar}")
    # 标记整数叶尖
    print(f"         ^ 整数叶尖 x={tx} (横扫带最左暗像素)")

    # 灰度梯度 (沿 x)
    grad = np.gradient(profile)
    print(f"\n灰度梯度绝对值 (沿 x):")
    for k, v in enumerate(grad):
        x = tx-20+k
        print(f"  x={x:>4}: {v:+6.1f}")
    c.close()
    break
