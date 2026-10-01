#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
opencv/refine_blade_tip.py — 叶尖亚像素精化

诊断结论 (关键):
  叶尖 = 叶片横扫带的最外端, 在图像里是"暗色叶片 → 亮色天空"的
  一维灰度阶跃边缘 (edge), 不是角点 (corner)。
  - Harris 响应在叶尖点为负 → 证明不是角点
  - cornerSubPix 需要角点结构(两个正交梯度), 对边缘端点会沿边缘漂移

提供两种方法对比:
  A. cornerSubPix (用户要求): 亚像素角点细化, 但对边缘端点漂移大
  B. 梯度峰值抛物线拟合 (推荐): 沿横扫方向(x)灰度阶跃边缘的亚像素定位

原理 (方法 B):
  叶尖处灰度剖面是暗→亮阶跃, 梯度 |dI/dx| 在阶跃处有峰值。
  取峰值点 k 及其左右两点 (g[k-1], g[k], g[k+1]), 抛物线拟合:
      δ = (g[k-1] - g[k+1]) / (2·(g[k-1] - 2·g[k] + g[k+1]))
  亚像素位置 x = k + δ  (δ∈[-0.5, 0.5])
"""
from __future__ import annotations
import csv
import os

import av
import cv2
import numpy as np

VIDEO = "data/test01-video-2026-08-28_134655_607.mp4"
BG_PATH = "outputs/TASK2_GATE_CLOSURE/bg_idle_median.npy"

DIFF_THR = 25
Y_MIN = 800
BOTTOM_REACH = 1430
MIN_AREA = 5000


def blade_tip_int(img, bg):
    """背景差分找叶尖整数坐标 (触底横扫前端)。"""
    diff = img - bg
    moving = np.abs(diff) > DIFF_THR
    moving[:Y_MIN, :] = False
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        moving.astype(np.uint8), 8)
    best = None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if a > MIN_AREA and y + h >= BOTTOM_REACH:
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


def refine_corner_subpix(img8, tip, win=7):
    """方法 A: cornerSubPix 亚像素细化 (对边缘端点漂移, 仅供参考)。"""
    pts = np.array([[tip[0], tip[1]]], dtype=np.float32).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    r = cv2.cornerSubPix(img8, pts, (win, win), (-1, -1), crit)
    return r.reshape(-1, 2)[0]


def refine_edge_parabola(img, tip, half=10):
    """方法 B: 梯度峰值抛物线拟合, 沿 x 方向灰度阶跃边缘亚像素定位。

    叶尖是横扫带最外端, 在触底行 y=ty 上沿 +x 方向(横扫方向)灰度由暗变亮。
    找 |dI/dx| 峰值, 抛物线拟合亚像素位置。
    """
    tx, ty = tip
    x0 = max(0, tx - half)
    x1 = min(img.shape[1], tx + half + 1)
    prof = img[ty, x0:x1].astype(np.float64)
    grad = np.gradient(prof)
    abs_grad = np.abs(grad)
    k = int(np.argmax(abs_grad))
    # 抛物线拟合: 用相邻点
    if 0 < k < len(grad) - 1:
        g1, g2, g3 = abs_grad[k-1], abs_grad[k], abs_grad[k+1]
        denom = 2.0 * (g1 - 2.0 * g2 + g3)
        if abs(denom) > 1e-9:
            delta = (g1 - g3) / denom
        else:
            delta = 0.0
        sub_x = x0 + k + delta
    else:
        sub_x = float(x0 + k)
    return sub_x, float(ty)


def main():
    bg = np.load(BG_PATH).astype(np.float32)
    c = av.open(VIDEO)
    s = c.streams.video[0]

    print(f"{'帧':>6} {'整数叶尖':>12} {'cornerSubPix':>16} {'边缘拟合(推荐)':>16} "
          f"{'corner位移':>10}")
    rows = []
    for i, f in enumerate(c.decode(s)):
        # 只在有叶片的帧处理 (叶片框触底)
        img = f.to_ndarray(format="gray").astype(np.float32)
        tip = blade_tip_int(img, bg)
        if tip is None:
            continue
        img8 = img.astype(np.uint8)

        rc = refine_corner_subpix(img8, tip)
        rx, ry = refine_edge_parabola(img, tip)

        print(f"{i:>6} ({tip[0]},{tip[1]})  "
              f"({rc[0]:.2f},{rc[1]:.2f})  "
              f"({rx:.2f},{ry:.2f})  "
              f"({rc[0]-tip[0]:+.2f},{rc[1]-tip[1]:+.2f})")
        rows.append({"frame": i, "tip_int_x": tip[0], "tip_int_y": tip[1],
                     "corner_x": round(rc[0], 3), "corner_y": round(rc[1], 3),
                     "edge_x": round(rx, 3), "edge_y": round(ry, 3)})

        if len(rows) >= 30:
            break

    c.close()

    csv_path = "outputs/tip_refine_compare.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "tip_int_x", "tip_int_y",
                                           "corner_x", "corner_y",
                                           "edge_x", "edge_y"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n对比结果 → {csv_path}")


if __name__ == "__main__":
    main()
