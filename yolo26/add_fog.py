#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yolo26/add_fog.py — 给视频加重雾 (大气散射模型), 保存新视频

雾的物理模型 (大气散射, Koschmieder):
  I(x) = J(x)·t(x) + A·(1 − t(x))
  - I: 观测到的有雾图像
  - J: 清晰场景
  - A: 大气光 (灰白天空, 约 220)
  - t: 透射率 t = e^(−β·d), β 散射系数, d 深度

深度假设 (本视频物理): 相机在机舱朝下拍, 塔筒上层(y小)离相机近,
叶片横扫在下层(y大)离相机远。故深度 d ∝ y (顶部近→雾轻, 底部远→雾浓)。

heavy fog: β 取大值, 底部(叶片)雾浓, 全图对比度大幅下降。
"""
import av
import cv2
import numpy as np

VIDEO = "data/test01-video-2026-08-28_134655_607.mp4"
OUT = "data/test01-video-2026-09-09-heavy-foggy.mp4"

A = 220.0     # 大气光 (灰白)
T = 0.25      # 均匀透射率 (heavy fog: 全图一致, 25% 清晰 + 75% 雾)


def add_fog(bgr, A=A, t=T):
    """均匀重雾: I = J·t + A·(1−t), 全图一致透射率, 不管深度。"""
    foggy = bgr.astype(np.float32) * t + A * (1.0 - t)
    return np.clip(foggy, 0, 255).astype(np.uint8)


def main():
    c = av.open(VIDEO)
    s = c.streams.video[0]
    fps = float(s.average_rate or 25.0)
    w = s.codec_context.width
    h = s.codec_context.height
    total = s.frames or 0

    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    print(f"视频: {VIDEO}  ({w}x{h}, {fps:.1f}fps, {total} 帧)")
    print(f"雾参数: A={A} 均匀透射率 t={T}")
    print(f"输出: {OUT}\n")

    n = 0
    for frame in c.decode(s):
        bgr = frame.to_ndarray(format="bgr24")
        writer.write(add_fog(bgr))
        n += 1
        if n % 500 == 0:
            print(f"  {n}/{total} 帧", flush=True)

    c.close()
    writer.release()
    print(f"\n完成: {n} 帧 → {OUT}")


if __name__ == "__main__":
    main()
