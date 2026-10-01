#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
opencv/build_bg.py — 从视频的 IDLE 帧(叶片静止时段)重建 median 背景

用于: 加雾等变换后的视频, 原背景模型不再适用时, 重新建背景。
  (背景是"这个视频"的塔筒/天空/光照, 换视频(尤其加雾后塔筒灰度变了)
   必须重建, 否则背景差分会把整个塔筒误判为运动。)

原理: 叶片横扫是周期性运动, IDLE 帧 = 帧间差分最小的帧(画面静止)。
取 diff 最小的 N 帧做逐像素 median, 得干净背景(塔筒+天空, 无叶片)。

用法:
  python opencv/build_bg.py --video <视频.mp4> --output <背景.npy>
"""
import argparse

import av
import numpy as np


def build_bg(video_path, out_path, n_idle=50):
    """从 IDLE 帧重建 median 背景, 保存 .npy。

    两遍解码设计:
      第一遍: 降采样 1/4 算帧间差分, 找 diff 最小的 n_idle 帧。
              降采样(640×360)加快计算, 帧间差分的统计意义不受影响。
      第二遍: 取这 n_idle 帧全分辨率灰度图, 逐像素 median 得背景。
    """
    # ── 第一遍: 降采样帧间差分, 找 IDLE 帧 ─────────────
    # diffs[i] = |frame[i+1] - frame[i]| 的降采样均值。
    # 叶片横扫帧 diff 大(画面突变), IDLE 帧 diff 小(画面静止)。
    c = av.open(video_path)
    s = c.streams.video[0]
    prev = None
    diffs = []
    for frame in c.decode(s):
        img = frame.to_ndarray(format="gray")
        small = img[::4, ::4].astype(np.float32)  # 降采样 1/4 → 640×360
        if prev is not None:
            diffs.append(float(np.abs(small - prev).mean()))
        prev = small
    c.close()

    diffs = np.array(diffs)
    print(f"扫描 {len(diffs)+1} 帧, 帧间差分范围 [{diffs.min():.2f}, {diffs.max():.2f}]")

    # IDLE 帧号 = diff 最小的 n_idle 帧 (diff[i] 小 => 帧 i+1 静止, 故 +1)
    idle_frames = sorted((np.argsort(diffs)[:n_idle] + 1).tolist())
    print(f"选取 {len(idle_frames)} 个 IDLE 帧: frame {idle_frames[0]}~{idle_frames[-1]}")

    # ── 第二遍: 取 IDLE 帧做 median 背景 ────────────────
    # 为什么 median 而非 mean: 叶片横扫的瞬时像素是少数(占空比~47%),
    # median 会把这些瞬时像素排到中位数之外, 得到"干净背景"。
    # mean 会残留叶片扫过的痕迹(瞬时像素拉高均值)。
    c = av.open(video_path)
    s = c.streams.video[0]
    want = set(idle_frames)
    frames = []
    for idx, frame in enumerate(c.decode(s)):
        if idx in want:
            frames.append(frame.to_ndarray(format="gray"))
            want.discard(idx)
        if not want:      # 全部取齐就提前退出, 不解码剩余帧(省时)
            break
    c.close()

    # 注意 astype(np.float32): 避免 np.median 把 uint8 stack 转 float64 导致内存翻倍
    bg = np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)
    np.save(out_path, bg)
    print(f"median 背景已保存: {out_path}  shape={bg.shape}  mean={bg.mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description="从 IDLE 帧重建 median 背景")
    parser.add_argument("--video", required=True, help="输入视频路径 (*.mp4)")
    parser.add_argument("--output", required=True, help="输出背景 .npy 路径")
    parser.add_argument("--n-idle", type=int, default=50,
                        help="IDLE 帧数量 (默认 50)")
    args = parser.parse_args()
    build_bg(args.video, args.output, args.n_idle)


if __name__ == "__main__":
    main()
