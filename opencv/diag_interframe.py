#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证: 帧间差分 (相邻25fps) vs 背景差分 的 motion mask 信噪比"""
import av, numpy as np, cv2, os

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
bg = np.load(os.path.join(OUT, "bg_idle_median.npy"))

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

# IDLE 段 + BladePass 段
idle_frames, idle_nos = extract(7200, 7300)
pass_frames, pass_nos = extract(1610, 1750)

# 帧间差分 motion mask (相邻帧)
def interframe_motion(frames, thr):
    areas = []
    for i in range(1, len(frames)):
        d = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
        areas.append((d > thr).mean())
    return np.mean(areas)

# 背景差分 motion mask
def bg_motion(frames, thr):
    areas = []
    for f in frames:
        d = np.abs(f.astype(float) - bg.astype(float))
        areas.append((d > thr).mean())
    return np.mean(areas)

print("=== 帧间差分 vs 背景差分 (motion_area %) ===")
print(f"{'thr':>4} | {'IDLE帧间':>8} {'Blade帧间':>9} {'SNR帧间':>7} | {'IDLE背景':>8} {'Blade背景':>9} {'SNR背景':>7}")
for thr in [5, 10, 15, 20, 25]:
    i_idle = interframe_motion(idle_frames, thr)
    i_pass = interframe_motion(pass_frames, thr)
    b_idle = bg_motion(idle_frames, thr)
    b_pass = bg_motion(pass_frames, thr)
    snr_i = i_pass / max(i_idle, 1e-6)
    snr_b = b_pass / max(b_idle, 1e-6)
    print(f"{thr:>4} | {i_idle*100:>7.2f}% {i_pass*100:>8.2f}% {snr_i:>6.1f}x | "
          f"{b_idle*100:>7.2f}% {b_pass*100:>8.2f}% {snr_b:>6.1f}x")
