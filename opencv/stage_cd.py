#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage C/D: clean IDLE 背景上的 StaticDiff + MOG2(warm-up), 验证 motion mask 质量"""
import av, numpy as np, cv2, os, csv

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
bg = np.load(os.path.join(OUT, "bg_idle_median.npy"))
H, W = bg.shape
TOWER_X = 434.0
print(f"clean IDLE median 背景加载: {bg.shape}")

# 提取测试段: 含 Blade Pass (frame 1610~1750) + IDLE 段 (frame 7200~7300)
def extract(f0, f1):
    container = av.open(VIDEO); stream = container.streams.video[0]; stream.thread_type = "AUTO"
    want = set(range(f0, f1)); frames = {}
    for idx, frame in enumerate(container.decode(stream)):
        if idx in want:
            frames[idx] = frame.to_ndarray(format="gray")
            want.discard(idx)
        if not want: break
    container.close()
    return [frames[k] for k in sorted(frames.keys())], sorted(frames.keys())

pass_frames, pass_nos = extract(1610, 1750)
idle_frames, idle_nos = extract(7200, 7300)
print(f"提取 BladePass 段 {len(pass_frames)} 帧, IDLE 段 {len(idle_frames)} 帧")

# StaticDiff threshold 扫描: 在 IDLE 段看 false motion, 在 BladePass 段看 blade 保留
print("\n=== StaticDiff threshold 扫描 (clean IDLE median 背景) ===")
for thr in [10, 15, 20, 25, 30]:
    idle_fm = [((np.abs(f.astype(float) - bg.astype(float)) > thr).mean()) for f in idle_frames]
    pass_fm = [((np.abs(f.astype(float) - bg.astype(float)) > thr).mean()) for f in pass_frames]
    print(f"thr={thr:2d}: IDLE false_motion_area={np.mean(idle_fm)*100:5.2f}%  "
          f"BladePass motion_area={np.mean(pass_fm)*100:5.2f}%")

# MOG2 warm-up 验证
print("\n=== MOG2 warm-up (50 IDLE 帧) ===")
mog = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
# warm-up on IDLE frames
for f in idle_frames[:50]:
    mog.apply(f, learningRate=0.01)
# 冻结后在 BladePass 段测试
fm = []
for f in pass_frames:
    fg = mog.apply(f, learningRate=0)
    fm.append((fg > 200).mean())
print(f"MOG2 warm-up 后 BladePass 段: motion_area mean={np.mean(fm)*100:.2f}%")

# 保存关键帧对比
os.makedirs(os.path.join(OUT, "check_frames"), exist_ok=True)
for name, f, no in [("idle", idle_frames[0], idle_nos[0]), ("blade", pass_frames[40], pass_nos[40])]:
    cv2.imwrite(os.path.join(OUT, "check_frames", f"{name}_{no}.png"), f)
print("saved check_frames")
