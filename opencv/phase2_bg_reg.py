#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase2: W0(IDLE)帧提取 + median背景 + Registration before/after residual"""
import av, numpy as np, csv, os

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
os.makedirs(os.path.join(OUT, "w0_frames"), exist_ok=True)

# 1. 读 motion timeline, 找 diff<0.3 的帧号
rows = list(csv.reader(open(os.path.join(OUT, "motion_timeline.csv"))))
timeline = np.array([[int(float(r[0])), float(r[1]), float(r[2])] for r in rows[1:]])
idle_idx = timeline[timeline[:, 2] < 0.3][:, 0].astype(int)
print(f"diff<0.3 的 IDLE 帧: {len(idle_idx)} 帧")

# 均匀采样 50 帧做背景
n_bg = 50
sample_idx = idle_idx[np.linspace(0, len(idle_idx)-1, n_bg).astype(int)]
print(f"均匀采样 {n_bg} 帧做背景: frame {sample_idx.min()}~{sample_idx.max()}")

# 2. 流式读视频, 提取 sample 帧
container = av.open(VIDEO)
stream = container.streams.video[0]
stream.thread_type = "AUTO"
want = set(sample_idx.tolist())
frames = {}
for idx, frame in enumerate(container.decode(stream)):
    if idx in want:
        frames[idx] = frame.to_ndarray(format="gray")
        want.discard(idx)
    if not want:
        break
container.close()
print(f"提取 {len(frames)} 帧背景帧")

# 3. median 背景
bg = np.median(np.stack(list(frames.values())), axis=0).astype(np.uint8)
cv2 = None
try:
    import cv2 as _cv2; cv2 = _cv2
except ImportError:
    pass
np.save(os.path.join(OUT, "bg_idle_median.npy"), bg)
if cv2:
    cv2.imwrite(os.path.join(OUT, "bg_idle_median.png"), bg)
print(f"median 背景保存: shape={bg.shape}")

# 4. Registration before/after residual (在 W0 帧上)
TOWER_ROI = (350, 550, 200, 600)  # x,y,w,h
# 提取连续一段 W0 帧做 registration 评估
# 选 diff<0.3 的最长连续段
idle_mask = timeline[:, 2] < 0.3
s = None; best = (0, 0, 0)
for i, m in enumerate(idle_mask):
    if m and s is None: s = i
    elif not m and s is not None:
        if i - s > best[2]: best = (s, i, i-s)
        s = None
f0, f1, L = best
print(f"\n最长连续 IDLE 段: timeline[{f0}~{f1}], {L} 帧 (t={timeline[f0,1]:.1f}~{timeline[f1-1,1]:.1f}s)")

# 提取这段的帧号 (对应视频帧号)
seg_frame_nos = timeline[f0:f1, 0].astype(int)
print(f"视频帧号: {seg_frame_nos.min()}~{seg_frame_nos.max()}")

# 流式提取这段帧
container = av.open(VIDEO)
stream = container.streams.video[0]
stream.thread_type = "AUTO"
want = set(seg_frame_nos.tolist())
seg = {}
for idx, frame in enumerate(container.decode(stream)):
    if idx in want:
        seg[idx] = frame.to_ndarray(format="gray")
        want.discard(idx)
    if not want:
        break
container.close()

# 对齐: 用第一帧的 tower ROI 做参考
order = sorted(seg.keys())
ref = seg[order[0]][TOWER_ROI[1]:TOWER_ROI[1]+TOWER_ROI[3], TOWER_ROI[0]:TOWER_ROI[0]+TOWER_ROI[2]]
res_before = []
res_after = []
import cv2
for k in order[1:]:
    g = seg[k]
    roi = g[TOWER_ROI[1]:TOWER_ROI[1]+TOWER_ROI[3], TOWER_ROI[0]:TOWER_ROI[0]+TOWER_ROI[2]]
    # before: 直接 diff
    res_before.append(float(np.abs(roi.astype(float) - ref.astype(float)).mean()))
    # after: ECC 对齐后 diff
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        _, w = cv2.findTransformECC(ref, roi, warp, cv2.MOTION_TRANSLATION, crit)
        aligned = cv2.warpAffine(roi, w, (roi.shape[1], roi.shape[0]))
        res_after.append(float(np.abs(aligned.astype(float) - ref.astype(float)).mean()))
    except cv2.error:
        res_after.append(res_before[-1])

res_before = np.array(res_before); res_after = np.array(res_after)
R_reg = 1 - res_after.mean() / res_before.mean()
print(f"\n=== Registration before/after residual (IDLE 段, {L}帧) ===")
print(f"E_before mean={res_before.mean():.3f}  E_after mean={res_after.mean():.3f}")
print(f"R_reg = 1 - E_after/E_before = {R_reg*100:.1f}%")
print(f"E_after p95={np.percentile(res_after,95):.3f}")

with open(os.path.join(OUT, "registration_before_after.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame", "res_before", "res_after"])
    for i, k in enumerate(order[1:]):
        w.writerow([k, res_before[i], res_after[i]])
print("saved registration_before_after.csv")
