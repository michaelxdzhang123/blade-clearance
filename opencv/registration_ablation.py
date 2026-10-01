#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ACTION-5: Registration A/B ablation - R0(无注册) vs R1(ECC平移注册)"""
import av, numpy as np, cv2, os, csv

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
TOWER_ROI = (350, 550, 200, 600)  # x, y, w, h

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

# IDLE 段 (frame 7200~7300) + Blade Pass 段 (1610~1680)
idle_f, idle_n = extract(7200, 7300)
blade_f, blade_n = extract(1610, 1680)
print(f"IDLE {len(idle_f)} 帧, BladePass {len(blade_f)} 帧")

def interframe_motion(frames, reg=False):
    """帧间差分 motion_area; reg=True 时先 ECC 注册相邻帧"""
    areas = []
    prev = frames[0]
    for i in range(1, len(frames)):
        cur = frames[i]
        if reg:
            roi_prev = prev[TOWER_ROI[1]:TOWER_ROI[1]+TOWER_ROI[3], TOWER_ROI[0]:TOWER_ROI[0]+TOWER_ROI[2]]
            roi_cur = cur[TOWER_ROI[1]:TOWER_ROI[1]+TOWER_ROI[3], TOWER_ROI[0]:TOWER_ROI[0]+TOWER_ROI[2]]
            warp = np.eye(2, 3, dtype=np.float32)
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            try:
                _, w = cv2.findTransformECC(roi_prev, roi_cur, warp, cv2.MOTION_TRANSLATION, crit)
                cur = cv2.warpAffine(cur, w, (cur.shape[1], cur.shape[0]))
            except cv2.error:
                pass
        d = np.abs(cur.astype(float) - prev.astype(float))
        areas.append((d > 15).mean())
        prev = frames[i]  # 注意: 注册只用于差分, 不改变 prev
    return np.mean(areas)

print("\n=== Registration A/B ablation ===")
r0_idle = interframe_motion(idle_f, reg=False)
r1_idle = interframe_motion(idle_f, reg=True)
r0_blade = interframe_motion(blade_f, reg=False)
r1_blade = interframe_motion(blade_f, reg=True)

print(f"R0 (无注册): IDLE false_motion={r0_idle*100:.3f}%  BladePass motion={r0_blade*100:.3f}%")
print(f"R1 (ECC注册): IDLE false_motion={r1_idle*100:.3f}%  BladePass motion={r1_blade*100:.3f}%")
print(f"IDLE false_motion 改善: {(1-r1_idle/max(r0_idle,1e-9))*100:.1f}%")
print(f"BladePass motion 变化: {(1-r1_blade/max(r0_blade,1e-9))*100:.1f}%")

# 决策
improvement = (1 - r1_idle / max(r0_idle, 1e-9)) * 100
if improvement < 5:
    decision = "REMOVE"
    reason = f"IDLE false_motion 改善仅 {improvement:.1f}% (<5%), Registration 不显著"
else:
    decision = "KEEP"
    reason = f"IDLE false_motion 改善 {improvement:.1f}% (>=5%)"
print(f"\n决策: REGISTRATION = {decision}")
print(f"理由: {reason}")

with open(os.path.join(OUT, "registration_ablation.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "idle_false_motion", "blade_pass_motion"])
    w.writerow(["R0_no_registration", r0_idle, r0_blade])
    w.writerow(["R1_ecc_translation", r1_idle, r1_blade])
    w.writerow(["decision", decision, reason])
print("saved registration_ablation.csv")
