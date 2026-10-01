#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断: 25fps 连续帧上 tower ROI 的真实帧间位移 (区分相机振动 vs Blade运动假象)"""
import av, numpy as np, cv2, os

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
TOWER_ROI = (350, 550, 200, 600)

container = av.open(VIDEO)
stream = container.streams.video[0]
stream.thread_type = "AUTO"

# 取两段: 一段 IDLE (frame 7200~7250), 一段含 Blade Pass (frame 1610~1660)
def measure(f0, f1, label):
    frames = {}
    want = set(range(f0, f1))
    for idx, frame in enumerate(container.decode(stream)):
        if idx in want:
            frames[idx] = frame.to_ndarray(format="gray")
            want.discard(idx)
        if not want:
            break
    order = sorted(frames.keys())
    disp = []
    for k in order[1:]:
        prev_roi = frames[order[0]][TOWER_ROI[1]:TOWER_ROI[1]+TOWER_ROI[3], TOWER_ROI[0]:TOWER_ROI[0]+TOWER_ROI[2]]
        cur_roi = frames[k][TOWER_ROI[1]:TOWER_ROI[1]+TOWER_ROI[3], TOWER_ROI[0]:TOWER_ROI[0]+TOWER_ROI[2]]
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        try:
            _, w = cv2.findTransformECC(prev_roi, cur_roi, warp, cv2.MOTION_TRANSLATION, crit)
            disp.append(np.hypot(w[0, 2], w[1, 2]))
        except cv2.error:
            disp.append(np.nan)
    d = np.array(disp)
    print(f"{label}: {len(d)} 相邻帧对, tower位移 mean={np.nanmean(d):.3f}px p95={np.nanpercentile(d,95):.3f}px max={np.nanmax(d):.3f}px")

# 注意: 需要重新打开 container (decode 是单向的)
container.close()
container = av.open(VIDEO); stream = container.streams.video[0]; stream.thread_type = "AUTO"
measure(7200, 7250, "IDLE段 frame7200~7250")
container.close()
container = av.open(VIDEO); stream = container.streams.video[0]; stream.thread_type = "AUTO"
measure(1610, 1660, "BladePass段 frame1610~1660")
container.close()
