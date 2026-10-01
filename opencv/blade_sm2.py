#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Blade-pass 状态机 v2: motion mask 外接框定位 + 时序外推 -> 连续 Blade edge"""
import av, numpy as np, cv2, os, csv, subprocess

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"
H, W = 1440, 2560

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

def read_edges(yml):
    fs = cv2.FileStorage(yml, cv2.FILE_STORAGE_READ)
    cn = fs.getNode("contours"); n = cn.size()
    xs, ys = [], []
    for i in range(n):
        c = cn.at(i); pn = c.getNode("points"); m = pn.size()
        pts = np.array([pn.at(k).real() for k in range(m)]).reshape(-1, 2)
        xs.append(pts[:, 0]); ys.append(pts[:, 1])
    fs.release()
    if not xs: return np.zeros((0, 2))
    return np.column_stack([np.concatenate(xs), np.concatenate(ys)])

def run_edge(img_path, yml):
    subprocess.run([EDGE_BIN, img_path, yml.replace(".yml", "_o.jpg"),
                    f"--data={yml}", "--alpha=1.0", "--low=20", "--high=40", "--mode=1"],
                   capture_output=True, timeout=120)

frames, nos = extract(1600, 1700)
print(f"窗口 {len(frames)} 帧")

# 状态机 v2: 用 motion mask 外接框
IDLE, TRACK = 0, 1
state = IDLE
bbox = None  # (x0, y0, x1, y1)
vel = None
T = 15
local_dir = os.path.join(OUT, "sm2_crops")
os.makedirs(local_dir, exist_ok=True)
log = []

for i in range(1, len(frames)):
    diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
    area = (diff > T).mean()
    ys, xs = np.where(diff > T)

    if state == IDLE:
        if area > 0.02 and len(xs) > 100:
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            state = TRACK
            bbox = (x0, y0, x1, y1)
            vel = np.array([0.0, 0.0])
            log.append((nos[i], "TRIGGER", area, x0, y0, x1, y1))
        else:
            continue

    if state == TRACK:
        if area > 0.015 and len(xs) > 100:
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            if bbox:
                vel = np.array([x0 - bbox[0], y0 - bbox[1]])
            bbox = (x0, y0, x1, y1)
            src = "TRACK"
        else:
            # 时序外推
            if bbox and vel is not None:
                bbox = (bbox[0]+vel[0], bbox[1]+vel[1], bbox[2]+vel[0], bbox[3]+vel[1])
            src = "EXTRAP"
        # 在 bbox 内 (+margin) 做 Local EdgesSubPix
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0 = max(0, x0-30); x1 = min(W, x1+30); y0 = max(0, y0-30); y1 = min(H, y1+30)
        if x1 - x0 < 20 or y1 - y0 < 20:
            x0, x1 = max(0, x0-100), min(W, x1+100); y0, y1 = max(0, y0-100), min(H, y1+100)
        crop = frames[i][y0:y1, x0:x1]
        img = os.path.join(local_dir, f"b{nos[i]}.png")
        cv2.imwrite(img, crop)
        yml = os.path.join(local_dir, f"b{nos[i]}.yml")
        run_edge(img, yml)
        pts = read_edges(yml)
        log.append((nos[i], src, area, x0, y0, x1, y1, len(pts)))
        # 退出
        if area < 0.01:
            state = IDLE
            log.append((nos[i], "EXIT", area, 0, 0, 0, 0))

print("\n=== 状态机 v2 结果 ===")
trig = [l for l in log if l[1] == "TRIGGER"]
track = [l for l in log if l[1] == "TRACK"]
extrap = [l for l in log if l[1] == "EXTRAP"]
print(f"TRIGGER: {len(trig)}  TRACK: {len(track)}  EXTRAP: {len(extrap)}")
edge = [l[7] for l in log if len(l) > 7 and l[1] in ("TRACK", "EXTRAP")]
if edge:
    e = np.array(edge)
    print(f"Blade edge 点数: mean={e.mean():.0f}  连续帧(>200pts)={np.mean(e>200)*100:.0f}%  (>50pts)={np.mean(e>50)*100:.0f}%")

with open(os.path.join(OUT, "state_machine_v2_log.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "state", "motion_area", "x0", "y0", "x1", "y1", "edge_points"])
    for l in log:
        w.writerow(l)
print("saved state_machine_v2_log.csv")
