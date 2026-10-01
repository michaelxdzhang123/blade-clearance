#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Blade-pass 状态机: 帧间差分触发 + 时序外推 -> 连续 Blade edge"""
import av, numpy as np, cv2, os, csv, subprocess

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"
TOWER_X = 434.0; H, W = 1440, 2560

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

# Blade Pass 窗口
frames, nos = extract(1600, 1700)
print(f"窗口 {len(frames)} 帧 (frame {nos[0]}~{nos[-1]})")

# 状态机
IDLE, TRACK = 0, 1
state = IDLE
blade_pos = None
velocity = np.array([0.0, 0.0])
T = 15
trigger_area = 0.02  # 2%
track_area = 0.01    # 1%
exit_area = 0.003    # 0.3%
exit_count = 0
local_dir = os.path.join(OUT, "state_machine_crops")
os.makedirs(local_dir, exist_ok=True)
log = []

for i in range(1, len(frames)):
    diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
    area = (diff > T).mean()
    ys, xs = np.where(diff > T)
    has_motion = len(xs) > 50

    if state == IDLE:
        if area > trigger_area and has_motion:
            state = TRACK
            blade_pos = np.array([xs.mean(), ys.mean()])
            velocity = np.array([0.0, 0.0])
            exit_count = 0
            log.append((nos[i], "TRIGGER", area, blade_pos[0], blade_pos[1]))
        else:
            continue

    if state == TRACK:
        if area > track_area and has_motion:
            # 有运动: 更新位置 + 速度
            new_pos = np.array([xs.mean(), ys.mean()])
            velocity = new_pos - blade_pos
            blade_pos = new_pos
            exit_count = 0
            src = "TRACK"
        else:
            # 运动减弱: 时序外推
            blade_pos = blade_pos + velocity
            exit_count += 1
            src = "EXTRAP"
        # 在 blade_pos 附近提取 Local EdgesSubPix
        bx = int(np.clip(blade_pos[0], 50, W-50))
        by = int(np.clip(blade_pos[1], 50, H-50))
        x0, x1 = max(0, bx-80), min(W, bx+80)
        y0, y1 = max(0, by-80), min(H, by+80)
        crop = frames[i][y0:y1, x0:x1]
        img = os.path.join(local_dir, f"b{nos[i]}.png")
        cv2.imwrite(img, crop)
        yml = os.path.join(local_dir, f"b{nos[i]}.yml")
        run_edge(img, yml)
        pts = read_edges(yml)
        log.append((nos[i], src, area, blade_pos[0], blade_pos[1], len(pts)))
        # 退出判断
        if exit_count >= 5:
            state = IDLE
            log.append((nos[i], "EXIT", area, blade_pos[0], blade_pos[1]))
            print(f"  -> Blade Pass 结束 (frame {nos[i]}, 外推 {exit_count} 帧后退出)")

print("\n=== 状态机结果 ===")
trig = [l for l in log if l[1] == "TRIGGER"]
extrap = [l for l in log if l[1] == "EXTRAP"]
track = [l for l in log if l[1] == "TRACK"]
print(f"TRIGGER 帧: {len(trig)}")
print(f"TRACK 帧 (有运动): {len(track)}")
print(f"EXTRAP 帧 (时序外推): {len(extrap)}")
print(f"总 Blade edge 帧: {len(trig)+len(track)+len(extrap)}")

# 连续 Blade edge 验证
edge_counts = [l[5] for l in log if len(l) > 5 and l[1] in ("TRACK", "EXTRAP", "TRIGGER")]
if edge_counts:
    edge_counts = np.array(edge_counts)
    print(f"Blade edge 点数: mean={edge_counts.mean():.0f} 连续帧(>100pts)={np.mean(edge_counts>100)*100:.0f}%")

with open(os.path.join(OUT, "state_machine_log.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "state", "motion_area", "blade_x", "blade_y", "edge_points"])
    for l in log:
        w.writerow(l[:6] if len(l) >= 6 else list(l) + [""])
print("saved state_machine_log.csv")
