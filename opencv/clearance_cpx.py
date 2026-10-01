#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ACTION-6: Cpx 计算 - measured-frame-only, EXTRAP 排除"""
import av, numpy as np, cv2, os, csv, subprocess

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"
TOWER_X = 434.0; H, W = 1440, 2560
TOWER_EDGE_X = 437.0  # band=10 测得的 tower 测量边

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

frames, nos = extract(1610, 1680)
print(f"Blade Pass 段 {len(frames)} 帧")

os.makedirs(os.path.join(OUT, "cpx_verify"), exist_ok=True)
rows = []
pass_id = 1
for i in range(1, len(frames)):
    diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
    area = (diff > 15).mean()
    # 状态: TRACK (area>1%) vs EXTRAP
    if area > 0.01:
        state = "TRACK"
        # 跑 EdgesSubPix (全图, 用于提取 Blade edge)
        img_path = os.path.join(OUT, "cpx_verify", f"f{nos[i]}.png")
        cv2.imwrite(img_path, frames[i])
        yml = os.path.join(OUT, "cpx_verify", f"f{nos[i]}.yml")
        run_edge(img_path, yml)
        pts = read_edges(yml)
        # Blade edge = motion band 内 edge (排除 tower band x∈[414,454])
        xi = np.clip(pts[:, 1].astype(int), 0, H-1); yi = np.clip(pts[:, 0].astype(int), 0, W-1)
        mm = (diff > 15).astype(np.uint8)
        mm = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
        in_motion = mm[xi, yi] > 0
        in_tower = np.abs(pts[:, 0] - TOWER_X) < 20
        blade_pts = pts[in_motion & ~in_tower]
        if len(blade_pts) > 10:
            # Cpx = min |blade_x - tower_edge_x| (水平距离)
            cpx = float(np.min(np.abs(blade_pts[:, 0] - TOWER_EDGE_X)))
            valid = "true"
        else:
            cpx = np.nan
            valid = "false"
        rows.append({"frame_id": nos[i], "pass_id": pass_id, "state": state,
                     "detected": "detected", "cpx": cpx if np.isfinite(cpx) else "", "valid": valid})
    else:
        rows.append({"frame_id": nos[i], "pass_id": pass_id, "state": "EXTRAP",
                     "detected": "extrapolated", "cpx": "", "valid": "false"})

# 汇总
valid_rows = [r for r in rows if r["valid"] == "true"]
detected = [r for r in rows if r["detected"] == "detected"]
cpx_vals = [r["cpx"] for r in valid_rows]
print(f"\n=== Cpx 结果 ===")
print(f"总帧: {len(rows)}, 检测帧: {len(detected)}, 有效 Cpx 帧: {len(valid_rows)}")
if cpx_vals:
    cpx_arr = np.array(cpx_vals)
    print(f"Cpx: min={cpx_arr.min():.1f}px  median={np.median(cpx_arr):.1f}px  max={cpx_arr.max():.1f}px")
    print(f"Cpx valid rate = {len(valid_rows)/max(len(detected),1)*100:.1f}% (检测帧中)")

with open(os.path.join(OUT, "clearance_px.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["frame_id", "pass_id", "state", "detected", "cpx", "valid"])
    w.writeheader(); w.writerows(rows)
print("saved clearance_px.csv")
