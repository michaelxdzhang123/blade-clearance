#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tower 直线拟合(RANSAC/fitLine) + Cpx 完整欧氏距离 修正"""
import av, numpy as np, cv2, os, csv, subprocess

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"
TOWER_X = 434.0; H, W = 1440, 2560
TOWER_Y0, TOWER_Y1 = 550, 1150

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

def dist_to_line(pts, vx, vy, x0, y0):
    """点到直线的欧氏距离, 直线方向 (vx,vy) 过 (x0,y0)"""
    dx = pts[:, 0] - x0; dy = pts[:, 1] - y0
    cross = np.abs(vx * dy - vy * dx)
    norm = np.sqrt(vx**2 + vy**2)
    return cross / max(norm, 1e-9)

frames, nos = extract(1610, 1680)
print(f"Blade Pass 段 {len(frames)} 帧")

os.makedirs(os.path.join(OUT, "tower_fit_crops"), exist_ok=True)

# 1. Tower 直线拟合 (band=10)
print("\n=== Tower 直线拟合 (band=10, RANSAC/fitLine) ===")
band = 10
x0 = TOWER_X - band; x1 = TOWER_X + band
resid_p50s = []; resid_p95s = []; switch = 0; valid = 0; total = 0
prev_side = None
for i, f in enumerate(frames):
    crop = f[TOWER_Y0:TOWER_Y1, int(x0):int(x1)]
    img = os.path.join(OUT, "tower_fit_crops", f"t{nos[i]}.png")
    cv2.imwrite(img, crop)
    yml = os.path.join(OUT, "tower_fit_crops", f"t{nos[i]}.yml")
    run_edge(img, yml)
    pts = read_edges(yml)
    if len(pts) < 50:
        continue
    total += 1
    gx = pts[:, 0] + x0; gy = pts[:, 1] + TOWER_Y0
    right = pts[pts[:, 0] > band - 2]
    left = pts[pts[:, 0] <= band - 2]
    if len(right) > 20:
        side = "right"; sel = np.column_stack([gx[pts[:, 0] > band-2], gy[pts[:, 0] > band-2]])
    elif len(left) > 20:
        side = "left"; sel = np.column_stack([gx[pts[:, 0] <= band-2], gy[pts[:, 0] <= band-2]])
    else:
        continue
    if prev_side is not None and side != prev_side:
        switch += 1
    prev_side = side
    # fitLine 直线拟合
    vx, vy, cx, cy = cv2.fitLine(sel, cv2.DIST_L2, 0, 0.01, 0.01)
    d = dist_to_line(sel, vx, vy, cx, cy)
    resid_p50s.append(np.percentile(d, 50)); resid_p95s.append(np.percentile(d, 95))
    if np.percentile(d, 95) < 5.0:
        valid += 1

print(f"Tower 直线拟合 (band=10): valid_rate={valid/max(total,1)*100:.1f}%  switch={switch}")
print(f"  fit_residual_p50={np.median(resid_p50s):.3f}px  p95={np.median(resid_p95s):.3f}px (直线拟合后, 原中位数法 2.99px)")

# 2. Cpx 完整欧氏距离 (Blade 点到 Tower 拟合线)
print("\n=== Cpx 完整欧氏距离 (min ||p - q||) ===")
# 先拟合 tower 线 (用第一帧的 tower 边作为参考)
ref_crop = frames[20][TOWER_Y0:TOWER_Y1, int(x0):int(x1)]
cv2.imwrite(os.path.join(OUT, "tower_fit_crops", "ref.png"), ref_crop)
run_edge(os.path.join(OUT, "tower_fit_crops", "ref.png"), os.path.join(OUT, "tower_fit_crops", "ref.yml"))
ref_pts = read_edges(os.path.join(OUT, "tower_fit_crops", "ref.yml"))
ref_right = ref_pts[ref_pts[:, 0] > band-2]
ref_sel = np.column_stack([ref_right[:, 0] + x0, ref_right[:, 1] + TOWER_Y0])
tvx, tvy, tcx, tcy = cv2.fitLine(ref_sel, cv2.DIST_L2, 0, 0.01, 0.01)
print(f"Tower 参考线: 方向({tvx[0]:.4f},{tvy[0]:.4f}) 过({tcx[0]:.1f},{tcy[0]:.1f})")

cpx_rows = []
for i in range(1, len(frames)):
    diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
    area = (diff > 15).mean()
    if area > 0.01:  # TRACK
        img_path = os.path.join(OUT, "tower_fit_crops", f"c{nos[i]}.png")
        cv2.imwrite(img_path, frames[i])
        yml = os.path.join(OUT, "tower_fit_crops", f"c{nos[i]}.yml")
        run_edge(img_path, yml)
        pts = read_edges(yml)
        xi = np.clip(pts[:, 1].astype(int), 0, H-1); yi = np.clip(pts[:, 0].astype(int), 0, W-1)
        mm = (diff > 15).astype(np.uint8)
        mm = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
        in_motion = mm[xi, yi] > 0
        in_tower = np.abs(pts[:, 0] - TOWER_X) < 20
        blade = pts[in_motion & ~in_tower]
        if len(blade) > 10:
            d = dist_to_line(blade, tvx, tvy, tcx, tcy)
            cpx = float(np.min(d))
            cpx_rows.append((nos[i], "TRACK", cpx, "true"))
        else:
            cpx_rows.append((nos[i], "TRACK", np.nan, "false"))
    else:
        cpx_rows.append((nos[i], "EXTRAP", np.nan, "false"))

valid_rows = [r for r in cpx_rows if r[3] == "true"]
cpx_vals = np.array([r[2] for r in valid_rows])
print(f"Cpx 欧氏距离: valid {len(valid_rows)} 帧, min={cpx_vals.min():.2f}px median={np.median(cpx_vals):.2f}px max={cpx_vals.max():.2f}px")

with open(os.path.join(OUT, "tower_cpx_v2.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_id", "state", "cpx_euclidean", "valid"])
    w.writerows(cpx_rows)
print("saved tower_cpx_v2.csv")
