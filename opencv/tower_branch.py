#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ACTION-4: Tower Branch 完整运行 - band扫描 + Local EdgesSubPix + 测量边锁定 + 拟合"""
import av, numpy as np, cv2, os, csv, subprocess

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"
TOWER_X = 434.0; H, W = 1440, 2560
TOWER_Y0, TOWER_Y1 = 550, 1150  # tower 可见范围

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

# 提取 60 帧 (IDLE 段 + Blade Pass 段混合, 覆盖 tower 稳定性)
frames, nos = extract(1600, 1720)
print(f"提取 {len(frames)} 帧")

os.makedirs(os.path.join(OUT, "tower_crops"), exist_ok=True)

# 对每个 band 测试
print(f"\n{'band':>5} | {'valid_rate':>10} | {'switch':>6} | {'resid_p50':>9} | {'resid_p95':>9} | {'edge_x_median':>13}")
summary = []
for band in [10, 20, 30, 40]:
    x0 = TOWER_X - band; x1 = TOWER_X + band
    valid = 0; switch = 0
    resid_all = []
    prev_side = None
    edge_xs = []
    for i, f in enumerate(frames):
        crop = f[TOWER_Y0:TOWER_Y1, int(x0):int(x1)]
        img = os.path.join(OUT, "tower_crops", f"b{band}_{nos[i]}.png")
        cv2.imwrite(img, crop)
        yml = os.path.join(OUT, "tower_crops", f"b{band}_{nos[i]}.yml")
        run_edge(img, yml)
        pts = read_edges(yml)
        if len(pts) < 50:
            continue
        # 全局 x = TOWER_X + (局部 x - band), 分离左右边
        gx = pts[:, 0] + x0
        # 右半边 (测量侧, Blade 从右来)
        right = pts[pts[:, 0] > band - 2]  # 局部 x > band-2, 即全局 x > 432
        left = pts[pts[:, 0] <= band - 2]
        # 测量边 = 右侧边 (Blade 接近侧)
        if len(right) > 20:
            side = "right"
            edge_x = np.median(gx[pts[:, 0] > band - 2])
        elif len(left) > 20:
            side = "left"
            edge_x = np.median(gx[pts[:, 0] <= band - 2])
        else:
            continue
        # switch 检测
        if prev_side is not None and side != prev_side:
            switch += 1
        prev_side = side
        # 拟合残差: edge 点相对中位数的标准差
        if side == "right":
            resid = np.std(gx[pts[:, 0] > band - 2])
        else:
            resid = np.std(gx[pts[:, 0] <= band - 2])
        resid_all.append(resid)
        edge_xs.append(edge_x)
        if resid < 5.0:  # 残差 < 5px 视为有效
            valid += 1
    total = len(resid_all)
    vr = valid / max(total, 1)
    resid_all = np.array(resid_all)
    edge_xs = np.array(edge_xs)
    print(f"{band:5d} | {vr*100:9.1f}% | {switch:6d} | {np.percentile(resid_all,50):8.2f}px | "
          f"{np.percentile(resid_all,95):8.2f}px | {np.median(edge_xs):13.1f}")
    summary.append({"band": band, "valid_rate": round(vr, 4), "switch": switch,
                    "resid_p50": round(np.percentile(resid_all,50), 3),
                    "resid_p95": round(np.percentile(resid_all,95), 3),
                    "edge_x_median": round(float(np.median(edge_xs)), 2)})

with open(os.path.join(OUT, "tower_metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)
print("\nsaved tower_metrics.csv")
