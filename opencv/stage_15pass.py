#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扩展验证: 15 个 Blade Pass 的 R_static 稳定性"""
import av, numpy as np, cv2, os, csv, subprocess

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"
TOWER_X = 434.0; H, W = 1440, 2560

# 1. 读 motion timeline, 找 Blade 峰
rows = list(csv.reader(open(os.path.join(OUT, "motion_timeline.csv"))))
tl = np.array([[int(float(r[0])), float(r[2])] for r in rows[1:]])  # [frame_idx, diff]
diff = tl[:, 1]

# 找局部最大峰 (diff>5, 间隔>25帧=1s)
peaks = []
i = 1
while i < len(diff) - 1:
    if diff[i] > 5 and diff[i] >= diff[i-1] and diff[i] >= diff[i+1]:
        peaks.append(int(tl[i, 0]))
        i += 25  # 跳过 1 秒
    else:
        i += 1
print(f"找到 {len(peaks)} 个 Blade 峰 (diff>5 局部最大):")
print(f"  frame: {peaks}")

# 取前 15 个
peaks = peaks[:15]
print(f"取前 15 个峰验证")

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

# 2. 对每个峰提取峰值帧, 跑 EdgesSubPix + 帧间差分 motion mask, 算 R_static
os.makedirs(os.path.join(OUT, "pass_verify"), exist_ok=True)
container = av.open(VIDEO); stream = container.streams.video[0]; stream.thread_type = "AUTO"

# 需要提取: 每个峰的峰值帧 + 前一帧 (帧间差分用)
need = set()
for p in peaks:
    need.add(p); need.add(p - 1)
frame_cache = {}
for idx, frame in enumerate(container.decode(stream)):
    if idx in need:
        frame_cache[idx] = frame.to_ndarray(format="gray")
        need.discard(idx)
    if not need:
        break
container.close()

results = []
for p in peaks:
    if p not in frame_cache or p-1 not in frame_cache:
        continue
    cur = frame_cache[p]; prev = frame_cache[p-1]
    # 帧间差分 motion mask
    d = np.abs(cur.astype(float) - prev.astype(float))
    mm = (d > 15).astype(np.uint8) * 255
    mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    mm = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
    # EdgesSubPix
    img_path = os.path.join(OUT, "pass_verify", f"p{p}.png")
    cv2.imwrite(img_path, cur)
    yml = os.path.join(OUT, "pass_verify", f"p{p}.yml")
    run_edge(img_path, yml)
    pts = read_edges(yml)
    n_raw = len(pts)
    in_tower = np.abs(pts[:, 0] - TOWER_X) < 40
    xi = np.clip(pts[:, 1].astype(int), 0, H-1); yi = np.clip(pts[:, 0].astype(int), 0, W-1)
    in_motion = mm[xi, yi] > 0
    n_tower = int(in_tower.sum())
    n_blade = int((in_motion & ~in_tower).sum())
    n_static_kept = int((~in_tower & ~in_motion).sum())
    n_unrelated = n_raw - n_tower
    r_static = 1 - n_static_kept / max(n_unrelated, 1)
    results.append({"peak_frame": p, "n_raw": n_raw, "n_tower": n_tower,
                    "n_blade": n_blade, "R_static": round(r_static, 4)})
    print(f"peak frame {p}: raw={n_raw} tower={n_tower} blade={n_blade} R_static={r_static*100:.1f}%")

# 3. 汇总
if results:
    with open(os.path.join(OUT, "pass_verify_rstatic.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    rs = np.array([r["R_static"] for r in results])
    nb = np.array([r["n_blade"] for r in results])
    print(f"\n=== 15 Blade Pass 汇总 ===")
    print(f"R_static: mean={rs.mean()*100:.1f}%  min={rs.min()*100:.1f}%  p50={np.median(rs)*100:.1f}%")
    print(f"Blade edge: mean={nb.mean():.0f}  min={nb.min()}  (连续检测率 {np.mean(nb>1000)*100:.0f}%)")
