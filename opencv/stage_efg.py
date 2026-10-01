#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage E/F/G 综合验证: 帧间差分 motion mask + Tower/Blade 双分支 + R_static"""
import av, numpy as np, cv2, os, csv, subprocess

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"
TOWER_X = 434.0
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

# 提取 Blade Pass 段 (含 diff>8 峰)
pass_frames, pass_nos = extract(1610, 1680)
print(f"BladePass 段 {len(pass_frames)} 帧 (frame {pass_nos[0]}~{pass_nos[-1]})")

# 对每个帧间差分做 motion mask, 找 Blade 瞬时帧
print("\n=== 帧间差分 motion 检测 (thr=15) ===")
motion_idx = []
for i in range(1, len(pass_frames)):
    d = np.abs(pass_frames[i].astype(float) - pass_frames[i-1].astype(float))
    area = (d > 15).mean()
    if area > 0.05:  # >5% 面积的强运动
        motion_idx.append(i)
print(f"强运动帧 (motion_area>5%): {len(motion_idx)} 帧, 帧号偏移 {motion_idx[:10]}")

# 选 3 个代表帧: 1 IDLE帧 + 2 Blade瞬时帧
rep = []
if motion_idx:
    rep = [motion_idx[len(motion_idx)//2]]  # Blade 帧
rep = [0] + rep  # 加一个 IDLE 参考帧
print(f"\n代表帧: {rep}")

# 对代表帧跑 EdgesSubPix, 分类 edge, 算 R_static
os.makedirs(os.path.join(OUT, "edge_verify"), exist_ok=True)
results = []
for ri in rep:
    no = pass_nos[ri]
    img_path = os.path.join(OUT, "edge_verify", f"f{no}.png")
    cv2.imwrite(img_path, pass_frames[ri])
    yml = os.path.join(OUT, "edge_verify", f"f{no}.yml")
    run_edge(img_path, yml)
    pts = read_edges(yml)
    n_raw = len(pts)
    # 帧间差分 motion mask (用 ri 帧与前帧差分)
    if ri > 0:
        d = np.abs(pass_frames[ri].astype(float) - pass_frames[ri-1].astype(float))
        mm = (d > 15).astype(np.uint8) * 255
        mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        mm = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
    else:
        mm = np.zeros((H, W), np.uint8)
    # 分类
    in_tower = np.abs(pts[:, 0] - TOWER_X) < 40
    xi = np.clip(pts[:, 1].astype(int), 0, H-1); yi = np.clip(pts[:, 0].astype(int), 0, W-1)
    in_motion = mm[xi, yi] > 0
    n_tower = in_tower.sum()
    n_blade = (in_motion & ~in_tower).sum()
    n_static = (~in_tower & ~in_motion).sum()  # 无关静止 edge
    r_static = 1 - n_static / max(n_raw - n_tower, 1)  # 无关静止 edge 被"保留"的比例的反面
    # 更准确: R_static = 被抑制的无关edge / 无关edge原始
    results.append({"frame": no, "n_raw": n_raw, "n_tower": n_tower,
                    "n_blade": n_blade, "n_static_kept": n_static})
    print(f"frame {no}: raw={n_raw} tower={n_tower} blade={n_blade} static_kept={n_static}")

# 汇总
if results:
    with open(os.path.join(OUT, "edge_verify_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print("\n=== 汇总 ===")
    for r in results:
        n_unrelated = r["n_raw"] - r["n_tower"]  # 无关 edge (非 Tower)
        suppressed = n_unrelated - r["n_static_kept"]
        print(f"frame {r['frame']}: 无关edge={n_unrelated}, 抑制={suppressed} ({100*suppressed/max(n_unrelated,1):.1f}%), "
              f"Blade保留={r['n_blade']}")
