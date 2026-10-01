#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ACTION-8: 生成 06_blade_tower_final_overlay.mp4 + 07_cpx_overlay.mp4"""
import av, numpy as np, cv2, os, subprocess

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"
TOWER_X = 434.0; TOWER_EDGE_X = 437.0; H, W = 1440, 2560

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

tmp = os.path.join(OUT, "video_tmp")
os.makedirs(tmp, exist_ok=True)

# 生成两个视频的帧
v06_frames = []
v07_frames = []
for i in range(1, len(frames)):
    g = frames[i]
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
    mm = (diff > 15).astype(np.uint8)
    mm_d = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
    # 跑 EdgesSubPix
    img_path = os.path.join(tmp, f"f{nos[i]}.png")
    cv2.imwrite(img_path, g)
    yml = os.path.join(tmp, f"f{nos[i]}.yml")
    run_edge(img_path, yml)
    pts = read_edges(yml)
    xi = np.clip(pts[:, 1].astype(int), 0, H-1); yi = np.clip(pts[:, 0].astype(int), 0, W-1)
    in_motion = mm_d[xi, yi] > 0
    in_tower = np.abs(pts[:, 0] - TOWER_X) < 10
    blade = pts[in_motion & ~in_tower]
    tower = pts[in_tower]
    # 06: blade(绿) + tower(红)
    for p in blade:
        cv2.circle(bgr, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
    for p in tower:
        cv2.circle(bgr, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)
    v06_frames.append(bgr.copy())
    # 07: Cpx 距离线
    if len(blade) > 10:
        nearest = blade[np.argmin(np.abs(blade[:, 0] - TOWER_EDGE_X))]
        cv2.line(bgr, (int(TOWER_EDGE_X), int(nearest[1])), (int(nearest[0]), int(nearest[1])),
                 (255, 255, 0), 2)
        cv2.putText(bgr, f"Cpx={abs(nearest[0]-TOWER_EDGE_X):.1f}px", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    v07_frames.append(bgr.copy())

# 写视频
def write_video(frames, out_name):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(os.path.join(OUT, out_name), cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"{out_name}: {len(frames)} 帧")

write_video(v06_frames, "06_blade_tower_final_overlay.mp4")
write_video(v07_frames, "07_cpx_overlay.mp4")
