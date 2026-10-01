#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多帧 Cpx 对比可视化: frame 1617/1620/1630, 同参数, 输出对比图 + 汇总"""
import av, numpy as np, cv2, os, subprocess, csv

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"
TOWER_X = 434.0; H, W = 1440, 2560
TOWER_Y0, TOWER_Y1 = 550, 1150
os.makedirs(os.path.join(OUT, "cpx_viz"), exist_ok=True)

def extract_frame(fn):
    container = av.open(VIDEO); stream = container.streams.video[0]; stream.thread_type = "AUTO"
    for idx, frame in enumerate(container.decode(stream)):
        if idx == fn:
            g = frame.to_ndarray(format="gray"); container.close(); return g
    container.close(); return None

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

def dist_to_line(p, vx, vy, x0, y0):
    dx = p[0] - x0; dy = p[1] - y0
    return abs(vx*dy - vy*dx) / max(np.sqrt(vx**2+vy**2), 1e-9)

FRAMES = [1617, 1620, 1625, 1630, 1635]
summary = []

for FN in FRAMES:
    g = extract_frame(FN); g_prev = extract_frame(FN - 1)
    if g is None:
        print(f"frame {FN} 提取失败"); continue

    # EdgesSubPix
    img_path = os.path.join(OUT, "cpx_viz", f"f{FN}.png")
    cv2.imwrite(img_path, g)
    yml = os.path.join(OUT, "cpx_viz", f"f{FN}.yml")
    run_edge(img_path, yml)
    pts = read_edges(yml)

    # Tower fitLine (band=10 右侧)
    band = 10; x0 = TOWER_X - band; x1 = TOWER_X + band
    crop = g[TOWER_Y0:TOWER_Y1, int(x0):int(x1)]
    cv2.imwrite(os.path.join(OUT, "cpx_viz", f"t{FN}.png"), crop)
    run_edge(os.path.join(OUT, "cpx_viz", f"t{FN}.png"), os.path.join(OUT, "cpx_viz", f"t{FN}.yml"))
    tpts = read_edges(os.path.join(OUT, "cpx_viz", f"t{FN}.yml"))
    t_right = tpts[tpts[:, 0] > band - 2]
    if len(t_right) < 20:
        print(f"frame {FN} tower edge 不足"); continue
    t_sel = np.column_stack([t_right[:, 0] + x0, t_right[:, 1] + TOWER_Y0])
    tvx, tvy, tcx, tcy = cv2.fitLine(t_sel, cv2.DIST_L2, 0, 0.01, 0.01)

    # motion mask + blade 候选
    diff = np.abs(g.astype(float) - g_prev.astype(float))
    mm = (diff > 15).astype(np.uint8)
    mm = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
    xi = np.clip(pts[:,1].astype(int),0,H-1); yi = np.clip(pts[:,0].astype(int),0,W-1)
    in_motion = mm[xi,yi] > 0
    in_tower = np.abs(pts[:,0] - TOWER_X) < 20
    blade = pts[in_motion & ~in_tower]

    if len(blade) == 0:
        print(f"frame {FN} blade 候选为空"); continue
    d = np.array([dist_to_line(p, tvx[0], tvy[0], tcx[0], tcy[0]) for p in blade])
    i_near = int(np.argmin(d))
    p_near = blade[i_near]; cpx = d[i_near]
    px, py = p_near
    t = ((px-tcx[0])*tvx[0] + (py-tcy[0])*tvy[0]) / (tvx[0]**2 + tvy[0]**2)
    foot = np.array([tcx[0] + t*tvx[0], tcy[0] + t*tvy[0]])
    motion_area = (diff > 15).mean()

    summary.append({"frame": FN, "motion_area": round(motion_area,4),
                    "blade_candidates": len(blade),
                    "nearest_x": round(p_near[0],1), "nearest_y": round(p_near[1],1),
                    "cpx_px": round(cpx,3)})
    print(f"frame {FN}: motion={motion_area*100:.1f}% blade候选={len(blade)} "
          f"最近点({p_near[0]:.1f},{p_near[1]:.1f}) Cpx={cpx:.2f}px")

    # 画图
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for p in blade:
        cv2.circle(bgr, (int(p[0]), int(p[1])), 1, (0,200,0), -1)
    for bx in [414, 454]:
        cv2.line(bgr, (bx, 0), (bx, H), (128,128,128), 1)
    L = 2000
    p1 = (int(tcx[0]-tvx[0]*L), int(tcy[0]-tvy[0]*L)); p2 = (int(tcx[0]+tvx[0]*L), int(tcy[0]+tvy[0]*L))
    cv2.line(bgr, p1, p2, (0,0,255), 3)
    cv2.circle(bgr, (int(p_near[0]), int(p_near[1])), 8, (0,255,255), -1)
    cv2.circle(bgr, (int(p_near[0]), int(p_near[1])), 12, (0,255,255), 2)
    cv2.circle(bgr, (int(foot[0]), int(foot[1])), 8, (255,0,0), -1)
    cv2.line(bgr, (int(p_near[0]), int(p_near[1])), (int(foot[0]), int(foot[1])), (0,255,255), 3)
    cv2.putText(bgr, f"frame {FN}  Cpx = {cpx:.2f} px", (50,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
    cv2.imwrite(os.path.join(OUT, "cpx_viz", f"cpx_frame{FN}.png"), bgr)

# 汇总
with open(os.path.join(OUT, "cpx_viz", "cpx_compare.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)

print("\n=== Cpx 对比汇总 ===")
print(f"{'frame':>6} | {'motion%':>7} | {'blade候选':>9} | {'最近x':>7} | {'最近y':>7} | {'Cpx(px)':>8}")
for s in summary:
    print(f"{s['frame']:>6} | {s['motion_area']*100:>6.1f}% | {s['blade_candidates']:>9} | "
          f"{s['nearest_x']:>7} | {s['nearest_y']:>7} | {s['cpx_px']:>8.2f}")
