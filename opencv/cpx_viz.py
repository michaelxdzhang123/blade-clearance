#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cpx_euclidean 可视化 v3: 与 tower_cpx_v2.py 一致参数 (in_tower<20), 诚实标注"""
import av, numpy as np, cv2, os, subprocess

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

FN = 1617
g = extract_frame(FN); g_prev = extract_frame(FN - 1)

img_path = os.path.join(OUT, "cpx_viz", f"f{FN}.png")
cv2.imwrite(img_path, g)
yml = os.path.join(OUT, "cpx_viz", f"f{FN}.yml")
run_edge(img_path, yml)
pts = read_edges(yml)

# Tower 拟合线 (band=10 右侧边)
band = 10; x0 = TOWER_X - band; x1 = TOWER_X + band
crop = g[TOWER_Y0:TOWER_Y1, int(x0):int(x1)]
cv2.imwrite(os.path.join(OUT, "cpx_viz", "tower_crop.png"), crop)
run_edge(os.path.join(OUT, "cpx_viz", "tower_crop.png"), os.path.join(OUT, "cpx_viz", "tower_crop.yml"))
tpts = read_edges(os.path.join(OUT, "cpx_viz", "tower_crop.yml"))
t_right = tpts[tpts[:, 0] > band - 2]
t_sel = np.column_stack([t_right[:, 0] + x0, t_right[:, 1] + TOWER_Y0])
tvx, tvy, tcx, tcy = cv2.fitLine(t_sel, cv2.DIST_L2, 0, 0.01, 0.01)

# motion mask (diff>15 + dilate21), in_tower<20 (与 tower_cpx_v2 一致)
diff = np.abs(g.astype(float) - g_prev.astype(float))
mm = (diff > 15).astype(np.uint8)
mm = cv2.dilate(mm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)))
xi = np.clip(pts[:,1].astype(int),0,H-1); yi = np.clip(pts[:,0].astype(int),0,W-1)
in_motion = mm[xi,yi] > 0
in_tower = np.abs(pts[:,0] - TOWER_X) < 20
blade = pts[in_motion & ~in_tower]

d = np.array([dist_to_line(p, tvx[0], tvy[0], tcx[0], tcy[0]) for p in blade])
i_near = int(np.argmin(d))
p_near = blade[i_near]; cpx = d[i_near]
px, py = p_near
t = ((px-tcx[0])*tvx[0] + (py-tcy[0])*tvy[0]) / (tvx[0]**2 + tvy[0]**2)
foot = np.array([tcx[0] + t*tvx[0], tcy[0] + t*tvy[0]])
print(f"Blade候选点={len(blade)}, 最近点({p_near[0]:.1f},{p_near[1]:.1f}) 垂足({foot[0]:.1f},{foot[1]:.1f}) Cpx={cpx:.3f}px")

# 画图
bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
for p in blade:
    cv2.circle(bgr, (int(p[0]), int(p[1])), 1, (0,200,0), -1)
# tower band 边界虚线 (x=414 和 x=454)
for bx in [414, 454]:
    cv2.line(bgr, (bx, 0), (bx, H), (128,128,128), 1)
# tower 拟合线
L = 2000
p1 = (int(tcx[0]-tvx[0]*L), int(tcy[0]-tvy[0]*L)); p2 = (int(tcx[0]+tvx[0]*L), int(tcy[0]+tvy[0]*L))
cv2.line(bgr, p1, p2, (0,0,255), 3)
# 最近点 + 垂足 + 垂线
cv2.circle(bgr, (int(p_near[0]), int(p_near[1])), 8, (0,255,255), -1)
cv2.circle(bgr, (int(p_near[0]), int(p_near[1])), 12, (0,255,255), 2)
cv2.circle(bgr, (int(foot[0]), int(foot[1])), 8, (255,0,0), -1)
cv2.line(bgr, (int(p_near[0]), int(p_near[1])), (int(foot[0]), int(foot[1])), (0,255,255), 3)
cv2.putText(bgr, f"Cpx = {cpx:.2f} px (euclidean)", (50,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
cv2.putText(bgr, "Tower fit line (red), band x=414~454 (grey dashed)", (50,110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
cv2.putText(bgr, "Blade candidate = motion-band minus tower-band (green)", (50,145), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2)
cv2.putText(bgr, "nearest (yellow) -> foot (blue)", (50,180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
cv2.imwrite(os.path.join(OUT, "cpx_viz", "cpx_euclidean_full.png"), bgr)

cx, cy = int((p_near[0]+foot[0])/2), int((p_near[1]+foot[1])/2)
z = 500
zx0, zy0 = max(0,cx-z//2), max(0,cy-z//2); zx1, zy1 = min(W,cx+z//2), min(H,cy+z//2)
bgr_z = bgr[zy0:zy1, zx0:zx1].copy()
cv2.putText(bgr_z, f"Cpx = {cpx:.2f} px", (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
cv2.imwrite(os.path.join(OUT, "cpx_viz", "cpx_euclidean_zoom.png"), bgr_z)
print("saved cpx_euclidean_full.png / zoom.png")
