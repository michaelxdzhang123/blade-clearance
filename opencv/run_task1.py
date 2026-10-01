#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_task1.py — 方案1-任务书-1 主脚本

Stage A~K 编排, 输出 stats/videos/benchmark 到 outputs/TASK1_EDGE_MOTION_MASK/
"""
import os, sys, time, csv, argparse, json
import numpy as np
import cv2
from opencv.pipeline import (run_edgesubpix, read_edges, border_remove,
                          register_ecc, static_frame_diff, motion_mask,
                          morphology, dilate_mask, mask_points,
                          tower_band_points, tower_orientation_gate,
                          pca_elongation, nearest_pair)
from opencv import stages_extra as se

EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs", "TASK1_EDGE_MOTION_MASK")
FRAMES_DIR = os.path.join(OUT, "frames")
RESULTS = os.path.join(OUT, "results")
VIDEOS = os.path.join(OUT, "videos")

# ---- 画面几何 (基于初始分析) ----
W, H = 2560, 1440
TOWER_X = 434.0            # 塔筒参考 x (两条垂直边缘 398/469 的中点)
TOWER_ROI = (350, 550, 200, 600)   # registration 用塔筒 ROI (x,y,w,h)
TRACK_ROI = (300, 500, 1100, 750)  # tracking ROI (x,y,w,h)

ALPHA, LOW, HIGH, MODE = 1.0, 20, 40, 1


def frame_paths():
    fs = sorted(os.listdir(FRAMES_DIR))
    return [os.path.join(FRAMES_DIR, f) for f in fs if f.endswith(".png")]


def frame_ids():
    return [os.path.basename(p)[:-4] for p in frame_paths()]


def load_gray(fp):
    return cv2.imread(fp, cv2.IMREAD_GRAYSCALE)


def save_video(png_list, out_mp4, fps=8):
    """把一组 overlay PNG 合成 mp4"""
    if not png_list:
        return
    first = cv2.imread(png_list[0])
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in png_list:
        vw.write(cv2.imread(p))
    vw.release()


def load_registration():
    """读 Stage C 的每帧平移量 (dx, dy)"""
    import csv as _csv
    reg = {}
    fp = os.path.join(RESULTS, "C_registration_stats.csv")
    if os.path.exists(fp):
        with open(fp) as f:
            for r in _csv.DictReader(f):
                reg[int(r["frame"])] = (float(r["dx"]), float(r["dy"]))
    return reg


def align_frame(g, dx, dy):
    """把当前帧对齐到参考帧坐标系 (消除相机振动)"""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(g, M, (W, H))


def compute_bg(grays):
    """mean 背景 (流式累加, O(HxW) 内存, 代替 median 避免 uint8->float64 OOM)"""
    acc = np.zeros(grays[0].shape, dtype=np.float64)
    for g in grays:
        acc += g
    return (acc / len(grays)).astype(np.uint8)


# ============================================================
# Stage C: Registration (塔筒 ROI 上 ECC 平移)
# ============================================================
def stage_c_registration(ids, grays):
    ref = grays[0][TOWER_ROI[1]:TOWER_ROI[1]+TOWER_ROI[3],
                   TOWER_ROI[0]:TOWER_ROI[0]+TOWER_ROI[2]]
    rows = []
    for i, (fid, g) in enumerate(zip(ids, grays)):
        roi = g[TOWER_ROI[1]:TOWER_ROI[1]+TOWER_ROI[3],
                TOWER_ROI[0]:TOWER_ROI[0]+TOWER_ROI[2]]
        w = register_ecc(ref, roi, cv2.MOTION_TRANSLATION)
        dx, dy = w[0, 2], w[1, 2]
        rows.append({"frame": i, "frame_id": fid, "dx": float(dx), "dy": float(dy),
                     "mag_px": float(np.hypot(dx, dy))})
    with open(os.path.join(RESULTS, "C_registration_stats.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    mags = np.array([r["mag_px"] for r in rows])
    print(f"[Stage C] Registration: 平移量 mean={mags.mean():.3f}px p95={np.percentile(mags,95):.3f}px max={mags.max():.3f}px")
    return rows


# ============================================================
# Stage D/E/F: Background -> Motion Mask -> 与 Raw Edge 相交
# ============================================================
def stage_d_e_f(ids, grays, bg, reg, bg_method="static", motion_thr=20, dilate_px=10):
    n = len(grays)
    # 背景: 预计算的 mean (static) 或 MOG2
    # MOG2
    mog = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=False)
    rows = []
    overlay_dir = os.path.join(OUT, "frames", "motion_overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    for i, (fid, g) in enumerate(zip(ids, grays)):
        dx, dy = reg.get(i, (0.0, 0.0))
        g_aligned = align_frame(g, dx, dy)
        if bg_method == "static":
            diff = static_frame_diff(bg, g_aligned)
        else:
            diff = mog.apply(g_aligned)
        mm = motion_mask(diff, motion_thr)
        mm = morphology(mm, op="close", ksize=5, iters=1)
        mm_band = dilate_mask(mm, dilate_px)
        # 读 raw edge 并相交
        yml = os.path.join(RESULTS, f"raw_{fid}.yml")
        xs, ys, resp, direc, n_cont = read_edges(yml)
        pts = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2))
        blade_cand, keep = mask_points(pts, mm_band, H, W)
        rows.append({
            "frame": i, "frame_id": fid,
            "raw_edge_points": len(pts),
            "motion_filtered_edge_points": len(blade_cand),
            "reduction_ratio": 1 - len(blade_cand)/max(len(pts), 1),
            "motion_mask_area_ratio": float((mm_band > 0).mean()),
        })
        # overlay (采样第 1/每10帧 存 PNG 供视频)
        if i % 10 == 0 or i < 3:
            color = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            color[mm_band > 0] = (0, 0, 255)  # 红=motion band
            cv2.imwrite(os.path.join(overlay_dir, f"mo_{fid}.png"), color)
    with open(os.path.join(RESULTS, f"F_motion_filter_stats_{bg_method}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    reds = np.array([r["reduction_ratio"] for r in rows])
    print(f"[Stage D/E/F] ({bg_method}) R_reduce: mean={reds.mean()*100:.1f}% p50={np.percentile(reds,50)*100:.1f}% min={reds.min()*100:.1f}%")
    return rows


# ============================================================
# Stage G: Tower Reference Band
# ============================================================
def stage_g_tower(ids, band_px=20, tol_deg=7):
    rows = []
    for i, fid in enumerate(ids):
        yml = os.path.join(RESULTS, f"raw_{fid}.yml")
        xs, ys, resp, direc, n_cont = read_edges(yml)
        pts = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2))
        t_pts, keep = tower_band_points(pts, TOWER_X, band_px, H, W)
        # orientation gate: 塔筒边缘垂直 -> 梯度方向水平 (0 或 pi)
        if len(t_pts):
            # 用梯度方向判断 (塔筒边缘的梯度接近水平)
            g_keep = (np.abs(direc[keep] - 0) < np.radians(tol_deg)) | \
                     (np.abs(direc[keep] - np.pi) < np.radians(tol_deg)) | \
                     (np.abs(direc[keep] - 2*np.pi) < np.radians(tol_deg))
            t_final = t_pts[g_keep]
        else:
            t_final = t_pts
        rows.append({"frame": i, "frame_id": fid,
                     "tower_band_points": len(t_pts),
                     "tower_orientation_points": len(t_final)})
    with open(os.path.join(RESULTS, "G_tower_edge_stats.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    nb = np.array([r["tower_band_points"] for r in rows])
    print(f"[Stage G] Tower band={band_px}px: mean={nb.mean():.0f} pts/frame (连续性={np.mean(nb>0)*100:.0f}% 帧有塔筒点)")
    return rows


# ============================================================
# Stage K: Nearest Point (Blade vs Tower)
# ============================================================
def stage_k_nearest(ids, grays, bg, reg, band_px=20, dilate_px=10, motion_thr=20):
    n = len(ids)
    rows = []
    for i, (fid, g) in enumerate(zip(ids, grays)):
        yml = os.path.join(RESULTS, f"raw_{fid}.yml")
        xs, ys, resp, direc, n_cont = read_edges(yml)
        pts = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2))
        # tower pts
        t_pts, _ = tower_band_points(pts, TOWER_X, band_px, H, W)
        # blade pts = motion mask 内的 edge (排除 tower band)
        dx, dy = reg.get(i, (0.0, 0.0))
        g_aligned = align_frame(g, dx, dy)
        diff = static_frame_diff(bg, g_aligned)
        mm = dilate_mask(morphology(motion_mask(diff, motion_thr), op="close", ksize=5, iters=1), dilate_px)
        b_pts, _ = mask_points(pts, mm, H, W)
        # 排除 tower band 内的点, 避免 blade=tower
        if len(b_pts):
            b_pts = b_pts[np.abs(b_pts[:, 0] - TOWER_X) > band_px]
        c_px, bp, tp = nearest_pair(b_pts, t_pts)
        rows.append({"frame": i, "frame_id": fid, "c_px": c_px,
                     "n_blade": len(b_pts), "n_tower": len(t_pts)})
    with open(os.path.join(RESULTS, "clearance_px.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    valid = [r for r in rows if np.isfinite(r["c_px"])]
    if valid:
        cs = [r["c_px"] for r in valid]
        print(f"[Stage K] Nearest Cpx: 有效 {len(valid)}/{len(rows)} 帧, mean={np.mean(cs):.1f}px p50={np.percentile(cs,50):.1f}px min={np.min(cs):.1f}px")
    else:
        print("[Stage K] 无有效 Cpx (需检查 blade/tower 检测)")
    return rows


# ============================================================
# 主编排
# ============================================================
def baseline():
    paths = frame_paths()
    print(f"[Baseline] 处理 {len(paths)} 帧全图 EdgesSubPix ...")
    rows = []
    t0 = time.time()
    for idx, fp in enumerate(paths):
        fid = os.path.basename(fp)[:-4]
        yml = os.path.join(RESULTS, f"raw_{fid}.yml")
        run_edgesubpix(fp, yml)
        xs, ys, resp, direc, n_contours = read_edges(yml)
        n_pts = len(xs)
        rows.append({"frame": idx, "total_contours": n_contours,
                     "total_edge_points": n_pts,
                     "mean_response": float(resp.mean()) if n_pts else 0.0,
                     "p50_response": float(np.percentile(resp, 50)) if n_pts else 0.0,
                     "p95_response": float(np.percentile(resp, 95)) if n_pts else 0.0})
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(paths)}  用时 {time.time()-t0:.0f}s")
    with open(os.path.join(RESULTS, "baseline_raw_edge_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[Baseline] 完成 {time.time()-t0:.0f}s")


def run_all():
    paths = frame_paths()
    ids = frame_ids()
    print(f"共 {len(paths)} 帧, 开始 Stage 流程")
    grays = [load_gray(fp) for fp in paths]
    bg = compute_bg(grays)
    reg = load_registration()
    print(f"背景模型已计算 (mean), 载入 registration 平移量 {len(reg)} 帧")

    t0 = time.time()
    print("\n=== Stage C: Registration ===")
    stage_c_registration(ids, grays)

    print("\n=== Stage D/E/F: Motion Mask (static + MOG2) ===")
    for method in ["static", "mog2"]:
        stage_d_e_f(ids, grays, bg, reg, bg_method=method)

    print("\n=== Stage G: Tower Reference ===")
    stage_g_tower(ids)

    print("\n=== Stage A: Border Removal ===")
    se.stage_a_border()

    print("\n=== Stage H: Blade PCA Gate ===")
    se.stage_h_blade(grays, bg, reg)

    print("\n=== Stage I: Local EdgesSubPix ===")
    se.stage_i_local(grays, bg, reg)

    print("\n=== Stage J: Temporal ===")
    se.stage_j_temporal(grays, bg, reg)

    print("\n=== Stage K: Nearest Point Cpx ===")
    stage_k_nearest(ids, grays, bg, reg)

    # 合成视频
    print("\n=== 生成对比视频 ===")
    mo_dir = os.path.join(OUT, "frames", "motion_overlay")
    mo_pngs = sorted([os.path.join(mo_dir, f) for f in os.listdir(mo_dir) if f.endswith(".png")])
    if mo_pngs:
        save_video(mo_pngs, os.path.join(VIDEOS, "03_motion_mask.mp4"))
        print(f"  03_motion_mask.mp4 ({len(mo_pngs)} 帧)")
    print(f"\n[全部完成] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["baseline", "all", "c", "def", "g", "k"])
    args = ap.parse_args()
    if args.stage == "baseline":
        baseline()
    else:
        run_all()
