#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stages_extra.py — Stage A(边框)/B(ROI)/H(Blade门)/I(局部EdgesSubPix)/J(时序) 补充"""
import os, csv
import numpy as np
import cv2
from opencv.pipeline import (run_edgesubpix, read_edges, border_remove,
                          mask_points, tower_band_points, dilate_mask,
                          morphology, motion_mask, static_frame_diff,
                          pca_elongation)

def align_frame(g, dx, dy):
    import cv2 as _cv
    M2 = np.float32([[1, 0, dx], [0, 1, dy]])
    return _cv.warpAffine(g, M2, (g.shape[1], g.shape[0]))


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "TASK1_EDGE_MOTION_MASK")
RESULTS = os.path.join(OUT, "results")
FRAMES_DIR = os.path.join(OUT, "frames")
W, H = 2560, 1440
TOWER_X = 434.0


def frame_ids():
    return [f[:-4] for f in sorted(os.listdir(FRAMES_DIR)) if f.endswith(".png")]


def stage_a_border(m_values=(10, 15, 20)):
    """Stage A: Border Removal, 统计 R_border"""
    ids = frame_ids()
    summary = {m: [] for m in m_values}
    for fid in ids:
        yml = os.path.join(RESULTS, f"raw_{fid}.yml")
        xs, ys, resp, direc, n_cont = read_edges(yml)
        pts = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2))
        n_before = len(pts)
        for m in m_values:
            keep, _ = border_remove(pts, W, H, m)
            summary[m].append(1 - keep.sum() / max(n_before, 1))
    with open(os.path.join(RESULTS, "A_border_filter_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["margin", "mean_R_border", "p50_R_border", "p95_R_border"])
        for m in m_values:
            a = np.array(summary[m])
            w.writerow([m, a.mean(), np.percentile(a, 50), np.percentile(a, 95)])
    for m in m_values:
        a = np.array(summary[m])
        print(f"[Stage A] margin={m:2d}px: R_border mean={a.mean()*100:.2f}% p50={np.percentile(a,50)*100:.2f}%")


def stage_h_blade(grays, bg, reg, min_elongations=(3.0, 5.0, 8.0), motion_thr=20, dilate_px=10):
    """Stage H: 对 motion-filtered 边缘点做 PCA elongation gate"""
    ids = frame_ids()
    rows = []
    for i, (fid, g) in enumerate(zip(ids, grays)):
        yml = os.path.join(RESULTS, f"raw_{fid}.yml")
        xs, ys, resp, direc, n_cont = read_edges(yml)
        pts = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2))
        dx, dy = reg.get(i, (0.0, 0.0)) if reg else (0.0, 0.0)
        g_aligned = align_frame(g, dx, dy)
        diff = static_frame_diff(bg, g_aligned)
        mm = dilate_mask(morphology(motion_mask(diff, motion_thr), op="close", ksize=5, iters=1), dilate_px)
        b_pts, _ = mask_points(pts, mm, H, W)
        if len(b_pts):
            b_pts = b_pts[np.abs(b_pts[:, 0] - TOWER_X) > 20]
        re_ = pca_elongation(b_pts)
        rows.append({"frame_id": fid, "blade_candidate_points": len(b_pts),
                     "pca_elongation_ratio": re_})
    with open(os.path.join(RESULTS, "H_blade_candidate_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    re_arr = np.array([r["pca_elongation_ratio"] for r in rows if r["pca_elongation_ratio"] < 1e9])
    n_pts = np.array([r["blade_candidate_points"] for r in rows])
    print(f"[Stage H] Blade candidates: mean={n_pts.mean():.0f} pts, "
          f"PCA elongation median={np.median(re_arr):.2f} (R_e>3 占比 {np.mean(re_arr>3)*100:.0f}%)")


def stage_i_local(grays, bg, reg, motion_thr=20, dilate_px=10, tower_band=20):
    """Stage I: Local EdgesSubPix — 在 blade band + tower band 内 crop 后重跑 EdgesSubPix"""
    ids = frame_ids()
    rows = []
    local_dir = os.path.join(OUT, "frames", "local_crop")
    os.makedirs(local_dir, exist_ok=True)
    for i, (fid, g) in enumerate(zip(ids, grays)):
        # tower band crop
        tx0 = int(TOWER_X - tower_band); tx1 = int(TOWER_X + tower_band)
        tower_crop = g[:, max(tx0,0):min(tx1,W)]
        tower_yml = os.path.join(RESULTS, f"local_tower_{fid}.yml")
        # blade band crop (motion mask 外接框)
        dx, dy = reg.get(i, (0.0, 0.0)) if reg else (0.0, 0.0)
        g_aligned = align_frame(g, dx, dy)
        diff = static_frame_diff(bg, g_aligned)
        mm = dilate_mask(morphology(motion_mask(diff, motion_thr), op="close", ksize=5, iters=1), dilate_px)
        ys_, xs_ = np.where(mm > 0)
        n_blade_pts = 0
        if len(xs_):
            bx0, bx1 = xs_.min(), xs_.max(); by0, by1 = ys_.min(), ys_.max()
            bx0 = max(0, bx0-20); bx1 = min(W, bx1+20); by0 = max(0, by0-20); by1 = min(H, by1+20)
            blade_crop = g[by0:by1, bx0:bx1]
            blade_img = os.path.join(local_dir, f"blade_{fid}.png")
            cv2.imwrite(blade_img, blade_crop)
            blade_yml = os.path.join(RESULTS, f"local_blade_{fid}.yml")
            run_edgesubpix(blade_img, blade_yml)
            bxs, bys, br, bd, bn = read_edges(blade_yml)
            n_blade_pts = len(bxs)
        # tower crop 保存并跑 EdgesSubPix
        tower_img = os.path.join(local_dir, f"tower_{fid}.png")
        cv2.imwrite(tower_img, tower_crop)
        run_edgesubpix(tower_img, tower_yml)
        txs, tys, tr, td, tn = read_edges(tower_yml)
        rows.append({"frame_id": fid, "tower_subpixel_points": len(txs),
                     "blade_subpixel_points": n_blade_pts})
    with open(os.path.join(RESULTS, "I_local_edgesubpix_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    tp = np.array([r["tower_subpixel_points"] for r in rows])
    bp = np.array([r["blade_subpixel_points"] for r in rows])
    print(f"[Stage I] Local EdgesSubPix: tower pts mean={tp.mean():.0f} (连续性 {np.mean(tp>0)*100:.0f}%), "
          f"blade pts mean={bp.mean():.0f} (连续性 {np.mean(bp>0)*100:.0f}%)")


def stage_j_temporal(grays, bg, reg, motion_thr=20, dilate_px=10):
    """Stage J: 时序连续性 — blade band 质心帧间位移"""
    ids = frame_ids()
    centroids = []
    for i, g in enumerate(grays):
        dx, dy = reg.get(i, (0.0, 0.0)) if reg else (0.0, 0.0)
        g_aligned = align_frame(g, dx, dy)
        diff = static_frame_diff(bg, g_aligned)
        mm = dilate_mask(morphology(motion_mask(diff, motion_thr), op="close", ksize=5, iters=1), dilate_px)
        ys_, xs_ = np.where(mm > 0)
        if len(xs_) == 0:
            centroids.append((np.nan, np.nan))
        else:
            centroids.append((float(xs_.mean()), float(ys_.mean())))
    rows = []
    for i in range(1, len(centroids)):
        dx = centroids[i][0] - centroids[i-1][0]
        dy = centroids[i][1] - centroids[i-1][1]
        rows.append({"frame": i, "centroid_dx": dx, "centroid_dy": dy,
                     "displacement_px": float(np.hypot(dx, dy)) if np.isfinite(dx) else np.nan})
    with open(os.path.join(RESULTS, "J_temporal_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    disp = np.array([r["displacement_px"] for r in rows if np.isfinite(r["displacement_px"])])
    print(f"[Stage J] Temporal: blade band 质心帧间位移 mean={disp.mean():.1f}px p95={np.percentile(disp,95):.1f}px")
