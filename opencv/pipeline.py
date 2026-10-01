#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pipeline.py — Stage A~K 核心实现 (Motion Mask + Edge Filtering 链)"""
import os, subprocess
import numpy as np
import cv2

EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"

# ---------- 共享 EdgesSubPix 调用 ----------
def run_edgesubpix(img_path, yml_path, alpha=1.0, low=20, high=40, mode=1):
    out_img = yml_path.replace(".yml", "_out.jpg")
    subprocess.run([EDGE_BIN, img_path, out_img, f"--data={yml_path}",
                    f"--alpha={alpha}", f"--low={low}", f"--high={high}",
                    f"--mode={mode}"], capture_output=True, timeout=120)

def read_edges(yml_path):
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    cn = fs.getNode("contours"); n = cn.size()
    xs, ys, resp, direc = [], [], [], []
    for i in range(n):
        c = cn.at(i)
        pn = c.getNode("points"); m = pn.size()
        pts = np.array([pn.at(k).real() for k in range(m)]).reshape(-1, 2)
        xs.append(pts[:, 0]); ys.append(pts[:, 1])
        rn = c.getNode("response")
        resp.append(np.array([rn.at(k).real() for k in range(rn.size())]))
        dn = c.getNode("direction")
        direc.append(np.array([dn.at(k).real() for k in range(dn.size())]))
    fs.release()
    if not xs:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0), 0
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(resp), np.concatenate(direc), n)

# ---------- Stage A: Border Removal ----------
def border_remove(points, W, H, m):
    x, y = points[:, 0], points[:, 1]
    keep = (x >= m) & (x <= W - m) & (y >= m) & (y <= H - m)
    return points[keep], keep

# ---------- Stage C: Registration (ECC, 塔筒 ROI) ----------
def register_ecc(ref_gray, cur_gray, warp_mode=cv2.MOTION_TRANSLATION):
    if warp_mode == cv2.MOTION_TRANSLATION:
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
        try:
            _, w = cv2.findTransformECC(ref_gray, cur_gray, warp, warp_mode, crit)
            return w
        except cv2.error:
            return np.eye(2, 3, dtype=np.float32)
    return np.eye(2, 3, dtype=np.float32)

# ---------- Stage D/E: Background + Motion Mask ----------
def static_frame_diff(bg_gray, cur_gray):
    return cv2.absdiff(bg_gray, cur_gray)

def motion_mask(diff, threshold):
    return ((diff > threshold) * 255).astype(np.uint8)

def morphology(mask, op="close", ksize=5, iters=1):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if op == "close":
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=iters)
    elif op == "open":
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=iters)
    return mask

def dilate_mask(mask, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.dilate(mask, k)

# ---------- Stage F: Mask 与 Edge 相交 ----------
def mask_points(points, mask_band, H, W):
    if len(points) == 0:
        return points, np.zeros(0, dtype=bool)
    xi = np.clip(points[:, 0].astype(int), 0, W - 1)
    yi = np.clip(points[:, 1].astype(int), 0, H - 1)
    keep = mask_band[yi, xi] > 0
    return points[keep], keep

# ---------- Stage G: Tower Reference Band ----------
def tower_band_points(points, tower_line_x, band_px, H, W):
    """tower_line_x: 塔筒参考 x 坐标; 保留 |x - tower_x| < band 的点"""
    if len(points) == 0:
        return points, np.zeros(0, dtype=bool)
    keep = np.abs(points[:, 0] - tower_line_x) < band_px
    return points[keep], keep

def tower_orientation_gate(points, direction, ref_theta, tol_deg):
    """塔筒边缘方向 gate (direction 是梯度方向, 与边缘垂直)"""
    if len(points) == 0:
        return points, np.zeros(0, dtype=bool)
    tol = np.radians(tol_deg)
    # 塔筒边缘接近垂直 -> 梯度方向接近水平 (0 或 pi)
    d = np.abs(direction - ref_theta)
    d = np.minimum(d, 2*np.pi - d)
    keep = d < tol
    return points[keep], keep

# ---------- Stage H: Blade PCA Elongation Gate ----------
def pca_elongation(points):
    if len(points) < 3:
        return 0.0
    c = points - points.mean(axis=0)
    cov = c.T @ c / len(points)
    eig = np.linalg.eigvalsh(cov)
    eig = np.sort(eig)[::-1]
    if eig[1] < 1e-9:
        return float("inf") if eig[0] > 0 else 0.0
    return float(eig[0] / eig[1])

# ---------- Stage K: Nearest Point ----------
def nearest_pair(blade_pts, tower_pts):
    """返回 (c_px, blade_pt, tower_pt)"""
    if len(blade_pts) == 0 or len(tower_pts) == 0:
        return float("nan"), None, None
    # 用 KD-tree 加速 (scipy)
    from scipy.spatial import cKDTree
    tree = cKDTree(tower_pts)
    dist, idx = tree.query(blade_pts, k=1)
    j = int(np.argmin(dist))
    return float(dist[j]), blade_pts[j], tower_pts[int(idx[j])]
