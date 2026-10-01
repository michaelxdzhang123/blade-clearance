#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mask_processor.py — Mask 处理模块 (TASK-1 第 10/11 节)

每个 blade 从二值 mask 中提取:
  - raw_contour      : cv2.findContours(CHAIN_APPROX_NONE) 原始轮廓点
  - raw_polygon      : 原始轮廓(不 approxPolyDP 简化, 永久保留)
  - simple_polygon   : approxPolyDP 简化(实时显示/快速几何)

铁律 (Rule 5): raw_polygon 永不因 simplify 操作被覆盖。
"""
from __future__ import annotations

import cv2
import numpy as np


def binarize_mask(mask) -> np.ndarray:
    """把 Ultralytics mask (H,W) 转成 0/255 二值 uint8 图。

    mask 可能是 CUDA tensor 或 numpy 数组。
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = (m > 0.5).astype(np.uint8)
    return (m > 0).astype(np.uint8) * 255


def extract_contour(binary_mask: np.ndarray):
    """从二值 mask 提取最大外轮廓 (RETR_EXTERNAL, CHAIN_APPROX_NONE)。

    返回 (N, 1, 2) 轮廓数组; 无轮廓时返回 None。
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_to_polygon(contour) -> list:
    """轮廓 (N,1,2) → polygon [[x,y], ...] 列表。"""
    if contour is None:
        return []
    pts = np.asarray(contour).reshape(-1, 2)
    return [list(map(float, p)) for p in pts]


def simplify_polygon(polygon: list, epsilon_ratio: float = 0.01) -> list:
    """approxPolyDP 简化 polygon (不修改原列表)。

    epsilon = epsilon_ratio * 周长。返回简化后的 [[x,y], ...] 列表。
    """
    if len(polygon) < 3:
        return [list(p) for p in polygon]
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(pts, True)
    epsilon = epsilon_ratio * peri
    approx = cv2.approxPolyDP(pts, epsilon, True)
    return [list(map(float, p)) for p in approx.reshape(-1, 2)]


def process_mask(mask, epsilon_ratio: float = 0.01):
    """一站式处理: mask → (binary, raw_polygon, simple_polygon)。

    返回 dict:
      {
        "binary_mask":    np.ndarray (H,W) uint8 0/255,
        "raw_polygon":    [[x,y],...],  # 原始轮廓, 永久保留
        "simple_polygon": [[x,y],...],  # 简化轮廓
        "raw_node_count": int,
        "simple_node_count": int,
      }
    无轮廓时 raw_polygon/simple_polygon 为空列表。
    """
    binary = binarize_mask(mask)
    contour = extract_contour(binary)
    if contour is None:
        return {
            "binary_mask": binary,
            "raw_polygon": [],
            "simple_polygon": [],
            "raw_node_count": 0,
            "simple_node_count": 0,
        }
    raw_polygon = contour_to_polygon(contour)
    simple_polygon = simplify_polygon(raw_polygon, epsilon_ratio)
    return {
        "binary_mask": binary,
        "raw_polygon": raw_polygon,
        "simple_polygon": simple_polygon,
        "raw_node_count": len(raw_polygon),
        "simple_node_count": len(simple_polygon),
    }
