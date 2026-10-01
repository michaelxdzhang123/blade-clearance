#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
coordinate_mapper.py — 坐标系映射模块 (TASK-1 第 13/14 节)

三个坐标系 (TASK-1 第 13 节):
  A. RTSP Original Frame       —— 最终标准输出坐标系
  B. ROI Local Frame           —— 裁剪后的局部坐标系 (仅 use_roi=true 时)
  C. YOLO Internal 640 Frame   —— 模型内部推理尺寸 (不对外输出)

关键事实 (TASK-1 第 14 节):
  Ultralytics `result.masks.xy` 返回的 polygon 坐标已经是「输入给
  model.predict() 的那张图」的坐标, 不是 640 内部坐标。因此:
    - 若输入原始帧 → masks.xy 已是 A 系坐标, 无需映射。
    - 若输入 ROI    → masks.xy 是 B 系坐标, 需加 roi offset 恢复到 A 系。

本模块只负责 B → A 的平移恢复, 以及 bbox/center/polygon 的批量映射。
"""
from __future__ import annotations

import numpy as np


def roi_to_global(points, roi_offset_x: float, roi_offset_y: float):
    """把 ROI 局部坐标平移回原始帧坐标。

    points: (N, 2) 数组 [[x, y], ...] 或 list
    roi_offset_x/y: ROI 左上角在原始帧中的坐标 (x1, y1)

    返回同结构的平移后坐标。
    """
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr + np.array([roi_offset_x, roi_offset_y], dtype=np.float64)
    return arr


def map_bbox(bbox, roi_offset_x: float, roi_offset_y: float):
    """bbox [x1, y1, x2, y2] 从 ROI 坐标恢复到原始帧坐标。"""
    x1, y1, x2, y2 = bbox
    return [x1 + roi_offset_x, y1 + roi_offset_y,
            x2 + roi_offset_x, y2 + roi_offset_y]


def map_center(center, roi_offset_x: float, roi_offset_y: float):
    """center [cx, cy] 从 ROI 坐标恢复到原始帧坐标。"""
    cx, cy = center
    return [cx + roi_offset_x, cy + roi_offset_y]


def map_polygon(polygon, roi_offset_x: float, roi_offset_y: float):
    """polygon [[x,y],...] 从 ROI 坐标恢复到原始帧坐标。"""
    return [list(pt) for pt in roi_to_global(polygon, roi_offset_x, roi_offset_y)]


def map_mask(mask_2d, roi_offset_x: float, roi_offset_y: float,
             full_width: int, full_height: int):
    """把 ROI 局部二值 mask 平移到原始帧尺寸的画布上。

    mask_2d: (H_roi, W_roi) 二值 uint8 数组
    返回 (H_full, W_full) 二值 uint8 数组。
    """
    canvas = np.zeros((full_height, full_width), dtype=np.uint8)
    h, w = mask_2d.shape[:2]
    x0 = int(round(roi_offset_x))
    y0 = int(round(roi_offset_y))
    x1 = min(x0 + w, full_width)
    y1 = min(y0 + h, full_height)
    canvas[y0:y1, x0:x1] = mask_2d[: (y1 - y0), : (x1 - x0)]
    return canvas
