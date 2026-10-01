#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
blade_locator.py — Blade 定位模块 (TASK-1 第 15/16 节)

对每个 blade instance 输出 (全部为 Original Frame Coordinate):
  bbox / center / raw_polygon / simple_polygon / confidence / mask_area

并可选输出 blade_tip_candidate (标记为 candidate, 不是真实物理叶尖, 第 16 节)。
"""
from __future__ import annotations

import logging

import numpy as np

from . import coordinate_mapper, mask_processor

logger = logging.getLogger("blade_rtsp.blade_locator")


def _compute_center(bbox) -> list:
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def blade_tip_candidate(raw_polygon: list):
    """叶尖候选点 (candidate, 非真实物理叶尖)。

    规则: blade polygon 中 y 坐标最大的顶点(叶片自由端触底端)。
    这继承自 opencv/refine_blade_tip.py 的几何先验, 明确标记为候选。
    返回 (x, y) 或 None。
    """
    if not raw_polygon:
        return None
    pts = np.asarray(raw_polygon, dtype=np.float64)
    idx = int(np.argmax(pts[:, 1]))
    return [float(pts[idx, 0]), float(pts[idx, 1])]


def locate_blades(blade_instances: list, roi_offset=(0.0, 0.0),
                  frame_width: int = 0, frame_height: int = 0) -> list:
    """对每个 blade instance 做 mask 处理 + 坐标恢复, 产出最终定位结果。

    blade_instances: blade_segmenter.extract_blades() 的 blade_instances
    roi_offset: (x1, y1) ROI 左上角 (无 ROI 时为 (0,0))
    frame_width/height: 原始帧尺寸 (mask 恢复到原始帧用)

    返回 list, 每个元素含:
      instance_id, class_id, class_name, confidence,
      bbox, center, mask_area,
      raw_polygon, simple_polygon, raw_node_count, simple_node_count,
      blade_tip_candidate
    所有坐标均为原始帧坐标系。
    """
    ox, oy = roi_offset
    located = []
    for inst in blade_instances:
        mask = inst.get("mask")
        # 1. mask 处理 (在 ROI 局部坐标)
        proc = mask_processor.process_mask(mask) if mask is not None else {
            "binary_mask": None, "raw_polygon": [], "simple_polygon": [],
            "raw_node_count": 0, "simple_node_count": 0}

        # 2. 坐标恢复: ROI → 原始帧
        bbox_global = coordinate_mapper.map_bbox(inst["bbox"], ox, oy)
        center_global = coordinate_mapper.map_center(_compute_center(inst["bbox"]), ox, oy)
        raw_poly_global = coordinate_mapper.map_polygon(proc["raw_polygon"], ox, oy)
        simple_poly_global = coordinate_mapper.map_polygon(proc["simple_polygon"], ox, oy)

        located.append({
            "instance_id": inst["instance_id"],
            "class_id": inst["class_id"],
            "class_name": inst["class_name"],
            "confidence": inst["confidence"],
            "bbox": [float(v) for v in bbox_global],
            "center": [float(v) for v in center_global],
            "mask_area": float(inst["mask_area"]),
            "raw_polygon": raw_poly_global,
            "simple_polygon": simple_poly_global,
            "raw_node_count": proc["raw_node_count"],
            "simple_node_count": proc["simple_node_count"],
            "blade_tip_candidate": blade_tip_candidate(raw_poly_global),
        })
    return located
