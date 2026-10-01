#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
blade_segmenter.py — Blade 提取模块 (TASK-1 第 9 节)

从 Ultralytics Results 提取所有 blade instance:
  class_id / confidence / bbox / mask / polygon

规则:
  - 一帧无 blade → blade_detected=False, 程序继续不退出
  - 一帧多 blade → 保留全部 blade_instances[], 不静默只取最大者
  - 可额外输出 primary_blade (默认 = 最大 mask area 的 instance, 规则明确写在代码里)
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("blade_rtsp.blade_segmenter")


def _mask_area(mask) -> float:
    """计算二值 mask 面积 (像素数)。mask 可能是 CUDA tensor 或 numpy 数组。"""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    m = np.asarray(mask)
    if m.dtype == np.uint8:
        return float((m > 0).sum())
    return float((m > 0.5).sum())


def extract_blades(result, class_name: str = "blade") -> dict:
    """从 Results 提取 blade instances。

    返回 dict:
      {
        "blade_detected": bool,
        "blade_count": int,
        "blade_instances": [ {instance_id, class_id, class_name, confidence,
                              bbox:[x1,y1,x2,y2], mask, raw_polygon:[], ...}, ... ],
        "primary_instance_id": int | None,
      }

    注意: 此处的 bbox/mask/polygon 坐标属于「输入给 model.predict 的图像」
    坐标系 (ROI 或原始帧, 见 coordinate_mapper.py)。若输入是 ROI,
    需调用 coordinate_mapper 恢复到原始帧。
    """
    instances = []
    if result is None or result.masks is None or result.boxes is None:
        return {"blade_detected": False, "blade_count": 0,
                "blade_instances": instances, "primary_instance_id": None}

    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy().astype(float)
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
    masks_data = result.masks.data  # (N, H, W) 或 None
    names = result.names if hasattr(result, "names") else {}

    for idx in range(len(boxes)):
        cls_id = int(classes[idx])
        name = names.get(cls_id, str(cls_id))
        # 只保留 blade 类 (单类时 names={0:'blade'})
        if class_name is not None and name != class_name:
            continue
        mask = masks_data[idx] if masks_data is not None else None
        instances.append({
            "instance_id": idx,
            "class_id": cls_id,
            "class_name": name,
            "confidence": float(confs[idx]),
            "bbox": [float(v) for v in boxes[idx]],
            "mask": mask,
            "mask_area": _mask_area(mask) if mask is not None else 0.0,
        })

    # primary = 最大 mask area (规则明确)
    primary_id = None
    if instances:
        primary_id = max(instances, key=lambda x: x["mask_area"])["instance_id"]

    return {
        "blade_detected": len(instances) > 0,
        "blade_count": len(instances),
        "blade_instances": instances,
        "primary_instance_id": primary_id,
    }
