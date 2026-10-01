#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_coordinate_mapper.py — 坐标映射单元测试 (TASK-1 第 26.1 节)

验证: ROI offset=(100,200), ROI point=(50,60) → Global=(150,260)
      以及 bbox / center / polygon 的坐标恢复。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import coordinate_mapper  # noqa: E402


def test_roi_to_global_point():
    pts = coordinate_mapper.roi_to_global([[50, 60]], 100, 200)
    assert pts.shape == (1, 2)
    assert pts[0, 0] == 150.0
    assert pts[0, 1] == 260.0


def test_roi_to_global_no_offset():
    pts = coordinate_mapper.roi_to_global([[10, 20], [30, 40]], 0, 0)
    assert pts[0, 0] == 10.0
    assert pts[1, 1] == 40.0


def test_map_bbox():
    bbox = coordinate_mapper.map_bbox([50, 60, 150, 160], 100, 200)
    assert bbox == [150, 260, 250, 360]


def test_map_center():
    center = coordinate_mapper.map_center([50, 60], 100, 200)
    assert center == [150, 260]


def test_map_polygon():
    poly = coordinate_mapper.map_polygon([[50, 60], [70, 80]], 100, 200)
    assert poly == [[150, 260], [170, 280]]


def test_map_mask():
    import numpy as np
    roi_mask = np.ones((10, 20), dtype=np.uint8) * 255
    full = coordinate_mapper.map_mask(roi_mask, 100, 200, 400, 300)
    assert full.shape == (300, 400)
    # ROI 区域左上角 (200,100) 应为 255
    assert full[200, 100] == 255
    # ROI 外 (0,0) 应为 0
    assert full[0, 0] == 0


if __name__ == "__main__":
    test_roi_to_global_point()
    test_roi_to_global_no_offset()
    test_map_bbox()
    test_map_center()
    test_map_polygon()
    test_map_mask()
    print("test_coordinate_mapper: ALL PASS")
