#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_mask_processor.py — Mask 处理单元测试 (TASK-1 第 26.2 节)

验证:
  contour exists / raw_polygon node count > 0 /
  simple_polygon node count > 0 /
  simple node count <= raw node count /
  raw_polygon 不因 simplify 被修改
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import mask_processor  # noqa: E402


def make_rect_mask(h=100, w=100, pad=20):
    """中间一个矩形叶片 mask。"""
    m = np.zeros((h, w), dtype=np.uint8)
    m[pad:h - pad, pad:w - pad] = 255
    return m


def test_process_mask_basic():
    m = make_rect_mask()
    res = mask_processor.process_mask(m)
    assert len(res["raw_polygon"]) > 0, "raw_polygon 应有节点"
    assert len(res["simple_polygon"]) > 0, "simple_polygon 应有节点"
    assert res["simple_node_count"] <= res["raw_node_count"], \
        "简化后节点数不应多于原始"


def test_raw_polygon_unchanged_after_simplify():
    m = make_rect_mask()
    res = mask_processor.process_mask(m)
    raw_before = [list(p) for p in res["raw_polygon"]]
    # 再次 simplify raw_polygon, 不应影响 res 里的 raw_polygon
    _ = mask_processor.simplify_polygon(res["raw_polygon"], 0.01)
    assert res["raw_polygon"] == raw_before, \
        "raw_polygon 不应被 simplify 操作修改"


def test_process_mask_empty():
    m = np.zeros((100, 100), dtype=np.uint8)
    res = mask_processor.process_mask(m)
    assert res["raw_polygon"] == []
    assert res["simple_polygon"] == []
    assert res["raw_node_count"] == 0


if __name__ == "__main__":
    test_process_mask_basic()
    test_raw_polygon_unchanged_after_simplify()
    test_process_mask_empty()
    print("test_mask_processor: ALL PASS")
