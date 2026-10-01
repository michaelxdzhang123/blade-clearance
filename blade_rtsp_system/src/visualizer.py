#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualizer.py — 可视化模块 (TASK-1 第 20 节)

实时窗口显示: Frame ID / FPS / confidence / bbox / mask / polygon。
绘制用 simple_polygon (raw polygon 不全部绘制, 避免影响实时性能)。
"""
from __future__ import annotations

import cv2
import numpy as np

COLOR_MASK = (0, 255, 0)      # 绿色
COLOR_BBOX = (255, 0, 0)      # 蓝色
COLOR_TIP = (0, 0, 255)       # 红色
COLOR_TEXT = (0, 255, 255)    # 黄色


class Visualizer:
    def __init__(self, show_window: bool = True):
        self.show_window = show_window

    def draw(self, frame, frame_id: int, blades: list, fps: float):
        """在 frame 副本上绘制标注, 返回标注图。"""
        vis = frame.copy()
        for b in blades:
            bbox = [int(v) for v in b["bbox"]]
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          COLOR_BBOX, 2)
            # 绘制 simple_polygon (原始 polygon 不全部绘制)
            if b["simple_polygon"]:
                pts = np.asarray(b["simple_polygon"], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], isClosed=True, color=COLOR_MASK, thickness=2)
            # 叶尖候选点
            tip = b.get("blade_tip_candidate")
            if tip is not None:
                cv2.circle(vis, (int(tip[0]), int(tip[1])), 6, COLOR_TIP, -1)
            # 置信度标签
            cx, cy = [int(v) for v in b["center"]]
            cv2.putText(vis, f"blade {b['confidence']:.2f}",
                        (cx - 30, max(20, cy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

        # 顶部信息条
        cv2.putText(vis, f"frame={frame_id}  fps={fps:.1f}  blades={len(blades)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
        return vis

    def show(self, frame, window_name: str = "Blade RTSP"):
        if not self.show_window:
            return
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    def close(self):
        if self.show_window:
            cv2.destroyAllWindows()
