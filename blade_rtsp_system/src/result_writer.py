#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
result_writer.py — 结果输出模块 (TASK-1 第 17/18/19 节)

  CSV  : output/csv/blade_positions.csv   (扁平行, polygon 不存)
  JSONL: output/jsonl/blade_positions.jsonl (每帧一行, polygon 存这里)

写入策略 (同 rtsp_daemon.py 的缓冲日志):
  运行期只把结果缓冲到内存 list (零磁盘 I/O), 进程结束时 close() 一次性写文件,
  避免逐帧写盘/逐帧 flush 的 I/O 拖慢主循环。

  代价: 全部结果驻留内存, 长跑(尤其 JSONL 含完整 polygon)内存占用会增长;
  若担心内存, 可改回「分批落盘」或用 csv/jsonl 的文件对象自带缓冲(去掉显式 flush)。
"""
from __future__ import annotations

import csv
import json
import logging
import os

logger = logging.getLogger("blade_rtsp.result_writer")

CSV_FIELDS = [
    "timestamp", "frame_id", "frame_width", "frame_height",
    "loop_ms",
    "blade_detected", "blade_count",
    "instance_id", "confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "center_x", "center_y", "mask_area",
]


class ResultWriter:
    """结果写出器: 运行期缓冲到内存, close() 时一次性落盘 CSV + JSONL。"""

    def __init__(self, csv_path: str, jsonl_path: str):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self._jsonl_lines: list[str] = []   # 缓冲 JSONL 行 (每帧一行, 已 json.dumps)
        self._csv_rows: list[dict] = []     # 缓冲 CSV 行 (每个 blade instance 一行 dict)

    def open(self):
        """准备输出目录 + 清空缓冲 (不打开文件, 不写盘)。"""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
        self._jsonl_lines = []              # 重置 JSONL 缓冲
        self._csv_rows = []                 # 重置 CSV 缓冲
        logger.info("结果输出(缓冲模式): csv=%s jsonl=%s", self.csv_path, self.jsonl_path)

    def write(self, frame_result: dict):
        """把一帧结果追加到内存缓冲 (不写盘)。

        frame_result = {frame_id, timestamp, frame_width, frame_height,
                        loop_ms, blade_detected, blade_count, blades:[...]}
        """
        # JSONL: 缓冲完整结构 (含 polygon), 只 append 到 list
        self._jsonl_lines.append(json.dumps(frame_result, ensure_ascii=False))

        # CSV: 缓冲扁平行 (无 blade 时也写一行)
        fid = frame_result["frame_id"]
        ts = frame_result["timestamp"]
        w = frame_result["frame_width"]
        h = frame_result["frame_height"]
        det = frame_result["blade_detected"]
        cnt = frame_result["blade_count"]
        loop_ms = frame_result["loop_ms"]
        blades = frame_result["blades"]

        if not blades:
            self._csv_rows.append({
                "timestamp": f"{ts:.6f}", "frame_id": fid,
                "frame_width": w, "frame_height": h,
                "loop_ms": f"{loop_ms:.2f}",
                "blade_detected": det, "blade_count": cnt,
                "instance_id": "", "confidence": "",
                "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "",
                "center_x": "", "center_y": "", "mask_area": "",
            })
        else:
            for b in blades:
                x1, y1, x2, y2 = b["bbox"]
                cx, cy = b["center"]
                self._csv_rows.append({
                    "timestamp": f"{ts:.6f}", "frame_id": fid,
                    "frame_width": w, "frame_height": h,
                    "loop_ms": f"{loop_ms:.2f}",
                    "blade_detected": det, "blade_count": cnt,
                    "instance_id": b["instance_id"],
                    "confidence": f"{b['confidence']:.6f}",
                    "bbox_x1": f"{x1:.2f}", "bbox_y1": f"{y1:.2f}",
                    "bbox_x2": f"{x2:.2f}", "bbox_y2": f"{y2:.2f}",
                    "center_x": f"{cx:.2f}", "center_y": f"{cy:.2f}",
                    "mask_area": f"{b['mask_area']:.1f}",
                })

    def close(self):
        """一次性把全部缓冲写入文件 (JSONL + CSV), 写后清空缓冲。"""
        # JSONL: 一次性写全部行
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            if self._jsonl_lines:
                f.write("\n".join(self._jsonl_lines) + "\n")

        # CSV: 一次性写 header + 全部行
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._csv_rows)

        logger.info("结果已落盘: jsonl=%d 行 csv=%d 行",
                    len(self._jsonl_lines), len(self._csv_rows))
        self._jsonl_lines = []              # 落盘后清空
        self._csv_rows = []                 # 落盘后清空
