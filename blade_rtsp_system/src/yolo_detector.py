#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yolo_detector.py — YOLO26 推理模块 (TASK-1 第 8 节)

要求:
  - 模型只加载一次 (禁止每帧重建 YOLO 对象)
  - 记录 inference time
  - GPU 不可用时明确报错或按配置切换 CPU (不允许静默切换不记录日志)
"""
from __future__ import annotations

import logging
import time

from ultralytics import YOLO

logger = logging.getLogger("blade_rtsp.yolo_detector")


class YOLODetector:
    """封装 YOLO 模型加载与单帧推理。"""

    def __init__(self, model_path: str, imgsz: int = 640,
                 conf: float = 0.5, device: str | int = 0,
                 retina_masks: bool = True):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.retina_masks = retina_masks
        self.model: YOLO | None = None
        self.loaded_device: str | None = None

    def load(self):
        """加载模型 (只调用一次)。加载失败抛异常 (Rule: 不允许静默)。"""
        logger.info("加载 YOLO 模型: %s", self.model_path)
        self.model = YOLO(self.model_path)
        # 确认实际使用的设备
        self.loaded_device = self._resolve_device()
        logger.info("模型加载成功, device=%s, imgsz=%d", self.loaded_device, self.imgsz)

    def _resolve_device(self) -> str:
        """解析并记录实际推理设备 (不允许静默切换)。"""
        if str(self.device).lower() == "cpu":
            return "cpu"
        # 尝试用 torch 确认 CUDA 是否可用
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA 不可用, 回退到 CPU")
                return "cpu"
            return f"cuda:{self.device}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch 查询失败(%s), 回退到 CPU", exc)
            return "cpu"

    def predict(self, frame):
        """对单帧做分割推理, 返回 ultralytics Results (首个)。

        frame: BGR ndarray (任意分辨率, 不做 resize, 保留原始分辨率)。
        """
        if self.model is None:
            raise RuntimeError("模型未加载, 请先调用 load()")
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            retina_masks=self.retina_masks,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0
        result = results[0] if results else None
        return result, inference_ms
