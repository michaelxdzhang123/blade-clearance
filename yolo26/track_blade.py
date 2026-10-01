#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track one wind-turbine blade with YOLO26n + YOLO26-seg.

YOLO26n provides the persistent ByteTrack identity. YOLO26-seg provides the
instance mask for the selected blade. The target identity is selected once and
then followed across frames; a new target is selected only after a configurable
number of missed frames.

Example:
    uv run python yolo26/track_blade.py \
        --video data/example.mp4 \
        --det-model yolo26/yolo26n.pt \
        --seg-model yolo26n-seg.pt \
        --output blade-track
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def _xyxy(box: Any) -> tuple[float, float, float, float]:
    values = [float(v) for v in box.xyxy[0].detach().cpu().tolist()]
    return values[0], values[1], values[2], values[3]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _choose_target(candidates: list[dict[str, Any]], previous_center: tuple[float, float] | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    if previous_center is None:
        return max(candidates, key=lambda c: c["confidence"] * np.sqrt(max(c["area"], 1.0)))
    px, py = previous_center
    frame_scale = max(c["frame_diagonal"] for c in candidates)
    def score(c: dict[str, Any]) -> float:
        cx, cy = c["center"]
        distance_penalty = np.hypot(cx - px, cy - py) / frame_scale
        return c["confidence"] * np.sqrt(max(c["area"], 1.0)) * np.exp(-2.0 * distance_penalty)
    return max(candidates, key=score)


def track_blade_video(
    video_path: str | Path,
    det_model_path: str | Path = "yolo26/yolo26n.pt",
    seg_model_path: str | Path = "yolo26n-seg.pt",
    output_dir: str | Path = "blade-track",
    blade_class: int = 0,
    conf: float = 0.25,
    iou: float = 0.5,
    tracker: str = "bytetrack.yaml",
    max_missed: int = 12,
    device: str | int | None = None,
) -> dict[str, Any]:
    """Track one blade and write an annotated MP4 plus per-frame CSV.

    Returns a dictionary containing output paths and frame statistics. The CSV
    records the selected ByteTrack ID, bounding box, confidence, and whether a
    YOLO26-seg mask was matched to that box.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / f"{video_path.stem}_blade_track.mp4"
    csv_path = output_dir / f"{video_path.stem}_blade_track.csv"

    detector = YOLO(str(det_model_path))
    segmenter = YOLO(str(seg_model_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    diagonal = float(np.hypot(width, height))
    target_id: int | None = None
    previous_center: tuple[float, float] | None = None
    missed = 0
    frames = tracked = masked = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame", "track_id", "confidence", "x1", "y1", "x2", "y2",
            "mask_found", "mask_area_px",
        ])
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            kwargs = {"conf": conf, "iou": iou, "classes": [blade_class], "persist": True,
                      "tracker": tracker, "verbose": False}
            if device is not None:
                kwargs["device"] = device
            det_result = detector.track(frame, **kwargs)[0]
            candidates: list[dict[str, Any]] = []
            if det_result.boxes is not None:
                ids = det_result.boxes.id
                for index, box in enumerate(det_result.boxes):
                    xyxy = _xyxy(box)
                    x1, y1, x2, y2 = xyxy
                    track_value = int(ids[index].item()) if ids is not None else -1
                    candidates.append({
                        "id": track_value, "box": xyxy,
                        "confidence": float(box.conf[0].item()),
                        "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        "frame_diagonal": diagonal,
                    })
            selected = next((c for c in candidates if target_id is not None and c["id"] == target_id), None)
            if selected is None and target_id is None:
                selected = _choose_target(candidates, previous_center)
                if selected is not None and selected["id"] >= 0:
                    target_id = selected["id"]
            if selected is None:
                missed += 1
                if missed > max_missed:
                    target_id = None
                csv_writer.writerow([frames, "", "", "", "", "", "", 0, 0])
                writer.write(frame)
                frames += 1
                continue

            missed = 0
            tracked += 1
            previous_center = selected["center"]
            seg_kwargs = {"conf": conf, "iou": iou, "classes": [blade_class], "verbose": False}
            if device is not None:
                seg_kwargs["device"] = device
            seg_result = segmenter.predict(frame, **seg_kwargs)[0]
            best_mask = None
            best_iou = 0.0
            if seg_result.boxes is not None and seg_result.masks is not None:
                for index, seg_box in enumerate(seg_result.boxes):
                    overlap = _iou(selected["box"], _xyxy(seg_box))
                    if overlap > best_iou:
                        best_iou = overlap
                        best_mask = seg_result.masks.data[index].detach().cpu().numpy() > 0.5
            mask_area = int(best_mask.sum()) if best_mask is not None else 0
            if best_mask is not None:
                masked += 1
                mask_uint8 = (cv2.resize(best_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) * 255)
                overlay = frame.copy()
                overlay[mask_uint8 > 0] = (0.65 * overlay[mask_uint8 > 0] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
                frame = overlay
            x1, y1, x2, y2 = [int(round(v)) for v in selected["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"blade id={selected['id']}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            csv_writer.writerow([frames, selected["id"], f"{selected['confidence']:.5f}", x1, y1, x2, y2, int(best_mask is not None), mask_area])
            writer.write(frame)
            frames += 1

    capture.release()
    writer.release()
    return {
        "video": str(video_path), "annotated_video": str(annotated_path),
        "csv": str(csv_path), "frames": frames, "tracked_frames": tracked,
        "masked_frames": masked, "track_id": target_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Track one blade with YOLO26n + YOLO26-seg")
    parser.add_argument("--video", required=True)
    parser.add_argument("--det-model", default="yolo26/yolo26n.pt")
    parser.add_argument("--seg-model", default="yolo26n-seg.pt")
    parser.add_argument("--output", default="blade-track")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-missed", type=int, default=12)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(track_blade_video(
        video_path=args.video,
        det_model_path=args.det_model,
        seg_model_path=args.seg_model,
        output_dir=args.output,
        conf=args.conf,
        max_missed=args.max_missed,
        device=args.device,
    ))


if __name__ == "__main__":
    main()
