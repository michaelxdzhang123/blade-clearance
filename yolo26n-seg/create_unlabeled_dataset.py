#!/usr/bin/env python3
"""Create a leakage-aware unlabeled image dataset from source MP4 videos.

The script intentionally writes images only; it never writes YOLO label files.
By default it:
  - reads source videos from the training YAML;
  - skips videos assigned to the labeled validation split;
  - skips frames already present in the labeled dataset;
  - letterboxes each image to a square output size.

Example:
  uv run python yolo26n-seg/create_unlabeled_dataset.py \
      --config yolo26/train-0915/train_config.yaml \
      --labeled-root train-0915-640 \
      --out unlabeled-0915-640 \
      --imgsz 640 \
      --every 4

The output directory is suitable for:
  --unlabeled-dir unlabeled-0915-640
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import av
import cv2
import numpy as np
import yaml

IMAGE_SIZE_CHOICES = (640, 1024, 1280)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FRAME_NAME_RE = re.compile(r"^v(?P<video>\d+)_f(?P<frame>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create unlabeled images from MP4 videos without YOLO labels."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML containing videos and dataset.val_videos.",
    )
    parser.add_argument(
        "--labeled-root",
        type=Path,
        required=True,
        help="Existing auto-labeled dataset, used to avoid duplicate labeled frames.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory containing unlabeled JPG images only.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        choices=IMAGE_SIZE_CHOICES,
        required=True,
        help="Square output image size: 640, 1000, or 1250.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=4,
        help="Keep every Nth source frame. Default: 4.",
    )
    parser.add_argument(
        "--include-val-videos",
        action="store_true",
        help="Also extract videos assigned to the labeled validation split. "
        "Disabled by default to avoid validation leakage.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for a smoke test.",
    )
    return parser.parse_args()


def load_config(path: Path) -> tuple[list[Path], int]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    videos = [Path(p) for p in cfg["videos"]]
    val_videos = int(cfg.get("dataset", {}).get("val_videos", 0))
    if val_videos < 0 or val_videos >= len(videos):
        raise ValueError(
            f"dataset.val_videos={val_videos} must be in [0, {len(videos) - 1}]"
        )
    return videos, val_videos


def collect_labeled_frames(labeled_root: Path) -> set[tuple[int, int]]:
    """Return (video_index, frame_index) pairs already present in labeled images."""
    blocked: set[tuple[int, int]] = set()
    image_root = labeled_root / "images"
    for image in image_root.rglob("*") if image_root.exists() else []:
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
            continue
        match = FRAME_NAME_RE.match(image.stem)
        if match:
            blocked.add((int(match.group("video")), int(match.group("frame"))))
    return blocked


def letterbox(image, size: int):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=image.dtype)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def main() -> None:
    args = parse_args()
    if args.every <= 0:
        raise ValueError("--every must be a positive integer")
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    if not args.labeled_root.exists():
        raise FileNotFoundError(args.labeled_root)

    videos, val_videos = load_config(args.config)
    blocked = collect_labeled_frames(args.labeled_root)
    val_ids = set(range(len(videos) - val_videos, len(videos)))

    args.out.mkdir(parents=True, exist_ok=True)
    # Do not silently mix old images from a previous extraction.
    old_images = [p for p in args.out.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if old_images:
        raise FileExistsError(
            f"Output already contains {len(old_images)} images: {args.out}. "
            "Choose a new directory or remove it explicitly before running."
        )

    selected_videos = [
        (video_id, path)
        for video_id, path in enumerate(videos)
        if args.include_val_videos or video_id not in val_ids
    ]

    saved = 0
    skipped_labeled = 0
    skipped_sampling = 0
    video_stats = []

    for video_id, video_path in selected_videos:
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        video_saved = 0
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index % args.every != 0:
                skipped_sampling += 1
                continue
            if (video_id, frame_index) in blocked:
                skipped_labeled += 1
                continue
            bgr = frame.to_ndarray(format="bgr24")
            output = letterbox(bgr, args.imgsz)
            filename = f"v{video_id:02d}_f{frame_index:05d}.jpg"
            if not cv2.imwrite(
                str(args.out / filename), output, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"Failed to write {args.out / filename}")
            saved += 1
            video_saved += 1
            if args.max_images is not None and saved >= args.max_images:
                break
        container.close()
        video_stats.append(
            {
                "video_id": video_id,
                "source": str(video_path),
                "saved": video_saved,
                "validation_video": video_id in val_ids,
            }
        )
        if args.max_images is not None and saved >= args.max_images:
            break

    manifest = {
        "output": str(args.out.resolve()),
        "imgsz": args.imgsz,
        "every": args.every,
        "contains_labels": False,
        "labeled_root": str(args.labeled_root.resolve()),
        "blocked_labeled_frames": len(blocked),
        "skipped_labeled_frames": skipped_labeled,
        "skipped_by_sampling": skipped_sampling,
        "include_val_videos": args.include_val_videos,
        "saved_images": saved,
        "videos": video_stats,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
