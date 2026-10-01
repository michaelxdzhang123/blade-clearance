#!/usr/bin/env python3
"""
YOLO26 blade/tower instance-segmentation training with labeled + unlabeled data.

Pipeline
--------
Stage 1: train YOLO26-Seg FROM SCRATCH on auto-labeled data.
Stage 2: use Stage-1 best.pt as a teacher to create high-confidence pseudo labels
         for unlabeled blade images.
Stage 3: merge auto-labeled + pseudo-labeled training images and continue
         training a student model. Validation/test use the held-out auto-labeled split only.

This script does NOT use external pretrained weights. It builds Stage 1 from
`yolo26n-seg.yaml` with randomly initialized weights.

Expected input layout
---------------------
LABELED_ROOT/
  images/
    train/
    val/
    test/          # optional
  labels/
    train/
    val/
    test/          # optional

UNLABELED_DIR/
  **/*.jpg|jpeg|png|bmp|tif|tiff|webp

YOLO instance-segmentation label format:
  class_id x1 y1 x2 y2 ... xn yn
where polygon coordinates are normalized to [0, 1].

Example
-------
python train_blade_yolo26_semisupervised.py \
  --labeled-root /data/blade_labeled \
  --unlabeled-dir /data/blade_unlabeled \
  --work-dir /data/runs/blade_ssl \
  --names blade,tower \
  --model-yaml yolo26n-seg.yaml \
  --imgsz 1024 \
  --epochs-stage1 200 \
  --epochs-stage2 150 \
  --pseudo-conf 0.75 \
  --device 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLO26-Seg from scratch with labeled + pseudo-labeled blade data."
    )
    p.add_argument("--labeled-root", type=Path, required=True)
    p.add_argument("--unlabeled-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("runs/blade_ssl"))
    p.add_argument("--model-yaml", default="yolo26n-seg.yaml")
    p.add_argument(
        "--names",
        default="blade,tower",
        help="Comma-separated class names. Class IDs in your labels must match this order.",
    )
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs-stage1", type=int, default=200)
    p.add_argument("--epochs-stage2", type=int, default=150)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch", type=int, default=-1, help="-1 = Ultralytics auto batch")
    p.add_argument(
        "--device",
        default="auto",
        help="Device: '0'=cuda:0, 'cpu', or 'auto'(默认, 检测到 GPU 用 0, 否则 cpu)",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--mosaic", type=float, default=0.5)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--copy-paste", type=float, default=0.0)

    # Pseudo-label filtering. High confidence is intentionally conservative.
    p.add_argument("--pseudo-conf", type=float, default=0.75)
    p.add_argument("--pseudo-iou", type=float, default=0.60)
    p.add_argument(
        "--min-polygon-points",
        type=int,
        default=6,
        help="Reject tiny/degenerate pseudo-mask polygons.",
    )
    p.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.00075,
        help="Minimum normalized polygon area (image area = 1.0). 0.00075 = 307px @ 640x640 (blade min observed).",
    )
    p.add_argument(
        "--include-empty-pseudo",
        action="store_true",
        help=(
            "Include unlabeled images with no accepted pseudo objects as empty-label background images. "
            "Default is OFF because teacher false-negatives can otherwise damage training."
        ),
    )
    p.add_argument(
        "--stage2-from-scratch",
        action="store_true",
        help=(
            "Build a fresh YOLO26 model for Stage 2 instead of warm-starting from the Stage-1 teacher. "
            "Default warm-start still uses ONLY your own Stage-1 weights, not external pretrained weights."
        ),
    )
    p.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1. Requires --teacher-weights.",
    )
    p.add_argument("--teacher-weights", type=Path, default=None)
    p.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        help=(
            "Optional .pt to initialize Stage-1 teacher (fine-tune) instead of "
            "training from scratch. Leave unset to keep the default from-scratch behavior."
        ),
    )
    p.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Generate pseudo labels/combined dataset but do not run Stage-2 training.",
    )
    return p.parse_args()


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def validate_input_layout(root: Path) -> None:
    required = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required dataset paths:\n  " + "\n  ".join(missing))


def write_dataset_yaml(
    outfile: Path,
    train_images: Path,
    val_images: Path,
    names: list[str],
    test_images: Path | None = None,
) -> Path:
    data = {
        "train": str(train_images.resolve()),
        "val": str(val_images.resolve()),
        "names": {i: name for i, name in enumerate(names)},
    }
    if test_images is not None and test_images.exists():
        data["test"] = str(test_images.resolve())
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return outfile


def expected_label_path(image: Path, images_root: Path, labels_root: Path) -> Path:
    rel = image.relative_to(images_root)
    return labels_root / rel.with_suffix(".txt")


def short_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]


def safe_link_or_copy(src: Path, dst: Path) -> None:
    """优先 symlink 硬链接源文件, 失败则 copy(跨文件系统/权限不足时)。

    为什么优先 symlink: 大数据集下 copy 会占双倍磁盘 + 慢, symlink 零拷贝。
    先 unlink 旧 dst: 避免残留符号链接指向错误位置。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def polygon_area_normalized(poly: np.ndarray) -> float:
    """Shoelace(鞋带)公式算 Nx2 归一化 polygon 面积。

    原理: 多边形面积 = 0.5 * |Σ(x_i·y_{i+1} − x_{i+1}·y_i)|,
    用 np.roll(y, 1) 向量化实现相邻顶点错位乘。
    用于过滤"碎 mask"(面积≈0 的退化 polygon)。
    """
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def stage_real_training_data(labeled_root: Path, combined_root: Path) -> int:
    """把自动标注的训练对(图片+标注)链接/复制进 combined 训练集。

    关键设计:
      - 图片没标注 → 跳过(不静默当负样本): 因为"有图无标注"是歧义的,
        可能是漏标, 当负样本会引入错误监督信号。
      - uid = short_id(路径): 加 hash 前缀, 避免同名文件冲突。
      - safe_link_or_copy: 优先 symlink(省磁盘), 失败才 copy。
    """
    src_images = labeled_root / "images" / "train"
    src_labels = labeled_root / "labels" / "train"
    out_images = combined_root / "images" / "train"
    out_labels = combined_root / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    n = 0
    for img in list_images(src_images):
        lbl = expected_label_path(img, src_images, src_labels)
        if not lbl.exists():
            # A labeled-training image without label is ambiguous. Skip it rather than silently
            # treating it as a negative sample.
            print(f"[WARN] Missing auto-label, skipping: {img}")
            continue
        uid = short_id(img)
        out_img = out_images / f"real_{uid}_{img.name}"
        out_lbl = out_labels / f"real_{uid}_{img.stem}.txt"
        safe_link_or_copy(img, out_img)
        safe_link_or_copy(lbl, out_lbl)
        n += 1
    return n


def generate_pseudo_labels(
    teacher: YOLO,
    unlabeled_dir: Path,
    combined_root: Path,
    pseudo_csv: Path,
    imgsz: int,
    device: str,
    conf_threshold: float,
    iou_threshold: float,
    min_polygon_points: int,
    min_area_ratio: float,
    include_empty: bool,
) -> dict[str, int]:
    """
    Predict segmentation masks on unlabeled images and add accepted polygons to the combined set.

    IMPORTANT: polygons are NOT simplified with approxPolyDP. The full predicted contour is kept.
    """
    images = list_images(unlabeled_dir)
    if not images:
        raise FileNotFoundError(f"No images found under: {unlabeled_dir}")

    out_images = combined_root / "images" / "train"
    out_labels = combined_root / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    pseudo_csv.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "unlabeled_images": len(images),
        "accepted_images": 0,
        "empty_images_included": 0,
        "accepted_instances": 0,
        "rejected_instances": 0,
    }

    with pseudo_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "source_image",
                "combined_image",
                "accepted_instances",
                "mean_confidence",
                "classes",
                "status",
            ],
        )
        writer.writeheader()

        # stream=True prevents keeping Results for the whole unlabeled corpus in RAM.
        results = teacher.predict(
            source=[str(p) for p in images],
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            stream=True,
            verbose=False,
            batch=8,   # 1024 下 predict 默认 batch=16 会 OOM, 显式降到 8
        )

        for result in results:
            src = Path(result.path)
            uid = short_id(src)
            out_img = out_images / f"pseudo_{uid}_{src.name}"
            out_lbl = out_labels / f"pseudo_{uid}_{src.stem}.txt"

            accepted_lines: list[str] = []
            accepted_confs: list[float] = []
            accepted_classes: list[int] = []

            if result.masks is not None and result.boxes is not None and len(result.boxes) > 0:
                polygons = result.masks.xyn  # 归一化 mask polygon(全轮廓顶点)
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                confs = result.boxes.conf.detach().cpu().numpy().astype(float)

                # ── 三重保守过滤: 只接受"teacher 很有把握"的伪标注 ──
                # 错误伪标注会污染 Stage-2 训练, 故阈值故意保守
                for poly, cls_id, score in zip(polygons, classes, confs):
                    poly = np.asarray(poly, dtype=np.float32)

                    if score < conf_threshold:      # ① 置信度 < 阈值 → 拒绝
                        stats["rejected_instances"] += 1
                        continue
                    if len(poly) < min_polygon_points:  # ② 顶点太少(退化 mask) → 拒绝
                        stats["rejected_instances"] += 1
                        continue
                    if polygon_area_normalized(poly) < min_area_ratio:  # ③ 面积太小(碎 mask) → 拒绝
                        stats["rejected_instances"] += 1
                        continue

                    # 保留完整 mask polygon(不 approxPolyDP 简化, 保真度优先)
                    coords = " ".join(f"{float(v):.6f}" for v in poly.reshape(-1))
                    accepted_lines.append(f"{int(cls_id)} {coords}")
                    accepted_confs.append(float(score))
                    accepted_classes.append(int(cls_id))
                    stats["accepted_instances"] += 1

            if accepted_lines:
                safe_link_or_copy(src, out_img)
                out_lbl.write_text("\n".join(accepted_lines) + "\n", encoding="utf-8")
                stats["accepted_images"] += 1
                status = "accepted"
            elif include_empty:
                safe_link_or_copy(src, out_img)
                out_lbl.write_text("", encoding="utf-8")
                stats["accepted_images"] += 1
                stats["empty_images_included"] += 1
                status = "empty_included"
            else:
                status = "rejected_no_confident_mask"

            writer.writerow(
                {
                    "source_image": str(src),
                    "combined_image": str(out_img) if out_img.exists() else "",
                    "accepted_instances": len(accepted_lines),
                    "mean_confidence": (
                        f"{float(np.mean(accepted_confs)):.5f}" if accepted_confs else ""
                    ),
                    "classes": ",".join(map(str, sorted(set(accepted_classes)))),
                    "status": status,
                }
            )

    return stats


def train_stage1(args: argparse.Namespace, data_yaml: Path) -> tuple[YOLO, Path]:
    """Stage 1: 训练 teacher(默认从零; 传 --pretrained-weights 则在其上 fine-tune)。

    关键设计:
      - 默认 YOLO(args.model_yaml) 加载"架构 yaml"(随机初始化权重), 不是 .pt。
      - pretrained=False 明确禁止加载外部预训练权重 —— 这是脚本的核心理念:
        "从零 + 自动标注" 保证 teacher 的每个知识都来自你自己的数据。
      - teacher 越"干净"(只学自动标注), 后续伪标注越可信。
      - 若传 --pretrained-weights <existing.pt>, 则在已有权重上 fine-tune(迁移学习),
        用于"有了新数据集 / 已有 best.pt 后再训练"的场景。
    """
    if args.pretrained_weights is not None:
        print("\n========== STAGE 1: FINE-TUNE FROM PRETRAINED WEIGHTS ==========")
        print(f"Pretrained weights: {args.pretrained_weights}")
        model = YOLO(str(args.pretrained_weights))
    else:
        print("\n========== STAGE 1: AUTO-LABELED DATA / FROM SCRATCH ==========")
        print(f"Architecture: {args.model_yaml}")
        print("External pretrained weights: NONE")
        # YAML => new architecture with random weights. No .pt is loaded here.
        model = YOLO(args.model_yaml)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs_stage1,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        pretrained=False,
        project=str((args.work_dir / "training").resolve()),
        name="stage1_auto_labeled_scratch",
        exist_ok=True,
    )

    # 彻底释放训练后的 model(含 trainer/ema/optimizer 持有的 allocated 显存)。
    # 仅 empty_cache 不够 —— 它只释放缓存池, 不释放被 Python 对象引用的显存。
    del model
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    best = args.work_dir / "training" / "stage1_auto_labeled_scratch" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Stage-1 best.pt not found: {best}")
    return YOLO(str(best)), best


def train_stage2(args: argparse.Namespace, combined_yaml: Path, teacher_best: Path) -> Path:
    """Stage 2: 用"自动标注 + 伪标注"合并数据训练 student(最终模型)。

    关键设计:
      - 默认 warm-start: student 从 Stage-1 teacher 权重继续(而非从零),
        因为 teacher 已经学会了 blade 基础特征, warm-start 收敛更快。
      - 注意 warm-start 用的仍是"你自己的 Stage-1 权重", 不是外部预训练。
      - --stage2-from-scratch 可从零(对比实验用)。
    """
    print("\n========== STAGE 2: AUTO-LABELED + PSEUDO LABELS ==========")
    if args.stage2_from_scratch:
        print("Stage 2 initialization: fresh random YOLO26 weights")
        student = YOLO(args.model_yaml)
        pretrained_flag = False
    else:
        print(f"Stage 2 initialization: your own Stage-1 weights: {teacher_best}")
        student = YOLO(str(teacher_best))
        pretrained_flag = False  # no external pretrained checkpoint is requested

    student.train(
        data=str(combined_yaml),
        epochs=args.epochs_stage2,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        pretrained=pretrained_flag,
        project=str((args.work_dir / "training").resolve()),
        name="stage2_auto_labeled_plus_pseudo",
        exist_ok=True,
    )

    best = args.work_dir / "training" / "stage2_auto_labeled_plus_pseudo" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Stage-2 best.pt not found: {best}")
    return best


def main() -> None:
    args = parse_args()
    # device 条件化: auto → 有 GPU 用 0, 无则 cpu (torch.cuda 是权威判断)
    if args.device == "auto":
        import torch
        args.device = "0" if torch.cuda.is_available() else "cpu"
        kind = "GPU" if args.device == "0" else "CPU"
        print(f"[device] auto → 检测到 {kind}, 使用 device={args.device}")
    args.labeled_root = args.labeled_root.expanduser().resolve()
    args.unlabeled_dir = args.unlabeled_dir.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    validate_input_layout(args.labeled_root)
    if not args.unlabeled_dir.exists():
        raise FileNotFoundError(args.unlabeled_dir)

    names = [x.strip() for x in args.names.split(",") if x.strip()]
    if not names:
        raise ValueError("--names must contain at least one class")

    auto_labeled_yaml = write_dataset_yaml(
        args.work_dir / "auto_labeled_only.yaml",
        args.labeled_root / "images" / "train",
        args.labeled_root / "images" / "val",
        names,
        args.labeled_root / "images" / "test",
    )

    # ── Stage 1: 自动标注 → 从零训练 teacher ──────────
    # 为什么从零(pretrained=False): 脚本明确不用外部预训练权重, 只靠
    # 自动标注数据学习 blade 特征。teacher 是"弱但干净"的模型(只见过
    # 自动标注), 用它给无标注数据打伪标注才可信。
    if args.skip_stage1:
        if args.teacher_weights is None:
            raise ValueError("--skip-stage1 requires --teacher-weights")
        teacher_best = args.teacher_weights.expanduser().resolve()
        if not teacher_best.exists():
            raise FileNotFoundError(teacher_best)
        teacher = YOLO(str(teacher_best))
    else:
        teacher, teacher_best = train_stage1(args, auto_labeled_yaml)

    # ── Build combined train set ──────────────────────
    # combined_root 每次重建(rmtree): 避免上次运行残留的伪标注污染本次
    combined_root = args.work_dir / "combined_dataset"
    if combined_root.exists():
        shutil.rmtree(combined_root)
    (combined_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (combined_root / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # 先把自动标注训练数据复制/链接进 combined。
    real_count = stage_real_training_data(args.labeled_root, combined_root)
    print(f"\nAuto-labeled training images staged: {real_count}")

    # ── Stage 2(伪标注): teacher 对无标注图生成高置信度伪标注 ──
    # 伪标注 = teacher.predict 的 mask polygon, 用 conf>=0.75 保守过滤,
    # 只有"teacher 很有把握"的 mask 才进 combined(避免错误伪标注伤害训练)
    print("\n========== PSEUDO LABEL UNLABELED IMAGES ==========")
    pseudo_stats = generate_pseudo_labels(
        teacher=teacher,
        unlabeled_dir=args.unlabeled_dir,
        combined_root=combined_root,
        pseudo_csv=args.work_dir / "pseudo_label_audit.csv",
        imgsz=args.imgsz,
        device=args.device,
        conf_threshold=args.pseudo_conf,
        iou_threshold=args.pseudo_iou,
        min_polygon_points=args.min_polygon_points,
        min_area_ratio=args.min_area_ratio,
        include_empty=args.include_empty_pseudo,
    )
    for k, v in pseudo_stats.items():
        print(f"{k}: {v}")

    # ── 合并后的 data.yaml: train=自动标注+伪标注, val=只自动标注 ──
    # 为什么 val 不用伪标注: 伪标注是 teacher 自己预测的(有噪声), 若用
    # 伪标注做验证, mAP 会虚高(模型"自己考自己")。val 必须只含自动标注,
    # 才能真实反映 student 的泛化能力。
    combined_yaml = write_dataset_yaml(
        args.work_dir / "auto_labeled_plus_pseudo.yaml",
        combined_root / "images" / "train",
        args.labeled_root / "images" / "val",  # REAL labels only
        names,
        args.labeled_root / "images" / "test",
    )

    print("\nDataset YAMLs:")
    print(f"  auto-labeled-only:   {auto_labeled_yaml}")
    print(f"  auto-labeled+pseudo: {combined_yaml}")
    print(f"  pseudo audit CSV: {args.work_dir / 'pseudo_label_audit.csv'}")

    if args.skip_stage2:
        print("\n--skip-stage2 set. Finished after pseudo-label generation.")
        return

    stage2_best = train_stage2(args, combined_yaml, teacher_best)

    print("\n================ DONE ================")
    print(f"Stage-1 teacher: {teacher_best}")
    print(f"Stage-2 best:    {stage2_best}")
    print("Validation/test remained auto-labeled only.")
    print("Review pseudo_label_audit.csv before trusting Stage-2 results.")


if __name__ == "__main__":
    main()
