#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_from_videos_seg.py — 多视频自动标注 + 实例分割训练 (YOLO26-seg)

是 train_from_videos.py 的 seg 版。区别只在三处:
  1. 检测: blade_boxes(return_polygon=True) 返回 polygon 轮廓(而非 bbox 矩形)
  2. 标注: YOLO-seg 格式 <class> <x1 y1> <x2 y2> ... <xn yn>(归一化多边形顶点)
  3. 训练: yolo26n-seg.pt 实例分割权重(而非 yolo26n.pt 检测权重)

整体流程(五步, 见 main()):
  1. 读 train_config.yaml
  2. 逐视频: build_bg_array 重建背景 → detect_video_polygons 检测叶片 polygon
  3. 按视频划分 train/val
  4. write_dataset 合并成 YOLO-seg 数据集(图片 + polygon 标注)
  5. YOLO26-seg 迁移训练 blade 类 + 验证 box/mask mAP

用法:
  python yolo26/train_from_videos_seg.py                      # 用默认 train_config.yaml
  python yolo26/train_from_videos_seg.py --config my.yaml     # 指定配置
  python yolo26/train_from_videos_seg.py --skip-train         # 只生成数据集, 不训练
"""
import argparse
import os
import sys

import av
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# 把项目根加入 sys.path, 使 from opencv.xxx import 从任何目录运行都能找到
# (否则 python yolo26/train_from_videos_seg.py 时 sys.path[0] 是 yolo26/, 找不到 opencv 包)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencv.catch_blade import blade_boxes  # 复用叶片检测(背景差分+几何先验)

WRITE_IMG_SIZE = 640  # 数据集写入图片边长
CLASS_ID = 0         # blade 类的 ID(单类)
CLASS_NAME = "blade"


def build_bg_array(video_path, n_idle=50):
    """从视频 IDLE 帧重建 median 背景, 返回 float32 数组 (H,W)。

    原理(两遍解码): IDLE 帧 = 帧间差分最小的帧(画面静止), 取 diff 最小
    的 n_idle 帧做逐像素 median(非 mean, 抗叶片横扫的瞬时噪声)。
    第一遍降采样 1/4 算 diff 找 IDLE; 第二遍取 IDLE 帧全分辨率做 median。
    """
    # ── 第一遍: 降采样帧间差分 ──────────────────────────
    c = av.open(video_path)
    s = c.streams.video[0]
    prev = None
    diffs = []
    for frame in c.decode(s):
        img = frame.to_ndarray(format="gray")
        small = img[::4, ::4].astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(small - prev).mean()))
        prev = small
    c.close()

    diffs = np.array(diffs)
    idle_frames = sorted((np.argsort(diffs)[:n_idle] + 1).tolist())

    # ── 第二遍: 取 IDLE 帧做 median 背景 ─────────────────
    c = av.open(video_path)
    s = c.streams.video[0]
    want = set(idle_frames)
    frames = []
    for idx, frame in enumerate(c.decode(s)):
        if idx in want:
            frames.append(frame.to_ndarray(format="gray"))
            want.discard(idx)
        if not want:
            break
    c.close()

    bg = np.median(np.stack(frames).astype(np.float32), axis=0)
    return bg


def detect_video_polygons(video_path, bg, every):
    """检测一个视频的叶片 polygon, 返回 [(frame, polygon), ...] (已抽样)。

    与 detect 版的区别: blade_boxes(return_polygon=True) 返回 polygon
    轮廓顶点 [[x1,y1],[x2,y2],...] (像素坐标), 而非 bbox 矩形。
    polygon 来自最大连通域的 mask 的 findContours 轮廓, 精确到叶片边缘。
    """
    polygons = []
    c = av.open(video_path)
    s = c.streams.video[0]
    for idx, frame in enumerate(c.decode(s)):
        img = frame.to_ndarray(format="gray").astype(np.float32)
        poly = blade_boxes(img, bg, return_polygon=True)
        if poly is not None:
            polygons.append((idx, poly))
    c.close()
    return polygons[::every]


def letterbox_square(image, size):
    """保持宽高比 resize 到 size×size 正方形(letterbox), 与 create_unlabeled_dataset.py 一致。

    返回 (canvas, scale, left, top): canvas 为 letterbox 后的 size×size 图;
    scale/left/top 用于把原图坐标同步变换到 letterbox 后的坐标。
    """
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=image.dtype)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, left, top


def write_dataset(video_polygons, videos, val_video_ids, out_dir):
    """生成 YOLO-seg 数据集: 图片 + polygon 标注, 按视频划分 train/val。

    YOLO-seg 标注格式(与 detect 版的 bbox 格式不同):
      每行一个实例: <class> <x1> <y1> <x2> <y2> ... <xn> <yn>
      顶点全归一化到 [0,1], 空格分隔, 一行放整个多边形。
    detect 版写 4 个数字(cx cy w h), seg 版写 2N 个数字(N 个顶点)。
    """
    dirs = {
        "train_img": os.path.join(out_dir, "images", "train"),
        "train_lbl": os.path.join(out_dir, "labels", "train"),
        "val_img": os.path.join(out_dir, "images", "val"),
        "val_lbl": os.path.join(out_dir, "labels", "val"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    total = sum(len(vp) for vp in video_polygons)
    n_saved = 0
    for vi, vpath in enumerate(videos):
        split = "val" if vi in val_video_ids else "train"
        # 该视频的目标帧 → {frame号: polygon}
        target = {f: poly for (f, poly) in video_polygons[vi]}

        c = av.open(vpath)
        s = c.streams.video[0]
        for idx, frame in enumerate(c.decode(s)):
            if idx not in target:
                continue
            poly = target[idx]
            bgr = frame.to_ndarray(format="bgr24")

            name = f"v{vi:02d}_f{idx:05d}"
            out_img, scale, pad_l, pad_t = letterbox_square(bgr, WRITE_IMG_SIZE)
            cv2.imwrite(os.path.join(dirs[f"{split}_img"], name + ".jpg"),
                        out_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            # polygon 顶点: 先 scale+pad, 再归一化到 [0,1]
            pts = " ".join(
                f"{(x * scale + pad_l) / WRITE_IMG_SIZE:.6f} {(y * scale + pad_t) / WRITE_IMG_SIZE:.6f}"
                for x, y in poly
            )
            with open(os.path.join(dirs[f"{split}_lbl"], name + ".txt"), "w") as f:
                f.write(f"{CLASS_ID} {pts}\n")
            n_saved += 1
        c.close()
        print(f"  视频 {vi}: 已写入 {len(target)} 帧 polygon → {split}")
    print(f"总计 {n_saved} 帧 (目标 {total})")

    # data.yaml (相对路径, 数据集整体可搬到 GPU 服务器直接用)
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write(f"  {CLASS_ID}: {CLASS_NAME}\n")


def main():
    """编排五步流程: 读配置 → 逐视频 polygon 标注 → 划分 val → 写数据集 → seg 训练。"""
    parser = argparse.ArgumentParser(description="多视频自动标注 + 实例分割训练")
    parser.add_argument("--config", default="train_config.yaml",
                        help="配置文件路径 (默认 train_config.yaml)")
    parser.add_argument("--out", default=None,
                        help="数据集输出目录 (覆盖 config 里的 dataset.output)")
    parser.add_argument("--skip-train", action="store_true",
                        help="只生成数据集, 不训练")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="训练 imgsz(覆盖 config 的 train.imgsz); 只影响训练, 不影响数据集写入尺寸(WRITE_IMG_SIZE)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    videos = cfg["videos"]
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["train"]
    train_imgsz = args.imgsz or tr_cfg.get("imgsz", 640)

    # ── 1. 逐视频 polygon 标注 ─────────────────────────
    print(f"=== 自动标注 {len(videos)} 个视频 (polygon 模式) ===")
    video_polygons = []  # video_polygons[vi] = [(frame, polygon), ...]
    for vi, vpath in enumerate(videos):
        print(f"\n[{vi+1}/{len(videos)}] {vpath}")
        bg = build_bg_array(vpath)
        polys = detect_video_polygons(vpath, bg, ds_cfg["every"])
        video_polygons.append(polys)
        print(f"  检测到 {len(polys)} 帧叶片 polygon")

    # ── 2. 按视频划分 train/val ────────────────────────
    n_val = ds_cfg.get("val_videos", 2)
    # 保证 train 至少 1 个视频: 当视频数 ≤ val_videos 时, val 缩减为 len-1。
    # 否则 range(len-n_val, len) 从负数开始, 会把所有视频划进 val, train 空。
    if n_val >= len(videos):
        n_val = max(len(videos) - 1, 0)
        if len(videos) == 1:
            print(f"⚠️ 只有 1 个视频, 无法按视频划分 train/val: 全部划为 train, val 空。")
            print(f"   建议: 提供 ≥2 个视频, 或用单视频时间顺序划分(make_yolo_dataset.py)。")
        else:
            print(f"⚠️ 视频数({len(videos)})不足以划分 {ds_cfg.get('val_videos', 2)} 个 val, "
                  f"调整为 {n_val} 个 val / {len(videos)-n_val} 个 train")
    val_video_ids = set(range(len(videos) - n_val, len(videos)))
    n_train = sum(len(vp) for i, vp in enumerate(video_polygons)
                  if i not in val_video_ids)
    n_val_n = sum(len(vp) for i, vp in enumerate(video_polygons)
                  if i in val_video_ids)
    print(f"\n=== 生成数据集 ===")
    print(f"train 视频 {len(videos)-n_val} 个 ({n_train} 帧), "
          f"val 视频 {n_val} 个 ({n_val_n} 帧)")

    # ── 3. 写数据集 ────────────────────────────────────
    out_dir = args.out or ds_cfg["output"]  # CLI --out 优先, 否则用 config
    write_dataset(video_polygons, videos, val_video_ids, out_dir)
    print(f"数据集 → {out_dir}/")

    # ── 4. seg 训练 ────────────────────────────────────
    if args.skip_train:
        print("\n跳过训练 (--skip-train)")
        print(f"手动训练: python yolo26/train_blade.py --data {out_dir}/data.yaml")
        return

    # seg 权重: 若 config 的 model 是 yolo26n.pt(检测), 自动换成 -seg 版
    model_name = tr_cfg.get("model", "yolo26n-seg.pt")
    if "seg" not in model_name and model_name.endswith(".pt"):
        model_name = model_name.replace(".pt", "-seg.pt")
    print(f"\n=== 开始实例分割训练 (模型 {model_name}) ===")
    model = YOLO(model_name)
    model.train(
        data=os.path.join(out_dir, "data.yaml"),
        epochs=tr_cfg.get("epochs", 100),
        imgsz=train_imgsz,
        batch=tr_cfg.get("batch", 16),
        device=str(tr_cfg.get("device", 0)),
        workers=tr_cfg.get("workers", 8),
        patience=tr_cfg.get("patience", 30),
        name="blade_train_seg",
        project=out_dir,  # 训练输出直接写到 --out 目录(而非默认 runs/segment/)
    )

    # ── 5. 验证 (seg 输出 out_dir/blade_train_seg/, 有 box + mask 两个 mAP) ──
    best_path = os.path.join(out_dir, "blade_train_seg", "weights", "best.pt")
    if not os.path.exists(best_path):
        print(f"⚠️ 未找到 {best_path}, 检查训练是否成功/输出路径")
        return
    metrics = YOLO(best_path).val(
        data=os.path.join(out_dir, "data.yaml"),
        imgsz=train_imgsz,
        device=str(tr_cfg.get("device", 0)),
    )
    print(f"\n最佳权重: {best_path}")
    print(f"box  mAP50:    {metrics.box.map50:.4f}")   # 检测框精度
    print(f"box  mAP50-95: {metrics.box.map:.4f}")
    if hasattr(metrics, "seg") and metrics.seg is not None:
        print(f"mask mAP50:    {metrics.seg.map50:.4f}")   # 分割掩码精度
        print(f"mask mAP50-95: {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
