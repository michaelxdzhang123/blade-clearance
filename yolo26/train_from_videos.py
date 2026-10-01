#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_from_videos.py — 多视频自动标注 + 训练 (YAML 驱动)

整体流程(五步, 见 main()):
  1. 读 train_config.yaml (视频列表 + 数据集/训练参数)
  2. 逐视频: build_bg_array 重建背景 → detect_video_boxes 检测叶片框
  3. 按视频划分 train/val (最后 n_val 个视频做 val, 测泛化)
  4. write_dataset 合并成 YOLO 数据集 (图片 + 归一化标注)
  5. YOLO26 迁移训练 blade 类 + 验证 mAP

核心思想:
  无监督(背景差分)自动生成标注 → 监督学习(YOLO26)迁移训练。
  每个视频背景不同(塔筒/光照), 必须各自重建背景, 不能共用。

用法:
  python yolo26/train_from_videos.py                      # 用默认 train_config.yaml
  python yolo26/train_from_videos.py --config my.yaml     # 指定配置
  python yolo26/train_from_videos.py --skip-train         # 只生成数据集, 不训练
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
# (否则 python yolo26/train_from_videos.py 时 sys.path[0] 是 yolo26/, 找不到 opencv 包)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencv.catch_blade import blade_boxes  # 复用叶片检测(背景差分+几何先验)

W, H = 2560, 1440    # 视频分辨率(归一化坐标的分母)
CLASS_ID = 0         # blade 类的 ID(单类检测)
CLASS_NAME = "blade"


def build_bg_array(video_path, n_idle=50):
    """从视频 IDLE 帧重建 median 背景, 返回 float32 数组 (H,W)。

    原理(两遍解码):
      - IDLE 帧 = 叶片静止、画面无运动的帧, 即"帧间差分最小"的帧。
        叶片横扫是周期性运动, IDLE 帧占多数, diff 小; 横扫帧 diff 大。
      - median(而非 mean)抗噪声: 叶片横扫的瞬时像素是少数, median 会把
        它们排到背景之外, 得到"干净背景"(塔筒+天空, 无叶片)。

    第一遍: 降采样 1/4(640×360) 算帧间差分, 找 diff 最小的 n_idle 帧。
            降采样加快计算, 帧间差分的统计意义不受影响。
    第二遍: 取这 n_idle 帧全分辨率灰度图, 逐像素 median 得背景。
    """
    # ── 第一遍: 降采样帧间差分 ──────────────────────────
    c = av.open(video_path)
    s = c.streams.video[0]
    prev = None
    diffs = []  # diffs[i] = |frame[i+1] - frame[i]| 的降采样均值
    for frame in c.decode(s):
        img = frame.to_ndarray(format="gray")
        small = img[::4, ::4].astype(np.float32)  # 降采样 1/4 → 640×360
        if prev is not None:
            diffs.append(float(np.abs(small - prev).mean()))
        prev = small
    c.close()

    diffs = np.array(diffs)
    # IDLE 帧号 = diff 最小的 n_idle 帧 (diff[i] 小 → 帧 i+1 静止, 故 +1)
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
        if not want:      # 全部取齐就提前退出, 不解码剩余帧
            break
    c.close()

    bg = np.median(np.stack(frames).astype(np.float32), axis=0)
    return bg


def detect_video_boxes(video_path, bg, every):
    """检测一个视频的叶片框, 返回 [(frame, x, y, w, h), ...] (已抽样)。

    逐帧调用 catch_blade.blade_boxes(img, bg) —— 复用其背景差分 + 几何先验:
      - |img - bg| > 25  分离运动叶片 vs 静止塔筒
      - 下层 y>800       排除上层塔筒
      - 触底 y+h≥1430    叶尖横扫到画面底端
      - 面积>5000        叶片=最大运动物体(塔筒阴影噪声<3000)

    最后 [::every] 抽样: 25fps 横扫周期~2.5s, 相邻帧高度冗余。
    """
    boxes = []
    c = av.open(video_path)
    s = c.streams.video[0]
    for idx, frame in enumerate(c.decode(s)):
        img = frame.to_ndarray(format="gray").astype(np.float32)
        box = blade_boxes(img, bg)
        if box is not None:
            boxes.append((idx, box[0], box[1], box[2], box[3]))
    c.close()
    return boxes[::every]


def write_dataset(video_boxes, videos, val_video_ids, out_dir):
    """生成 YOLO 数据集: 图片 + 归一化标注, 按视频划分 train/val。

    关键设计:
      - 按视频划分 val(而非视频内混合): 最后 n_val 个视频做 val。
        相邻帧几乎一样, 若视频内混合, val 会"见过"train 的近邻帧 → 虚高。
        按视频划分 = val 是模型"没见过的视频", mAP 真实反映泛化。
      - 文件名 v{vi}_f{idx}: 多视频帧号会重复, 加视频号前缀避免冲突。
      - YOLO 标注格式: <class> <cx> <cy> <w> <h> 全归一化到 [0,1]。
        像素框 (x,y,w,h) → 中心点 (cx,cy)=(x+w/2, y+h/2), 宽高除以 W,H。
    """
    # 四个输出目录
    dirs = {
        "train_img": os.path.join(out_dir, "images", "train"),
        "train_lbl": os.path.join(out_dir, "labels", "train"),
        "val_img": os.path.join(out_dir, "images", "val"),
        "val_lbl": os.path.join(out_dir, "labels", "val"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 逐个视频解码, 命中目标帧就抽图 + 写标注
    total = sum(len(vb) for vb in video_boxes)
    n_saved = 0
    for vi, vpath in enumerate(videos):
        split = "val" if vi in val_video_ids else "train"
        # 该视频的目标帧 → {frame号: (x,y,w,h)}
        target = {f: (x, y, w, h) for (f, x, y, w, h) in video_boxes[vi]}

        c = av.open(vpath)
        s = c.streams.video[0]
        for idx, frame in enumerate(c.decode(s)):
            if idx not in target:
                continue
            x, y, w, h = target[idx]
            bgr = frame.to_ndarray(format="bgr24")

            name = f"v{vi:02d}_f{idx:05d}"
            cv2.imwrite(os.path.join(dirs[f"{split}_img"], name + ".jpg"),
                        bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            # 像素框 → 归一化中心点+宽高 (YOLO 格式)
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            nw = w / W
            nh = h / H
            with open(os.path.join(dirs[f"{split}_lbl"], name + ".txt"), "w") as f:
                f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            n_saved += 1
        c.close()
        print(f"  视频 {vi}: 已写入 {len(target)} 帧 → {split}")
    print(f"总计 {n_saved} 帧 (目标 {total})")

    # data.yaml (相对路径, 数据集整体可搬到 GPU 服务器直接用)
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write(f"  {CLASS_ID}: {CLASS_NAME}\n")


def main():
    """编排五步流程: 读配置 → 逐视频标注 → 划分 val → 写数据集 → 训练。"""
    parser = argparse.ArgumentParser(description="多视频自动标注 + 训练")
    parser.add_argument("--config", default="train_config.yaml",
                        help="配置文件路径 (默认 train_config.yaml)")
    parser.add_argument("--out", default=None,
                        help="数据集输出目录 (覆盖 config 里的 dataset.output)")
    parser.add_argument("--skip-train", action="store_true",
                        help="只生成数据集, 不训练")
    args = parser.parse_args()

    # 读 YAML 配置: videos(视频列表) / dataset(抽样+val划分) / train(超参数)
    cfg = yaml.safe_load(open(args.config))
    videos = cfg["videos"]
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["train"]

    # ── 1. 逐视频自动标注 ──────────────────────────────
    # 每个视频独立: build_bg_array 重建背景 → detect_video_boxes 检测框
    # (背景不能共用, 因为不同视频的塔筒/光照/雾况不同)
    print(f"=== 自动标注 {len(videos)} 个视频 ===")
    video_boxes = []  # video_boxes[vi] = [(frame, x, y, w, h), ...]
    for vi, vpath in enumerate(videos):
        print(f"\n[{vi+1}/{len(videos)}] {vpath}")
        bg = build_bg_array(vpath)
        boxes = detect_video_boxes(vpath, bg, ds_cfg["every"])
        video_boxes.append(boxes)
        print(f"  检测到 {len(boxes)} 帧叶片框")

    # ── 2. 按视频划分 train/val ────────────────────────
    # 最后 n_val 个视频做 val(测泛化), 其余 train
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
    n_train = sum(len(vb) for i, vb in enumerate(video_boxes)
                  if i not in val_video_ids)
    n_val_n = sum(len(vb) for i, vb in enumerate(video_boxes)
                  if i in val_video_ids)
    print(f"\n=== 生成数据集 ===")
    print(f"train 视频 {len(videos)-n_val} 个 ({n_train} 帧), "
          f"val 视频 {n_val} 个 ({n_val_n} 帧)")

    # ── 3. 写数据集 ────────────────────────────────────
    out_dir = args.out or ds_cfg["output"]  # CLI --out 优先, 否则用 config
    write_dataset(video_boxes, videos, val_video_ids, out_dir)
    print(f"数据集 → {out_dir}/")

    # ── 4. 训练 ───────────────────────────────────────
    if args.skip_train:
        print("\n跳过训练 (--skip-train)")
        print(f"手动训练: python yolo26/train_blade.py --data {out_dir}/data.yaml")
        return

    print(f"\n=== 开始训练 ===")
    model = YOLO(tr_cfg["model"])  # 加载预训练(如 yolo26n.pt), 迁移学习
    model.train(
        data=os.path.join(out_dir, "data.yaml"),
        epochs=tr_cfg.get("epochs", 100),
        imgsz=tr_cfg.get("imgsz", 640),
        batch=tr_cfg.get("batch", 16),
        device=str(tr_cfg.get("device", 0)),
        workers=tr_cfg.get("workers", 8),
        patience=tr_cfg.get("patience", 30),
        name="blade_train",
        nms=tr_cfg.get("nms", False),  # YOLO26: False=NMS-free one-to-one head
        project=out_dir,  # 训练输出直接写到 --out 目录(而非默认 runs/detect/)
    )

    # ── 5. 验证 ───────────────────────────────────────
    # best.pt 在 out_dir/blade_train/weights/best.pt (project=out_dir + name)
    best_path = os.path.join(out_dir, "blade_train", "weights", "best.pt")
    if not os.path.exists(best_path):
        print(f"⚠️ 未找到 {best_path}, 检查训练是否成功/输出路径")
        return
    metrics = YOLO(best_path).val(
        data=os.path.join(out_dir, "data.yaml"),
        imgsz=tr_cfg.get("imgsz", 640),
        device=str(tr_cfg.get("device", 0)),
    )
    print(f"\n最佳权重: {best_path}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
