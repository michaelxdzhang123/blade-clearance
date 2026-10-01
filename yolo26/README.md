## train_from_video.py video-->labling-->bbox-->dataset-blade/images/val -->training -->model best.bt based on COCO dataset
#1. video --> catch-blade.py 

把 catch_blade.py 的叶片框(blade_boxes.csv) 转成 YOLO 监督学习数据集。
这是"无监督方法(背景差分)自动生成监督学习(YOLO)训练数据"的桥梁:
  视频 → catch_blade 背景差分检测 → blade_boxes.csv (像素框)
       → 本脚本 → 抽帧图片 + YOLO 归一化标注 → dataset/ + data.yaml
#2. yolo26/make_yolo_dataset.py — 视频 → 自动标注 → YOLO 训练数据集
关键设计:
  - 均匀抽样: 10720 帧高度冗余(25fps 横扫周期 ~2.5s), 每 N 行取 1 帧
  - 时间顺序划分 train/val: 前 80% train, 后 20% val, 避免相邻帧泄漏
    (单视频场景用时间顺序; 多视频场景用 train_from_videos.py 的按视频划分)
  - YOLO 标注: <class_id> <x_center> <y_center> <width> <height> (归一化 0~1)

用法:
  python yolo26/make_yolo_dataset.py \
      --video data/training-data/test01-video-2026-08-28_134655_607.mp4 \
      --boxes blade-box/blade_boxes.csv \
      --out dataset_blade \
      --every 4
#3. 
