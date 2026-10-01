#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证: Blade narrow band 内 Local EdgesSubPix 的 runtime 与 edge 质量"""
import av, numpy as np, cv2, os, subprocess, time

VIDEO = os.path.expanduser("~/projects/blade-clearance/data/test01-video-2026-08-28_134655_607.mp4")
OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
EDGE_BIN = "/home/mich/projects/EdgesSubPix/build/Edge"

def extract(f0, f1):
    container = av.open(VIDEO); stream = container.streams.video[0]; stream.thread_type = "AUTO"
    want = set(range(f0, f1)); frames = {}
    for idx, frame in enumerate(container.decode(stream)):
        if idx in want:
            frames[idx] = frame.to_ndarray(format="gray")
            want.discard(idx)
        if not want: break
    container.close()
    ks = sorted(frames.keys())
    return [frames[k] for k in ks], ks

frames, nos = extract(1618, 1650)
print(f"Blade 主体段 {len(frames)} 帧")

# 对每帧: diff>30 最大连通域 bbox, 在 bbox 内跑 EdgesSubPix, 记录 runtime
os.makedirs(os.path.join(OUT, "local_runtime"), exist_ok=True)
times = []
bbox_areas = []
for i in range(1, len(frames)):
    d = np.abs(frames[i].astype(float) - frames[i-1].astype(float))
    m30 = (d > 30).astype(np.uint8)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(m30, connectivity=8)
    if n <= 1:
        continue
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x0 = stats[largest, cv2.CC_STAT_LEFT]; y0 = stats[largest, cv2.CC_STAT_TOP]
    w = stats[largest, cv2.CC_STAT_WIDTH]; h = stats[largest, cv2.CC_STAT_HEIGHT]
    # margin
    x0 = max(0, x0-10); y0 = max(0, y0-10)
    w = min(2560-x0, w+20); h = min(1440-y0, h+20)
    crop = frames[i][y0:y0+h, x0:x0+w]
    img = os.path.join(OUT, "local_runtime", f"c{nos[i]}.png")
    cv2.imwrite(img, crop)
    yml = os.path.join(OUT, "local_runtime", f"c{nos[i]}.yml")
    t0 = time.time()
    subprocess.run([EDGE_BIN, img, yml.replace(".yml", "_o.jpg"),
                    f"--data={yml}", "--alpha=1.0", "--low=20", "--high=40", "--mode=1"],
                   capture_output=True, timeout=120)
    dt = time.time() - t0
    times.append(dt); bbox_areas.append(w*h)

times = np.array(times); bbox_areas = np.array(bbox_areas)
print(f"\n=== Local EdgesSubPix (Blade narrow band) runtime ===")
print(f"crop 尺寸: mean={np.sqrt(bbox_areas.mean()):.0f}x{np.sqrt(bbox_areas.mean()):.0f}px "
      f"(面积占全图 {100*bbox_areas.mean()/(2560*1440):.1f}%)")
print(f"runtime: mean={times.mean()*1000:.0f}ms  p95={np.percentile(times,95)*1000:.0f}ms")
print(f"全图 EdgesSubPix = 964ms/帧")
print(f"加速比 = {964/(times.mean()*1000):.1f}x")
