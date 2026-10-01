#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase1b: 精细分析 diff 时间序列, 区分真 Blade Pass / IDLE / 动态背景噪声"""
import numpy as np, csv, os

OUT = os.path.expanduser("~/projects/blade-clearance/outputs/TASK2_GATE_CLOSURE")
rows = list(csv.reader(open(os.path.join(OUT, "motion_timeline.csv"))))
d = np.array([r[2] for r in rows[1:]], dtype=float)
N = len(d)
print(f"总帧间差分: {N}")

# 真正的 Blade Pass = 高 diff 峰 (>8)
hi_thr = 8.0
print(f"\n=== 真 Blade Pass (diff > {hi_thr}) ===")
hi_mask = d > hi_thr
seg = []; s = None
for i, m in enumerate(hi_mask):
    if m and s is None: s = i
    elif not m and s is not None:
        seg.append((s, i, d[s:i].max(), d[s:i].mean())); s = None
if s is not None: seg.append((s, N, d[s:].max(), d[s:].mean()))
print(f"{len(seg)} 个高运动峰:")
for f0, f1, mx, mn in seg:
    print(f"  frame {f0}~{f1} (t={f0/25:.1f}~{f1/25:.1f}s) peak={mx:.1f} mean={mn:.1f}")

# 低运动段 (可能的 IDLE), 用更宽松阈值
for thr in [0.3, 0.5, 0.8, 1.0]:
    lm = d < thr
    s = None; low_seg = []
    for i, m in enumerate(lm):
        if m and s is None: s = i
        elif not m and s is not None:
            if i - s >= 10: low_seg.append((s, i, d[s:i].mean()))
            s = None
    if s is not None and N - s >= 10: low_seg.append((s, N, d[s:].mean()))
    print(f"\ndiff<{thr}: {len(low_seg)} 个 >=10帧低运动段")
    for f0, f1, mn in sorted(low_seg, key=lambda x: x[2])[:8]:
        print(f"  frame {f0}~{f1} (t={f0/25:.1f}~{f1/25:.1f}s, {f1-f0}帧) mean_diff={mn:.2f}")

# 全局最低 20 帧 (最可能无 Blade)
print("\n=== 全局最低运动 20 个时刻 ===")
idx = np.argsort(d)[:20]
for i in idx:
    print(f"  frame {i} (t={i/25:.2f}s) diff={d[i]:.2f}")
