import os, sys, numpy as np, cv2
sys.path.insert(0, '.')
from opencv.pipeline import static_frame_diff, motion_mask, dilate_mask, morphology, read_edges
from run_task1 import frame_paths, frame_ids, load_gray, load_registration, align_frame, W, H, RESULTS, TOWER_X

N = 30
paths = frame_paths(); ids = frame_ids()
grays = [load_gray(fp) for fp in paths[:N]]
reg = load_registration()

def classify(pts, mm, tower_band=40):
    in_tower = np.abs(pts[:, 0] - TOWER_X) < tower_band
    xi = np.clip(pts[:, 1].astype(int), 0, H - 1)
    yi = np.clip(pts[:, 0].astype(int), 0, W - 1)
    in_motion = mm[xi, yi] > 0
    keep = in_tower | (in_motion & ~in_tower)
    return len(pts), in_tower.sum(), (in_motion & ~in_tower).sum(), keep.sum()

# 方法1: MOG2 (learningRate=0 冻结)
print("=== MOG2 (learningRate=0) ===")
mog = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
reds = []
for i, g in enumerate(grays):
    dx, dy = reg.get(i, (0, 0)); ga = align_frame(g, dx, dy)
    fg = mog.apply(ga, learningRate=0)
    mm = dilate_mask(morphology(fg, op="close", ksize=5, iters=1), 5)
    xs, ys, resp, direc, nc = read_edges(os.path.join(RESULTS, f"raw_{ids[i]}.yml"))
    pts = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2))
    n_raw, n_t, n_b, n_keep = classify(pts, mm)
    reds.append((1 - n_keep / max(n_raw, 1), (mm > 0).mean()))
    if i in (0, 10, 20):
        print(f"帧{i}: raw={n_raw} tower={n_t} blade={n_b} keep={n_keep} "
              f"R_reduce={reds[-1][0]*100:.1f}% motion_area={reds[-1][1]*100:.1f}%")
a = np.array(reds)
print(f"MOG2: R_reduce mean={a[:,0].mean()*100:.1f}%  motion_area mean={a[:,1].mean()*100:.1f}%")

# 方法2: 分块 median 背景
print("\n=== 分块 median 背景 (float32) ===")
arr = np.stack(grays).astype(np.float32)
bg_med = np.median(arr, axis=0).astype(np.uint8)
reds2 = []
for i, g in enumerate(grays):
    dx, dy = reg.get(i, (0, 0)); ga = align_frame(g, dx, dy)
    diff = static_frame_diff(bg_med, ga)
    mm = dilate_mask(morphology(motion_mask(diff, 30), op="close", ksize=5, iters=1), 5)
    xs, ys, resp, direc, nc = read_edges(os.path.join(RESULTS, f"raw_{ids[i]}.yml"))
    pts = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2))
    n_raw, n_t, n_b, n_keep = classify(pts, mm)
    reds2.append((1 - n_keep / max(n_raw, 1), (mm > 0).mean()))
b = np.array(reds2)
print(f"median: R_reduce mean={b[:,0].mean()*100:.1f}%  motion_area mean={b[:,1].mean()*100:.1f}%")
