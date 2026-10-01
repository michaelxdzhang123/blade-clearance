import os, sys, numpy as np, cv2
sys.path.insert(0, '.')
from opencv.pipeline import static_frame_diff, motion_mask, dilate_mask, morphology, read_edges
from run_task1 import frame_paths, frame_ids, load_gray, compute_bg, load_registration, align_frame, W, H, RESULTS, TOWER_X

N = 30
paths = frame_paths(); ids = frame_ids()
grays = [load_gray(fp) for fp in paths[:N]]
bg = compute_bg(grays); reg = load_registration()
tower_band = 40; thr = 40; dp = 5
stats = []
for i, g in enumerate(grays):
    dx, dy = reg.get(i, (0, 0)); ga = align_frame(g, dx, dy)
    diff = static_frame_diff(bg, ga)
    mm = dilate_mask(morphology(motion_mask(diff, thr), op="close", ksize=5, iters=1), dp)
    xs, ys, resp, direc, nc = read_edges(os.path.join(RESULTS, f"raw_{ids[i]}.yml"))
    pts = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2))
    n_raw = len(pts)
    in_tower = np.abs(pts[:, 0] - TOWER_X) < tower_band
    xi = np.clip(pts[:, 1].astype(int), 0, H - 1)
    yi = np.clip(pts[:, 0].astype(int), 0, W - 1)
    in_motion = mm[xi, yi] > 0
    keep = in_tower | (in_motion & ~in_tower)
    n_keep = keep.sum()
    r_reduce = 1 - n_keep / max(n_raw, 1)
    stats.append((n_raw, in_tower.sum(), (in_motion & ~in_tower).sum(), n_keep, r_reduce))
    if i in (0, 10, 20):
        print(f"帧{i}: raw={n_raw} tower={in_tower.sum()} blade={(in_motion & ~in_tower).sum()} "
              f"keep={n_keep} R_reduce={r_reduce*100:.1f}%")
a = np.array(stats)
print(f"\n[正确架构] Tower∪Blade 保留 edge: mean={a[:,3].mean():.0f} ({100*a[:,3].mean()/a[:,0].mean():.1f}%)")
print(f"  R_reduce (Static Edge Suppression) = {100*(1-a[:,3].mean()/a[:,0].mean()):.1f}%")
print(f"  其中 Tower edge 占 {100*a[:,1].mean()/a[:,0].mean():.1f}%, Blade edge 占 {100*a[:,2].mean()/a[:,0].mean():.1f}%")
