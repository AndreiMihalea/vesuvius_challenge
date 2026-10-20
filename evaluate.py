"""Step 3: score the validation fold with the official leaderboard metric.
Sliding-window inference -> threshold [hysteresis] -> [split merged sheets] -> drop small blobs -> topometrics (TopoScore, SurfaceDice@2, VOI).
Probabilities are cached per checkpoint in work/probs/, so post-processing variants only cost metric time."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import topometrics.leaderboard as lb  # installed by setup_metric.sh
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from common import IGNORE, WORK, UNet3D, load, predict

CONN26 = np.ones((3, 3, 3), bool)


def split_sheets(prob, mask, seed_thr, min_seed=1000):
    """Seeded watershed: confident cores (prob > seed_thr) flood the mask along high probability.
    Unseeded blobs are dropped; where two sheets meet, the lower label loses its contact voxels (1-voxel gap)."""
    seeds, _ = ndi.label(prob > seed_thr, CONN26)
    seeds[(np.bincount(seeds.ravel()) < min_seed)[seeds]] = 0
    lab = watershed(-prob, seeds, mask=mask)
    return (lab > 0) & ~(ndi.maximum_filter(lab, size=3) > lab)


def hysteresis(prob, low, high):
    """Keep the 26-connected components of prob > low that contain at least one confident voxel (prob > high)."""
    cc, _ = ndi.label(prob > low, CONN26)
    keep = np.zeros(cc.max() + 1, bool)
    keep[np.unique(cc[prob > high])] = True
    keep[0] = False
    return keep[cc]


def postprocess(prob, thr, min_size, seed_thr=None, hyst_high=None):
    mask = prob > thr if hyst_high is None else hysteresis(prob, thr, hyst_high)
    if seed_thr is not None:
        mask = split_sheets(prob, mask, seed_thr)
    cc, _ = ndi.label(mask, CONN26)
    keep = np.bincount(cc.ravel()) >= min_size
    keep[0] = False
    return keep[cc]


def official_score(pred, lab):
    """Leaderboard: 0.30 * TopoScore + 0.35 * SurfaceDice@2 + 0.35 * VOI_score, label 2 ignored."""
    r = lb.compute_leaderboard_score(
        predictions=pred.astype(np.uint8), labels=lab.astype(np.uint8),
        dims=(0, 1, 2), spacing=(1.0, 1.0, 1.0), surface_tolerance=2.0,
        voi_connectivity=26, voi_transform="one_over_one_plus", voi_alpha=0.3,
        combine_weights=(0.3, 0.35, 0.35), fg_threshold=None,
        ignore_label=IGNORE, ignore_mask=None)
    return dict(score=float(r.score), topo_score=float(r.topo.toposcore),
                surface_dice=float(r.surface_dice), voi_score=float(r.voi.voi_score),
                **{f"topo_f1_k{k}": float(v) for k, v in r.topo.topoF1_by_dim.items()})


def score_volume(prob, lab, thr, min_size, seed_thr, hyst_high):
    return official_score(postprocess(prob.astype(np.float32), thr, min_size, seed_thr, hyst_high), lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--thr", type=float, default=0.26)
    ap.add_argument("--min-size", type=int, default=20000, help="drop predicted components smaller than this")
    ap.add_argument("--seed-thr", type=float, help="split merged sheets by seeded watershed from prob > seed-thr")
    ap.add_argument("--hyst-high", type=float, help="hysteresis: --thr grows, but only blobs with prob > this survive")
    ap.add_argument("--n", type=int, help="only the first N validation volumes (quick sweeps)")
    ap.add_argument("--out", default="val_metrics.csv")
    ap.add_argument("--workers", type=int, default=12, help="CPU processes for post-processing + metric")
    ap.add_argument("--probs-dir", help="score nnU-Net's saved probabilities (<id>.npz from `nnUNetv2_train --npz`) "
                                        "instead of our model")
    args = ap.parse_args()

    ckpt = WORK / args.ckpt
    split = pd.read_csv(WORK / "split.csv")
    ids = split[split.split == "val"].id.tolist()[:args.n]

    def get_prob(vid):
        nonlocal model
        if args.probs_dir:
            return np.load(Path(args.probs_dir) / f"{vid}.npz")["probabilities"][1]  # channel 1 = surface
        f = WORK / "probs" / f"{ckpt.stem}_{int(ckpt.stat().st_mtime)}" / f"{vid}.npy"  # retrained ckpt -> new cache
        if not f.exists():
            if model is None:
                model = UNet3D().cuda().eval()
                model.load_state_dict(torch.load(ckpt))
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, predict(model, load(vid, "images")).astype(np.float16))
        return np.load(f)

    # GPU predicts (or a cache provides) volumes one by one; CPU processes post-process + score in parallel
    model, jobs = None, []
    with ProcessPoolExecutor(args.workers) as pool:
        for vid in ids:
            jobs.append(pool.submit(score_volume, get_prob(vid), np.asarray(load(vid, "labels")),
                                    args.thr, args.min_size, args.seed_thr, args.hyst_high))
        rows = [dict(id=vid, **job.result()) for vid, job in zip(ids, jobs)]

    res = pd.DataFrame(rows)
    res.to_csv(WORK / args.out, index=False)
    print(f"\nMean over {len(res)} validation volumes:\n" + res.drop(columns="id").mean().round(4).to_string())


if __name__ == "__main__":
    main()
