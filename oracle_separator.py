"""Oracle test for the separator idea: carve the TRUE gaps between wraps (from the labels) out of the prediction.
If this upper bound barely beats the plain prediction, a learned separator class is not worth training.

Separator = background voxels on the border between the regions nearest to two different label sheets
(both sides, so about 2 voxels thick: a 1-voxel cut leaks under 26-connectivity and only punches tunnels),
at most --max-gap voxels from a sheet.

Labels miss many real sheets, so carving everywhere also chops unlabeled predicted sheets into fragments.
Variant "merged" carves only inside predicted components that touch 2+ label sheets (true merges)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from common import IGNORE, WORK, load
from evaluate import CONN26, official_score, postprocess


def separator(lab, max_gap, inst=None):
    sheets = lab == 1
    if inst is None:
        inst, _ = ndi.label(sheets, CONN26)
    dist, idx = ndi.distance_transform_edt(~sheets, return_indices=True)
    owner = inst[tuple(idx)]  # which sheet each voxel is closest to
    border = (ndi.maximum_filter(owner, size=3) != owner) | (ndi.minimum_filter(owner, size=3) != owner)
    return border & ~sheets & (lab != IGNORE) & (dist <= max_gap)


def merged_components(prob, thr, inst):
    """Mask of the components of prob > thr that overlap 2+ label sheets."""
    cc, n = ndi.label(prob > thr, CONN26)
    hit = (cc > 0) & (inst > 0)
    pairs = np.unique(np.stack([cc[hit], inst[hit]]), axis=1)
    merged = np.bincount(pairs[0], minlength=n + 1) > 1
    merged[0] = False
    return merged[cc]


def score_volume(prob, lab, thr, min_size, gaps):
    prob = prob.astype(np.float32)
    inst, _ = ndi.label(lab == 1, CONN26)
    merged = merged_components(prob, thr, inst)
    rows = {"none": official_score(postprocess(prob, thr, min_size), lab)}
    for g in gaps:
        sep = separator(lab, g, inst)
        for name, cut in ((f"gap{g:g}", sep), (f"merged{g:g}", sep & merged)):
            # carve, size filter, then refill enclosed separator patches: a partial cut only adds cavities
            rows[name] = official_score(ndi.binary_fill_holes(postprocess(np.where(cut, 0, prob), thr, min_size)), lab)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-dir", required=True)
    ap.add_argument("--gaps", type=float, nargs="+", default=[6])
    ap.add_argument("--thr", type=float, default=0.26)
    ap.add_argument("--min-size", type=int, default=20000)
    ap.add_argument("--n", type=int)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default="oracle_separator.csv")
    args = ap.parse_args()

    split = pd.read_csv(WORK / "split.csv")
    ids = split[split.split == "val"].id.tolist()[:args.n]
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(score_volume, np.load(Path(args.probs_dir) / f"{vid}.npz")["probabilities"][1],
                            np.asarray(load(vid, "labels")), args.thr, args.min_size, args.gaps) for vid in ids]
        rows = [dict(id=vid, variant=v, **m) for vid, j in zip(ids, jobs) for v, m in j.result().items()]

    res = pd.DataFrame(rows)
    res.to_csv(WORK / args.out, index=False)
    print(res.drop(columns="id").groupby("variant", sort=False).mean().round(4).to_string())
    s = res.pivot(index="id", columns="variant", values="score")
    for v in s.columns.drop("none"):
        d = s[v] - s["none"]
        print(f"{v} - none: {d.mean():+.4f} +- {d.sem():.4f}, wins {(d > 0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
