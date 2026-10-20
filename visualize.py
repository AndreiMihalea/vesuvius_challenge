"""Show one volume: middle slice along z, y and x. Rows: CT image, label, and (with --ckpt) the model prediction.
Saves work/vis/<id>.png.  Usage: uv run visualize.py 1004283650 [--ckpt best.pt]"""
import argparse

import matplotlib
matplotlib.use("Agg")  # no display needed on the remote machine
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import ListedColormap

from common import DATA, WORK

LABEL_COLORS = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.6), (1, 1, 0, 0.3)])  # 0 transparent, 1 red, 2 yellow
ERROR_COLORS = ListedColormap([(0, 0, 0, 0), (0, 1, 0, 0.5), (1, 0, 0, 0.8), (0, 0.4, 1, 0.8)])  # 0 transparent, hit, missed, false


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--ckpt", help="also show the prediction of this checkpoint, e.g. best.pt")
    args = ap.parse_args()

    img = tifffile.imread(DATA / "train_images" / f"{args.id}.tif")  # (z, y, x) uint8
    lab = tifffile.imread(DATA / "train_labels" / f"{args.id}.tif")  # 0 background, 1 surface, 2 ignore
    rows = {"image": (None, None), "label (red=surface, yellow=ignore)": (lab, LABEL_COLORS)}
    if args.ckpt:
        import torch
        from common import UNet3D, predict
        from evaluate import postprocess
        model = UNet3D().cuda().eval()
        model.load_state_dict(torch.load(WORK / args.ckpt))
        pred = postprocess(predict(model, img), 0.5, 1000).astype(bool)
        rows["prediction"] = (pred.astype(np.uint8), LABEL_COLORS)

        # errors, ignore voxels (label 2) are not counted
        truth, valid = lab == 1, lab != 2
        errors = np.zeros(lab.shape, np.uint8)
        errors[pred & truth] = 1  # hit
        errors[~pred & truth] = 2  # missed surface
        errors[pred & ~truth & valid] = 3  # false surface
        rows["errors (green=hit, red=missed, blue=false)"] = (errors, ERROR_COLORS)
        tp, fn, fp = (errors == k).sum() for k in (1, 2, 3)
        print(f"precision {tp / max(tp + fp, 1):.3f}  recall {tp / max(tp + fn, 1):.3f}  dice {2 * tp / max(2 * tp + fp + fn, 1):.3f}")

    print(f"image {img.shape} {img.dtype}, label values {np.unique(lab).tolist()}")
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)), squeeze=False)
    for r, (name, (overlay, cmap)) in enumerate(rows.items()):
        for a in range(3):
            i = img.shape[a] // 2
            ax = axes[r, a]
            ax.imshow(np.take(img, i, axis=a), cmap="gray")
            if overlay is not None:
                ax.imshow(np.take(overlay, i, axis=a), cmap=cmap, vmin=0, vmax=cmap.N - 1, interpolation="nearest")
            ax.set_title(f"{name} | {'zyx'[a]}={i}", fontsize=9)
            ax.axis("off")

    out = WORK / "vis" / f"{args.id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print("saved", out)


if __name__ == "__main__":
    main()
