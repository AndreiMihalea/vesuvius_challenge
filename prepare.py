"""Step 1: decode the LZW TIFFs once to .npy (fast random crops via memmap) and make the train/val split.

Split = nnU-Net's default 5-fold CV (what the competition's nnU-Net notebooks validate on):
case ids sorted as strings -> KFold(5, shuffle=True, random_state=12345) -> fold `--fold` is validation.
`--fold all` trains on all 786 volumes with no validation (how the final Kaggle models were trained).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tifffile
from sklearn.model_selection import KFold

from common import DATA, NPY, WORK


def convert(vid):
    for kind in ("images", "labels"):
        dst = NPY / kind / f"{vid}.npy"
        if not dst.exists():
            np.save(dst, tifffile.imread(DATA / f"train_{kind}" / f"{vid}.tif"))


def nnunet_val_ids(ids, fold):
    """Copy of nnunetv2.utilities.crossval_split.generate_crossval_split with nnU-Net's defaults."""
    keys = np.sort([str(i) for i in ids])
    val_idx = list(KFold(n_splits=5, shuffle=True, random_state=12345).split(keys))[fold][1]
    return set(keys[val_idx])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0", help="0-4, or 'all'")
    args = ap.parse_args()

    for kind in ("images", "labels"):
        (NPY / kind).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA / "train.csv")
    df = df[[(DATA / "train_images" / f"{i}.tif").exists() for i in df.id]]  # drop the deprecated volumes
    val = set() if args.fold == "all" else nnunet_val_ids(df.id, int(args.fold))
    df["split"] = np.where(df.id.astype(str).isin(val), "val", "train")

    with ProcessPoolExecutor(16) as ex:
        list(ex.map(convert, df.id))
    df.to_csv(WORK / "split.csv", index=False)
    print(df.groupby(["split", "scroll_id"]).size())
