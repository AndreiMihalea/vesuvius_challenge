"""Step 5: build the Kaggle submission. nnU-Net probabilities -> threshold -> drop small blobs -> uint8 <id>.tif -> submission.zip.
Local check:  uv run predict.py --test-dir data/train_images --limit 3 --out work/sub_check.zip
On Kaggle:    python predict.py   (defaults below; weights and wheels come from attached datasets)"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import tifffile
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scipy import ndimage as ndi

CONN26 = np.ones((3, 3, 3), bool)


def postprocess(prob, thr, min_size):
    """Same as evaluate.py: threshold, then drop 26-connected components under min_size voxels."""
    cc, _ = ndi.label(prob > thr, CONN26)
    keep = np.bincount(cc.ravel()) >= min_size
    keep[0] = False
    return keep[cc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="nnU-Net results folder (holds plans.json, dataset.json, fold_*/)")
    ap.add_argument("--folds", default="0", help="comma-separated folds; 'all' for a model trained with --fold all")
    ap.add_argument("--ckpt", default="checkpoint_final.pth")
    ap.add_argument("--test-dir", default="/kaggle/input/vesuvius-challenge-surface-detection/test_images")
    ap.add_argument("--out", default="/kaggle/working/submission.zip")
    ap.add_argument("--thr", type=float, default=0.26)
    ap.add_argument("--min-size", type=int, default=20000)
    ap.add_argument("--limit", type=int, help="only the first N volumes (local check)")
    args = ap.parse_args()

    folds = [f if f == "all" else int(f) for f in args.folds.split(",")]
    predictor = nnUNetPredictor(device=torch.device("cuda"), perform_everything_on_device=True, verbose=False)
    predictor.initialize_from_trained_model_folder(args.model_dir, folds, checkpoint_name=args.ckpt)

    tifs = sorted(Path(args.test_dir).glob("*.tif"))[:args.limit]
    assert tifs, f"no .tif in {args.test_dir}"
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for i, tif in enumerate(tifs, 1):
            img = tifffile.imread(tif).astype(np.float32)[None]  # (1, z, y, x), as Tiff3DIO reads it
            _, probs = predictor.predict_single_npy_array(img, {"spacing": [1.0, 1.0, 1.0]}, None, None, True)
            mask = postprocess(probs[1], args.thr, args.min_size).astype(np.uint8)
            tmp = Path(args.out).with_suffix(".tmp.tif")
            tifffile.imwrite(tmp, mask)
            z.write(tmp, f"{tif.stem}.tif")
            tmp.unlink()
            print(f"{i}/{len(tifs)} {tif.stem}: {mask.mean():.3f} foreground", flush=True)


if __name__ == "__main__":
    main()
