"""Kaggle submission script (runs offline on Kaggle's GPU): final nnU-Net model -> masks -> submission.zip.

Same steps as the fold-0 validation (0.6045): nnU-Net sliding window with mirroring TTA -> prob > 0.26
-> drop components < 20k voxels -> one uint8 <id>.tif per test volume, zipped.
Locally: uv run kaggle_inference.py --input data --model <model folder> --out work/submission.zip"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

THR, MIN_SIZE = 0.26, 20000


def find(pattern):
    """First match under /kaggle/input (the mount path of datasets differs between Kaggle versions)."""
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        sys.exit(f"not found under /kaggle/input: {pattern}")
    return Path(hits[0])


def install_offline(wheels):
    """Kaggle has no internet: install nnU-Net and its missing helpers from the wheels in our dataset."""
    for name in sorted({w.name.split("-")[0] for w in wheels.iterdir()}):  # one by one: a wheel built for
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",  # another Python
                        "--no-build-isolation", "--find-links", str(wheels), name],  # fails alone, the rest
                       check=False)                                                    # still installs


def predict_shard(files, model, out_dir):
    """Predict + post-process `files` on one GPU, writing <id>.tif masks to out_dir."""
    import numpy as np
    import tifffile
    import torch
    from scipy import ndimage as ndi
    from nnunetv2.imageio.tif_reader_writer import Tiff3DIO
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=torch.device("cuda"), allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model), use_folds=("all",), checkpoint_name="checkpoint_final.pth")
    for i, f in enumerate(files):
        t = time.time()
        img, props = Tiff3DIO().read_images([str(f)])  # the reader used in training: same axis order
        _, prob = predictor.predict_single_npy_array(img, props, None, None, True)
        cc, _ = ndi.label(prob[1] > THR, np.ones((3, 3, 3), bool))
        keep = np.bincount(cc.ravel()) >= MIN_SIZE
        keep[0] = False
        mask = keep[cc].astype(np.uint8)
        tifffile.imwrite(out_dir / f.name, mask)
        print(f"{i + 1}/{len(files)} {f.stem}: {mask.mean():.3f} foreground, {time.time() - t:.0f} s", flush=True)


def main():
    on_kaggle = Path("/kaggle/input").exists()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="competition folder with test_images/ (default: found under /kaggle/input)")
    ap.add_argument("--model", help="model folder with plans.json and fold_all/ (default: found under /kaggle/input)")
    ap.add_argument("--out", default="/kaggle/working/submission.zip" if on_kaggle else "work/submission.zip")
    ap.add_argument("--shard", type=int, nargs=2, metavar=("K", "N"), help="internal: predict every N-th file from K")
    args = ap.parse_args()

    model = Path(args.model) if args.model else find("plans.json").parent
    test_dir = Path(args.input) / "test_images" if args.input else find("test_images/*.tif").parent
    files = sorted(test_dir.glob("*.tif"))
    masks = Path(args.out).parent / "masks"
    if args.shard:
        k, n = args.shard
        return predict_shard(files[k::n], model, masks)

    if on_kaggle:
        install_offline(model.parent / "wheels")
    masks.mkdir(parents=True, exist_ok=True)
    import torch
    n = max(torch.cuda.device_count(), 1)  # Kaggle's 2x T4: one process per GPU halves the time
    print(f"{len(files)} test volumes on {n} GPU(s), model {model}", flush=True)
    me = [sys.executable, __file__, "--model", str(model), "--out", args.out]
    if args.input:
        me += ["--input", args.input]
    procs = [subprocess.Popen(me + ["--shard", str(k), str(n)], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)})
             for k in range(n)]
    if any(p.wait() for p in procs):
        sys.exit("a GPU process failed")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(masks / f.name, f.name)
    shutil.rmtree(masks)  # Kaggle keeps everything in /kaggle/working as output: leave only the zip
    print("wrote", args.out)


if __name__ == "__main__":
    main()
