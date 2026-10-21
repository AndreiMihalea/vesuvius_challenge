# Vesuvius Challenge: surface detection baseline

This is a plain 3D U-Net trained from scratch in PyTorch. It gives a simple, readable reference point before moving to nnU-Net ensembles (see `deep_research.md`).

```
data/                 # competition data (train.csv, train_images/, train_labels/, ...)
common.py             # paths, 3D U-Net, sliding-window inference
prepare.py            # step 1: data processing
train.py              # step 2: training
evaluate.py           # step 3: evaluation on the validation fold
nnunet_trainers.py    # nnU-Net fine-tune trainers (control + crossing loss)
REPORT.md             # what was done, results, difference from leaderboard solutions
work/                 # created by the scripts: .npy cache, split, checkpoints, metrics
```

## Run

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once, if uv is missing
bash setup_metric.sh     # env + official metric; needs work/metric/vesuvius-metric-resources.zip (see section 3)
uv run prepare.py
uv run train.py          # defaults: 200 epochs x 250 iterations, batch 4
uv run evaluate.py       # writes work/val_metrics.csv
```

Training takes hours, so run it inside `tmux` or with `nohup`. For a quick smoke test, run `uv run train.py --epochs 1 --iters 10`.

## 1. Data processing (`prepare.py`)

- **Input:** 786 CT chunks (`uint8`, 320×320 in-plane, z×y×x) from 6 scrolls. Labels: `0` background, `1` papyrus surface, `2` ignore. The 20 deprecated volumes listed in `train.csv` are skipped.
- **Decode once:** the TIFFs are LZW-compressed, so each one is decoded once to `.npy`. Training then reads random crops through a memory map without decompressing the whole volume.
- **Split:** the same split the competition's nnU-Net notebooks validate on, nnU-Net's default 5-fold CV. Case ids are sorted as strings and split with `KFold(5, shuffle=True, random_state=12345)`. Fold 0 (158 volumes) is the validation set and the other 628 volumes are the training set. `prepare.py --fold 1..4` selects another fold. `prepare.py --fold all` trains on all 786 volumes with no validation, which is how the final Kaggle models were trained. The folds are random, not grouped by scroll, so neighbouring chunks of one scroll can land on both sides and the validation score is somewhat optimistic.
- **Per patch** (at training time): z-score normalization, then random flips on each axis and a random axis permutation. The voxels are isotropic, so every axis order is a valid sample.

## 2. Training (`train.py`)

- **Model:** a 5-level 3D U-Net (32→320 channels, InstanceNorm and LeakyReLU, as in nnU-Net) with one sigmoid output for "surface".
- **Samples:** random 128³ crops from random training volumes. A crop is re-drawn when more than half of it is `ignore`.
- **Loss:** BCE + soft Dice. Both are computed only on voxels whose label is not `2`.
- **Optional sheet-crossing loss (ours, `--topo <weight>`):** along every axis-aligned line of voxels, the prediction should enter a sheet as many times as the label does. The soft count is the sum of the positive steps of the probability along the line, and for a binary line it equals the number of foreground runs. A hole in a sheet or a bridge between two wraps removes a crossing, and a stray blob adds one. A sheet that is slightly too thick keeps the count on lines that cross it, so the term focuses on structural errors, which cost TopoScore and VOI, rather than boundary errors, which SurfaceDice@2 tolerates. The loss is the summed absolute count error divided by the label's total count. Ignore voxels count as background in both prediction and label. Use `--tag _topo` to save `best_topo.pt` instead of overwriting `best.pt`.
- **Optimization:** AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, bf16 autocast, gradient clipping at 12.
- **Checkpoints:** `work/last.pt` after every epoch. Every 5 epochs, Dice on 8 validation volumes picks `work/best.pt`.

## 3. Evaluation (`evaluate.py`)

For every volume of the validation fold:

1. **Inference:** a 128³ window slides over the full volume with 50% overlap, and the overlapping probabilities are averaged.
2. **Post-processing:** threshold at `--thr` (default 0.26), then drop 26-connected components smaller than `--min-size` (default 20000) voxels. With `--hyst-high H` (hysteresis), a component of `prob > thr` survives only if it contains a voxel with `prob > H`.
3. **Metric:** the official leaderboard score, computed by the competition's own library (`topometrics`):

   **score = 0.30 · TopoScore + 0.35 · SurfaceDice@2 + 0.35 · VOI_score**

   - `topo_score`: Betti matching between prediction and label. It compares components (k=0), tunnels (k=1) and cavities (k=2), computes an F1 per dimension, and averages them with weights 0.34 / 0.33 / 0.33. It punishes bridges between wraps, breaks within a wrap, and spurious holes.
   - `surface_dice`: share of surface voxels that lie within 2 voxels of the other mask's surface.
   - `voi_score`: `1 / (1 + 0.3 · VOI)`, where VOI is the variation of information between 26-connected components. It drops when sheets merge or split.
   - Voxels labelled `2` count as background in both prediction and label. All terms lie in [0, 1], and higher is better.

The script prints per-volume and mean scores and writes them to `work/val_metrics.csv`. The leaderboard averages the score over volumes in the same way.

**Installing the metric.** `topometrics` is not on PyPI or GitHub. Kaggle staff publish it as the dataset [`sohier/vesuvius-metric-resources`](https://www.kaggle.com/datasets/sohier/vesuvius-metric-resources), which contains the Python source and a C++ Betti-matching extension. Download the zip and place it at `work/metric/vesuvius-metric-resources.zip`. `setup_metric.sh` then unzips it, runs `uv sync` (`pyproject.toml` points `topometrics-3d` at the unzipped folder), and compiles the extension with CMake and pybind11.

The metric takes about 45 s per 320³ volume on CPU, so `evaluate.py` scores volumes in parallel processes (`--workers`, default 12) while the GPU predicts the next volume.

`best.pt` is chosen with 8 volumes of the validation fold, so its validation score is slightly optimistic. For an unbiased number, use `--ckpt last.pt`.

## nnU-Net (stronger model)

`nnunet_prepare.py` builds nnU-Net's raw dataset from symlinks, declares label 2 as nnU-Net's ignore label, and writes the same 5 folds as `prepare.py` to `splits_final.json`. Fold 0 is therefore identical for both models, and `evaluate.py --probs-dir` scores nnU-Net with the same metric:

```bash
export nnUNet_raw=$PWD/work/nnunet/raw nnUNet_preprocessed=$PWD/work/nnunet/preprocessed nnUNet_results=$PWD/work/nnunet/results
uv run nnunet_prepare.py
uv run nnUNetv2_plan_and_preprocess -d 100 -pl nnUNetPlannerResEncM -c 3d_fullres -np 8 --verify_dataset_integrity
uv run nnUNetv2_train 100 3d_fullres 0 -p nnUNetResEncUNetMPlans --npz    # 1000 epochs, saves val probabilities
uv run evaluate.py --probs-dir work/nnunet/results/Dataset100_VesuviusSurface/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation
```

## Final model and Kaggle submission

The best fold-0 recipe (score 0.6045) is retrained on all 786 volumes (fold `all`, no validation), which is how the leaderboard notebooks trained their final models. `run_final.sh` runs three stages: `nnUNetTrainerBase` 1000 epochs, then `nnUNetTrainerCrossing` 250 epochs with the crossing loss, then `nnUNetTrainerCrossingLong` 1000 more epochs. It takes about 20 h on one A100.

The competition is a code competition: a Kaggle notebook without internet predicts the hidden test set and writes `submission.zip`, one uint8 `<id>.tif` mask per test volume.

```bash
uv run export_submission.py --user <kaggle username>   # work/kaggle/dataset (model + offline wheels), work/kaggle/kernel
uvx kaggle datasets create -p work/kaggle/dataset -r zip  # private dataset, about 600 MB
uvx kaggle kernels push -p work/kaggle/kernel           # private GPU script, internet off
uvx kaggle competitions submit vesuvius-challenge-surface-detection -k <user>/vesuvius-crossing-nnunet-submit -v 1 -f submission.zip -m "crossing nnU-Net, fold all"
```

- `export_submission.py` keeps only the network weights and renames the trainer to nnU-Net's default `nnUNetTrainer`. Our trainers change only the loss, so Kaggle needs no custom code. It also downloads wheels for nnU-Net's whole dependency tree except torch and CUDA (Kaggle has its own), and Kaggle installs only the ones it lacks.
- Authentication: save the token from Kaggle's settings page to `~/.kaggle/access_token`.
- `kaggle_inference.py` installs those wheels offline, then predicts every test volume exactly like the fold-0 validation: nnU-Net sliding window with mirroring, threshold 0.26, components under 20k voxels removed. It runs one process per GPU (Kaggle's 2× T4) and takes about 82 s per volume on one T4 (18 s on an A100). Locally: `uv run kaggle_inference.py --input <folder with test_images/> --model work/kaggle/dataset/model --out work/submission.zip`.
