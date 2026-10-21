# Report: what was done and how it differs from the leaderboard solutions

Status on 2026-10-19. All scores use the official leaderboard metric (0.30·TopoScore + 0.35·SurfaceDice@2 + 0.35·VOI) on fold 0 of nnU-Net's default 5-fold split: 628 training volumes and 158 validation volumes. These local scores are not directly comparable with the Kaggle leaderboard, where 1st place scored 0.627 on the private set.

## 1. Pipeline

| Step | What | File |
|---|---|---|
| Data | Decode the 786 LZW TIFFs once to `.npy` (memory-mapped). Labels: 0 background, 1 surface, 2 ignore | `prepare.py` |
| Split | nnU-Net's default KFold(5, seed 12345); fold 0 for validation, the same split for both models | `prepare.py`, `nnunet_prepare.py` |
| Model A | Our 5-level 3D U-Net (nnU-Net-style blocks): 128³ patches, BCE + soft Dice, AdamW, 200 epochs × 250 iterations, 51 s/epoch | `common.py`, `train.py` |
| Model B | nnU-Net v2 ResEnc M, 3d_fullres, 1000 epochs, label 2 declared as ignore | `nnunet_prepare.py` |
| Evaluation | Sliding window (50% overlap) → threshold 0.26 → drop components < 20k voxels → official `topometrics` metric (parallel on CPU) | `evaluate.py` |

## 2. Results (158 validation volumes unless noted)

| Model / post-processing | score | Topo | SurfDice | VOI |
|---|---|---|---|---|
| U-Net, thr 0.5, min 1k | 0.406 | 0.076 | 0.578 | 0.518 |
| U-Net, thr 0.26, min 20k | 0.463 | 0.060 | 0.711 | 0.560 |
| U-Net + seeded watershed (n = 40, best seed) | 0.443 | 0.105 | | 0.486 |
| U-Net + crossing loss, soft | 0.466 | 0.016 | 0.745 | 0.573 |
| **U-Net + crossing loss, sharpened (ours)** | **0.492** | **0.117** | 0.736 | 0.571 |
| nnU-Net ResEnc M, thr 0.26, min 20k | **0.594** | 0.298 | 0.873 | 0.568 |
| nnU-Net + hysteresis (high 0.5 / 0.7 / 0.9, n = 40) | 0.576 / 0.576 / 0.571 (base 0.576) | | | |
| nnU-Net + 250 epochs fine-tune (control) | 0.596 | 0.303 | 0.875 | 0.568 |
| nnU-Net + 250 epochs fine-tune with crossing loss | 0.600 | 0.320 | 0.874 | 0.567 |
| **same + 1000 more epochs (2250 in total)** | **0.6045** | **0.329** | 0.877 | 0.568 |

What was rejected:
- **Seeded watershed** splits merged sheets, but it over-splits, so VOI drops.
- **Topology-guided hole filling** (a lower threshold near sheets) doubles the voxel count, so the score drops.
- **Hysteresis** changes nothing: every component above 20k voxels already contains confident voxels.
- **Separator carving** (oracle test, `oracle_separator.py`): cut the *true* gaps between label sheets out of the nnU-Net prediction. Even this upper bound lowers the score: 0.600 → 0.595 for gaps ≤ 2 voxels (−0.005 ± 0.001) and → 0.588 for gaps ≤ 6 (−0.013 ± 0.002), on all 158 volumes. Carving only inside components that touch 2+ label sheets is also worse (−0.004 / −0.008). The labels miss many real sheets, so wide cuts chop unlabeled predicted sheets into fragments, and narrow cuts find almost no bridges. After masking the ignore region, nnU-Net rarely merges wraps, so a learned separator class is not worth training.

The main error in every model is tunnels: k1 F1 is about 0.01 even for nnU-Net. For our U-Net they come from holes in sheets and from bridges between adjacent wraps. For nnU-Net the separator oracle suggests they are mostly holes inside single sheets.

## 3. New: sheet-crossing loss

**Idea.** Take every axis-aligned line of voxels. The prediction must enter a sheet as many times as the label does. A hole, or a bridge that fuses two wraps, removes a crossing, and a stray blob adds one. A sheet that is slightly too thick or too thin keeps the count, so the loss punishes structural errors (the ones TopoScore and VOI punish) and mostly ignores boundary errors (which SurfaceDice@2 tolerates).

**Formula.** The soft crossing count of a line is the sum of the positive steps of the probability along it. For a binary line, this is exactly the number of foreground runs, that is, the line's Betti-0. The loss is the summed absolute count error over all lines on all 3 axes, divided by the label's total count. Ignore voxels count as background in both prediction and label. It costs about 11 ms per step.

**Sharpening (the part that matters).** The sum of positive steps equals the run count averaged over *all* thresholds. A faint bridge (p ≈ 0.3) therefore costs little in the loss, yet after thresholding at 0.26 it becomes a full bridge. The sharpened version counts on `sigmoid((logit − logit(0.26)) / 0.2)`, which is the mask the evaluation actually produces.

| Variant | score | paired gain over baseline | wins out of 158 |
|---|---|---|---|
| soft | 0.466 | +0.003 ± 0.002 | 79 (noise) |
| sharpened | **0.492** | **+0.029 ± 0.002** | **146** |

TopoScore doubles (0.060 → 0.117), mainly through fewer merged wraps (k0 F1 0.03 → 0.20). Tunnels (k1) are still not fixed.

**On nnU-Net.** Both runs fine-tune the 0.594 model for 250 epochs at lr 1e-3. The control has the same setup without the crossing loss, so the effect of the extra epochs alone is measured.

| Run | score | paired gain | wins out of 158 |
|---|---|---|---|
| Finetune (control) − baseline | 0.596 | +0.002 ± 0.002 | 91 (noise) |
| Crossing − baseline | 0.600 | +0.007 ± 0.002 | 100 |
| Crossing − Finetune | | **+0.004 ± 0.002** | 84 |

The gain is small (about 2 SEM over the control) and comes entirely from TopoScore, mostly cavities (k2 F1 0.14 → 0.22). nnU-Net's output is already much cleaner than our U-Net's, so there is less to fix.

**Novelty.** A web search (2026-10-18) found persistent-homology, clDice, Skeleton Recall and soft Euler-characteristic losses, but nothing that counts crossings along voxel lines or counts them at the operating threshold. This does not prove nobody has done it.

Usage: `train.py --topo 1 --topo-temp 0.2 --tag _topo_sharp`. For nnU-Net: `nnunet_trainers.py`, trainer `nnUNetTrainerCrossing`.

## 4. Difference from the leaderboard solutions

| Aspect | Top solutions (1st–12th) | Ours |
|---|---|---|
| Model | nnU-Net ResEnc / plain, ensembles of 3–4 patch sizes (128–256) | One nnU-Net ResEnc M (128³) and one small U-Net |
| Training length | 2000–8000 epochs (12th place: 1000 → 4000 epochs gave +0.06) | 1000 epochs (nnU-Net), 200 (U-Net) |
| Topology loss | Skeleton Recall, Medial Surface Recall, clDice, or none | **New sheet-crossing loss, counted at the evaluation threshold** |
| Post-processing | Small-component removal + hole plugging, height-map patching, closing (1st); Frangi / hysteresis / median filters (others) | Only small-component removal; watershed, hole filling and hysteresis tested and rejected |
| Labels | 7th place reprocessed the train labels like the corrected test labels | Raw train labels |
| TTA | Flips or rotations | None |

The only part that is new compared with published solutions is the crossing loss. Everything else is a smaller version of the known recipe.

## 5. Scripts and options

All scripts run on the remote GPU server with `uv run <script>` from the project folder. Outputs go to `work/`.

**`setup_metric.sh`** (no options). Installs the official metric. Needs `work/metric/vesuvius-metric-resources.zip` from the Kaggle dataset `sohier/vesuvius-metric-resources`.

**`prepare.py`**: decodes the TIFFs to `work/npy/` and writes `work/split.csv`.

| Option | Default | Meaning |
|---|---|---|
| `--fold` | `0` | Validation fold `0`–`4`, or `all` (train on all 786 volumes, no validation) |

**`train.py`**: trains our U-Net. Saves `work/best<tag>.pt` (best validation Dice) and `work/last<tag>.pt`.

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 200 | Number of epochs |
| `--iters` | 250 | Iterations per epoch |
| `--batch` | 4 | Patches (128³) per batch |
| `--lr` | 1e-3 | AdamW learning rate |
| `--workers` | 16 | Data-loader processes |
| `--topo` | 0 | Weight of the sheet-crossing loss (0 = off) |
| `--topo-temp` | off | Sharpening temperature in logit units (0.2 = our best). Off = soft version |
| `--topo-thr` | 0.26 | Threshold the sharpened count is centred on (the evaluation threshold) |
| `--tag` | empty | Checkpoint suffix, e.g. `_topo_sharp` |

**`evaluate.py`**: post-processes and scores the validation fold with the official metric. Writes a per-volume CSV and prints the means.

| Option | Default | Meaning |
|---|---|---|
| `--ckpt` | `best.pt` | Our U-Net checkpoint in `work/`. Its probabilities are cached in `work/probs/` |
| `--probs-dir` | off | Score nnU-Net's saved probabilities (`<id>.npz`) instead of our U-Net |
| `--thr` | 0.26 | Probability threshold |
| `--min-size` | 20000 | Drop predicted components smaller than this (voxels) |
| `--seed-thr` | off | Split merged sheets by seeded watershed from `prob > seed-thr` |
| `--hyst-high` | off | Hysteresis: keep only components that contain a voxel with `prob > hyst-high` |
| `--n` | all | Score only the first N validation volumes (quick sweeps) |
| `--workers` | 12 | CPU processes for post-processing and the metric |
| `--out` | `val_metrics.csv` | Output CSV in `work/` |

**`visualize.py`**: saves the middle slices along z, y and x to `work/vis/<id>.png`.

| Option | Default | Meaning |
|---|---|---|
| `id` (positional) | required | Volume id |
| `--ckpt` | off | Also show this checkpoint's prediction |

**`nnunet_prepare.py`** (no options). Builds the nnU-Net raw dataset in `work/nnunet/raw` (symlinks, label 2 = ignore) and `splits_final.json` with the same folds. Run after `prepare.py`.

**`nnunet_trainers.py`** (no command-line options). Defines `nnUNetTrainerFinetune` and `nnUNetTrainerCrossing`, which start from the base model's `checkpoint_final.pth` of the same fold and train 250 epochs at lr 1e-3, and `nnUNetTrainerCrossingLong`, which continues the Crossing model for 1000 more epochs. `nnUNetTrainerBase` is nnU-Net's default trainer, except that it skips the final validation on fold `all`. The crossing loss uses weight 1.0, temperature 0.2 and threshold 0.26; change these in `CrossingLoss`. Link the file into nnU-Net once, then train:
```bash
ln -s $PWD/nnunet_trainers.py .venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/
export nnUNet_raw=$PWD/work/nnunet/raw nnUNet_preprocessed=$PWD/work/nnunet/preprocessed nnUNet_results=$PWD/work/nnunet/results
uv run nnUNetv2_train 100 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerCrossing --npz
```

**`run_final.sh`** (no options). Trains the final model on fold `all` (all 786 volumes, no validation) with the best fold-0 recipe: `nnUNetTrainerBase` 1000 epochs, then `nnUNetTrainerCrossing` 250, then `nnUNetTrainerCrossingLong` 1000. About 19 h. Run it in tmux; re-running skips finished stages.

**`export_submission.py`**: packages the final model for Kaggle into `work/kaggle/dataset` (weights only, trainer renamed to `nnUNetTrainer`, plus offline wheels for nnU-Net's whole dependency tree except torch/CUDA) and `work/kaggle/kernel` (script + metadata).

| Option | Default | Meaning |
|---|---|---|
| `--user` | required | Kaggle username that owns the dataset and the kernel |
| `--slug` | `vesuvius-crossing-nnunet` | Name of the dataset; the kernel is `<slug>-submit` |

**`kaggle_inference.py`**: the Kaggle submission script. It predicts the test volumes (mirroring TTA, threshold 0.26, components < 20k removed) with one process per GPU and writes `submission.zip`. On Kaggle it finds its inputs under `/kaggle/input` by itself.

| Option | Default | Meaning |
|---|---|---|
| `--input` | found under `/kaggle/input` | Folder that contains `test_images/` |
| `--model` | found under `/kaggle/input` | Model folder with `plans.json` and `fold_all/` |
| `--out` | `/kaggle/working/submission.zip` (Kaggle), else `work/submission.zip` | Output zip |
| `--shard K N` | off | Internal: the per-GPU worker predicts every N-th file starting at K |

**`oracle_separator.py`**: carves the true gaps between label sheets out of nnU-Net's prediction and scores each variant (`none`, `gap<g>`, `merged<g>`).

| Option | Default | Meaning |
|---|---|---|
| `--probs-dir` | required | Folder with nnU-Net's `<id>.npz` probabilities |
| `--gaps` | 6 | One or more maximum distances (voxels) from a sheet where the separator is carved |
| `--thr` | 0.26 | Probability threshold |
| `--min-size` | 20000 | Drop predicted components smaller than this (voxels) |
| `--n` | all | Only the first N validation volumes |
| `--workers` | 20 | CPU processes |
| `--out` | `oracle_separator.csv` | Output CSV in `work/` |

## 6. Next steps

1. Longer training done: `nnUNetTrainerCrossingLong` continued the crossing-loss model for 1000 more epochs (2250 in total, 31 s/epoch, about 9 h). Score 0.6045: +0.004 ± 0.002 over the 250-epoch model (99/158 wins), +0.011 ± 0.002 over the baseline. Tunnels (k1 F1 0.013) are unchanged.
2. Final model trained on fold `all` with `run_final.sh` (2026-10-19 20:07 – 2026-10-20 16:09 UTC, 20 h). Checkpoint: `work/nnunet/results/Dataset100_VesuviusSurface/nnUNetTrainerCrossingLong__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_final.pth`.
3. Submission: `export_submission.py` + `kaggle_inference.py` (see README). Tested on the remote (3 training volumes score 0.59–0.77, masks are uint8 0/1 with the input shape) and on Kaggle (82 s per volume on a T4). Submitted on 2026-10-20 as a late submission (no prize): **public 0.576, private 0.599**, close to the local estimate of 0.6045. For comparison, 1st place scored 0.606 / 0.627, and its 4-model ensemble before post-processing scored 0.572 / 0.596.
4. Later: ensemble patch sizes.
