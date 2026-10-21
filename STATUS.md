# Status: Vesuvius Challenge surface detection baseline

Last updated: 2026-10-18 (afternoon). Read this first when resuming.

## Goal
A concise, easy-to-understand baseline experiment for the Kaggle "Vesuvius Challenge – Surface Detection" competition. It is scored with the official leaderboard metric on a local validation split. Code and explanations must stay short and readable. The user prefers concise answers.

## Environment
- **Code is written locally** in this folder, but everything **runs on the remote**: `ssh root@65.108.32.168`, project at `/root/vesuvius_challenge`. Never run training locally.
- Remote: 1× A100 80 GB, 22 CPUs, 117 GB RAM, Ubuntu. uv is in `~/.local/bin` (`source $HOME/.local/bin/env`).
- The VS Code sftp extension uploads only on editor save. After editing code files with tools, copy them with `scp <files> root@65.108.32.168:/root/vesuvius_challenge/`. Never copy `.md` files to the remote.
- Data: `data/` = PROJECT_PWD/data on the remote. It holds 786 train volumes (320×320×~320 uint8, LZW TIFF), labels 0/1/2 (2 = ignore), and 20 deprecated volumes (skipped). The local `test_images` holds a single placeholder volume, so there is no usable test set.
- Download checkpoints, logs and metrics to the local `work/` (about 3.5 GB; git ignores `work/`). It skips cached probabilities, validation `.npz` files and preprocessed data. Run it from the project folder; a rerun copies only changed files:
  ```bash
  rsync -am --info=stats1 --include='*/' --include='/*.pt' --include='/*.log' --include='/*.csv' --include='/sweep/*.csv' \
    --include='nnunet/results/**/*.pth' --include='nnunet/results/**/training_log*' --include='nnunet/results/**/progress.png' \
    --include='nnunet/results/**/debug.json' --include='nnunet/results/**/plans.json' --include='nnunet/results/**/dataset.json' \
    --include='nnunet/results/**/dataset_fingerprint.json' --exclude='*' \
    root@65.108.32.168:/root/vesuvius_challenge/work/ work/
  ```
- Long jobs: run them in tmux. The user once started training in a plain SSH shell, which dies when the shell closes.

## Files
| File | Purpose |
|---|---|
| `pyproject.toml` | uv env: torch cu128, topometrics (path dependency), nnunetv2, etc. |
| `setup_metric.sh` | Unzips `work/metric/vesuvius-metric-resources.zip` (Kaggle dataset `sohier/vesuvius-metric-resources`), runs `uv sync`, builds the C++ Betti extension. **Already done on the remote.** |
| `common.py` | Paths, our `UNet3D`, sliding-window `predict`, `dice` |
| `prepare.py` | TIFF → `work/npy`, and the split → `work/split.csv`. `--fold 0..4` or `all` (nnU-Net default KFold(5, shuffle, seed 12345) on ids sorted as strings) |
| `train.py` | Our U-Net: 128³ patches, BCE + soft Dice, AdamW, 200 epochs × 250 iterations. `--topo W` adds our sheet-crossing loss; `--tag _x` saves `best_x.pt`/`last_x.pt` |
| `evaluate.py` | Official metric (`topometrics.leaderboard.compute_leaderboard_score`, voi_alpha 0.3). Flags: `--thr` (default 0.26), `--min-size` (20000), `--seed-thr` (watershed, off), `--hyst-high` (hysteresis, off), `--n`, `--out`, `--probs-dir` (nnU-Net npz). Caches our model's probabilities in `work/probs/` |
| `visualize.py` | Middle slices of image + label (+ prediction) → `work/vis/<id>.png` |
| `nnunet_prepare.py` | nnU-Net raw dataset (symlinks, ignore label 2, Tiff3DIO) + `splits_final.json` (same folds) |
| `nnunet_trainers.py` | `nnUNetTrainerFinetune` (control) and `nnUNetTrainerCrossing`: 250-epoch fine-tunes from fold-0 `checkpoint_final.pth`, lr 1e-3. Symlinked into the installed nnunetv2 trainer folder |
| `REPORT.md` | Concise write-up: pipeline, results, crossing loss, difference from leaderboard solutions |
| `README.md` | User-facing explanation of data processing, training, evaluation, and the nnU-Net steps |
| `deep_research.md` | State-of-the-art research notes (top solutions, metric, tips) |

## Decisions
- **Split:** fold 0 of nnU-Net's default 5-fold CV (628 train / 158 val) for all experiments. The final submission model will be trained with `--fold all`, as the LB notebooks did (1st place and Jirka's notebook used `FOLD="all"` with no validation). The old leave-scroll-26010-out split is kept in `work/split_scroll26010.csv`.
- **Metric:** score = 0.30·TopoScore + 0.35·SurfaceDice@2 + 0.35·VOI_score. About 45 s per volume on CPU, run in parallel.

## Results (fold 0, official metric)
| Model / post-processing | n | score | Topo | SurfDice | VOI |
|---|---|---|---|---|---|
| Our U-Net, 200 epochs, thr 0.5, min 1k | 158 | 0.406 | 0.076 | 0.578 | 0.518 |
| Our U-Net, thr 0.26, min 20k | 158 | **0.463** | 0.060 | 0.711 | 0.560 |
| + watershed seed 0.5 / 0.7 / 0.9 | 40 | 0.443 / 0.441 / 0.350 | 0.105 / 0.095 / 0.059 | | 0.486 / 0.503 / 0.484 |
| Our U-Net + crossing loss (`--topo 1`, `best_topo.pt`), thr 0.26, min 20k | 158 | 0.466 | 0.016 | 0.745 | 0.573 |
| Our U-Net + **sharpened** crossing loss (`--topo 1 --topo-temp 0.2`, `best_topo_sharp.pt`), thr 0.26 | 158 | **0.492** | 0.117 | 0.736 | 0.571 |
| same, thr 0.4 / 0.5 / 0.6 | 40 | 0.458 / 0.439 / 0.396 | 0.041 / 0.069 / 0.059 | | |
| nnU-Net ResEnc M, 1000 epochs, thr 0.26, min 20k | 158 | **0.594** | 0.298 | 0.873 | 0.568 |
| nnU-Net, same, first 40 vols (sweep reference) | 40 | 0.576 | 0.262 | 0.858 | 0.562 |
| nnU-Net + hysteresis high 0.5 / 0.7 / 0.9 | 40 | 0.576 / 0.576 / 0.571 | | | |

The nnU-Net run finished on 2026-10-17. Its own validation Dice is 0.583 and the final pseudo Dice about 0.57. Plans: patch 128³, batch 2, about 30 s/epoch.

## Findings
- nnU-Net k1 F1 (tunnels) is still 0.012, k0 0.54, k2 0.12. Tunnels are the main TopoScore loss for both models.
- Hysteresis (`--hyst-high`) changes nothing at 0.5/0.7: every component over 20k voxels already holds confident voxels. At 0.9 it is slightly worse. It may only help together with a lower `--thr` (not tried).
- **Sheet-crossing loss (our new topology loss, `train.py crossing_loss`).** On every axis-aligned voxel line, the soft number of sheet entries (sum of positive probability steps) must match the label. Holes, bridges between wraps and stray blobs change the count; thickness errors mostly do not. No prior work found with this exact idea (web search on 2026-10-18; soft Euler-characteristic losses do exist). Synthetic test: relative to Dice, it is about 4× more sensitive to holes/bridges/blobs and 5× less sensitive to a 1-voxel thicker sheet. Known weakness: lines lying in the plane just outside a sheet still see thickening. Baseline model on val patches: crossing loss about 0.30 vs BCE+Dice 0.71, so weight 1 is used. Cost about 11 ms per step.
- Our U-Net: TopoScore ≈ 0. Per dimension, k1 (holes/tunnels) F1 ≈ 0. Predicted sheets have hundreds of holes (labels have 0). At thr 0.26 almost all sheets merge into one giant component (80–95% of voxels) with hundreds to thousands of tunnels caused by bridges between wraps.
- Tried and rejected:
  - topology-guided hole filling (lower the threshold near sheets): doubled the voxels, score went down;
  - seeded watershed: k0 F1 improved, but VOI dropped from over-splitting, so the score went down.
- Conclusion: the model output is too noisy, and the model needs to improve before post-processing can help much.
- Known gotcha: training labels are "approximate". Mid-competition the hosts reprocessed the test labels (not the train labels) with a pipeline in ScrollPrize/villa (7th place: applying it to the train labels helped a lot). Not investigated yet.

## Crossing-loss result (2026-10-18)
- Weight 1, 200 epochs: 0.466 vs 0.463 baseline. Paired difference +0.003 ± 0.002 (SEM), with 79 wins out of 158 volumes, so it is a tie. SurfDice +0.033 and VOI +0.012, but TopoScore fell from 0.060 to 0.016. k1 F1 stays at about 0, so tunnels are not fixed. thr 0.26 is still the best threshold.
- Likely reason: the sum of positive steps equals the run count averaged over all thresholds (∫ runs(p > τ) dτ). A faint bridge (p ≈ 0.3) costs little in the loss, but at thr 0.26 it becomes a full bridge. Idea to try: count on a sharpened map at the operating threshold, `sigmoid((logit - logit(0.26)) / T)`, and/or a higher weight.

## Sharpened crossing-loss result (2026-10-18 evening)
- `--topo 1 --topo-temp 0.2`: **0.492 vs 0.463** baseline. Paired difference +0.029 ± 0.0016 (SEM), with 146 wins out of 158 volumes, so the gain is real. TopoScore 0.060 → 0.117 (k0 F1 0.20, k2 0.14), SurfDice 0.711 → 0.736, VOI 0.560 → 0.571. k1 F1 is still about 0 (0.002), so tunnels remain unsolved.
- Counting at the operating threshold is what matters: the unsharpened version was a tie (+0.003).

## nnU-Net fine-tunes (started 2026-10-18 18:25, finished ~23:55)
Done. nnU-Net validation Dice: Crossing 0.582, Finetune 0.584 (baseline 0.583). Official metric (158 vols, thr 0.26, min 20k), scored 2026-10-19:
| Model | score | Topo | SurfDice | VOI | k0 / k1 / k2 F1 |
|---|---|---|---|---|---|
| nnU-Net baseline | 0.594 | 0.298 | 0.873 | 0.568 | 0.54 / 0.012 / 0.12 |
| + Finetune 250 epochs (control) | 0.596 | 0.303 | 0.875 | 0.568 | 0.54 / 0.010 / 0.14 |
| + **Crossing** 250 epochs | **0.600** | 0.320 | 0.874 | 0.567 | 0.56 / 0.014 / 0.22 |
Paired differences: Crossing − Finetune +0.004 ± 0.002 (84/158 wins); Crossing − baseline +0.007 ± 0.002 (100/158); Finetune − baseline +0.002 ± 0.002. Small gain, about 2 SEM over the control, all from TopoScore (k2). Tunnels (k1) still unsolved.

tmux `nnunet`, windows `nnCrossing` and `nnFinetune`. Logs: `work/nnunet_Crossing.log`, `work/nnunet_Finetune.log`. About 65 s/epoch × 250 epochs, so both should be done around 23:00, then nnU-Net writes `fold_0/validation/*.npz`. Score each with `evaluate.py --probs-dir <results>/nnUNetTrainer{Crossing,Finetune}__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation --out val_nn_{crossing,finetune}.csv` and compare with 0.594.

## Separator oracle (2026-10-19): rejected
Idea: a bridge between wraps is a hole in the gap ("separator") between them, so predict separators and carve them out. Oracle `oracle_separator.py` carves the TRUE separators (Voronoi border between label sheet instances, 2 voxels thick, then fill_holes) out of the Crossing nnU-Net prediction. All 158 vols: none 0.600, gap≤2 0.595, gap≤6 0.588; carving only inside merged components 0.597 / 0.592. VOI never improves. Reasons: labels miss many real sheets (wide cuts fragment them), and after the ignore mask nnU-Net rarely merges wraps. So nnU-Net tunnels are mostly holes inside single sheets. Don't train a separator class.

## Long training (2026-10-19 10:17–19:42 UTC): done, 0.6045
Result (158 vols): score 0.6045, Topo 0.329, SurfDice 0.877, VOI 0.568, k0/k1/k2 F1 0.57/0.013/0.24; nnU-Net val Dice 0.583. Paired: vs Crossing 0.600 +0.004 ± 0.002 (99/158 wins); vs baseline 0.594 +0.011 ± 0.002 (109/158). Tunnels (k1) unchanged.
Setup:
`nnUNetTrainerCrossingLong` (nnunet_trainers.py): continues the Crossing model's checkpoint_final for 1000 more epochs, lr 1e-3 poly, crossing loss on. About 31 s/epoch. Log `work/nnunet_CrossingLong.log`. Scored with `evaluate.py --probs-dir <results>/nnUNetTrainerCrossingLong__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation --out val_nn_crossing_long.csv` (`work/eval_nn_crossing_long.log`).
User decision 2026-10-19: do NOT port the 1st-place post-processing.

## Final model on fold all: DONE (2026-10-19 20:07 – 2026-10-20 16:09 UTC)
`run_final.sh` (tmux `nnunet` window `final`) retrains the best fold-0 recipe on all 786 volumes: nnUNetTrainerBase 1000 epochs -> nnUNetTrainerCrossing 250 -> nnUNetTrainerCrossingLong 1000. About 31 s/epoch, 20 h in total. All three stages finished without errors. No validation (fold all; nnUNetTrainerBase skips the final validation, which would predict all 786 training volumes). Logs `work/final_<trainer>.log`. Re-running the script skips stages that have `checkpoint_final.pth`. Output: `<results>/nnUNetTrainerCrossingLong__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_final.pth`.

## Submission (2026-10-20): scored public 0.5756, private 0.5989
- Kaggle user `andreimihalea`, token in remote `~/.kaggle/access_token` (Kaggle CLI 2.2.4 via `uvx kaggle`).
- Private dataset `andreimihalea/vesuvius-crossing-nnunet` (v3: model + 91 wheels = nnU-Net's full dependency tree minus torch/CUDA, for Python 3.12/3.13). Kaggle runs Python 3.13; it lacked `cc3d` and more, so all dependencies are bundled. Kaggle unpacks uploaded `.tar.gz`, so source-only packages are built into wheels. Upload with `-r zip`.
- Private kernel `andreimihalea/vesuvius-crossing-nnunet-submit` (a kernel cannot share the dataset's name: 409 Conflict). v2 ran fine on the public test placeholder: 82 s/volume on one T4, 2 GPUs used. Inputs mount at `/kaggle/input/datasets/<user>/<slug>/` and `/kaggle/input/competitions/<competition>/`.
- Submitted kernel v2 output on 2026-10-20 19:37 UTC as submission 57451195 (late submission, no prize): **public 0.5756, private 0.5989** (local fold-0 estimate 0.6045). For reference, 1st place: 0.606/0.627, and its ensemble without post-processing 0.572/0.596. A second submission, 57451653 (19:56 UTC, no description), was not made by Claude; it was still PENDING at 2026-10-21 00:55. Check: `uvx kaggle competitions submissions vesuvius-challenge-surface-detection`.

## Next steps
0. **Port the sharpened crossing loss to nnU-Net** as a custom trainer: subclass `nnUNetTrainer`, wrap `_build_loss`, and use the full-resolution output, softmax channel 1, ignore label 2. Fine-tune from fold-0 `checkpoint_final.pth` with `-pretrained_weights`, then compare against 0.594. Optional sweeps on our U-Net: `--topo-temp` 0.1/0.4, `--topo` 2.
1. ~~Score nnU-Net~~ (done, 0.594). Old command, kept for reference:
   ```bash
   cd /root/vesuvius_challenge
   uv run evaluate.py --probs-dir work/nnunet/results/Dataset100_VesuviusSurface/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation --out val_nnunet.csv
   ```
   Then try `--thr 0.5` and `--seed-thr 0.5` on `--n 40`.
2. Port the 1st-place post-processing (per-sheet closing r=3, plugging LUT, height-map patching, fill holes, 1-voxel gap between sheets). Code is in the top notebook `tonylica/nnunet-4-model-7-5-2-1-final-submit-so-long` (scriptVersionId 300373304).
3. Train longer (2000–4000 epochs need a custom trainer) and with larger patches (192/256), then ensemble.
4. Final model: `prepare.py --fold all` / nnU-Net fold `all`, then build the submission (zip of `<id>.tif` uint8 masks).

## nnU-Net commands (for reference)
```bash
export nnUNet_raw=$PWD/work/nnunet/raw nnUNet_preprocessed=$PWD/work/nnunet/preprocessed nnUNet_results=$PWD/work/nnunet/results
uv run nnunet_prepare.py
uv run nnUNetv2_plan_and_preprocess -d 100 -pl nnUNetPlannerResEncM -c 3d_fullres -np 8 --verify_dataset_integrity
uv run nnUNetv2_train 100 3d_fullres 0 -p nnUNetResEncUNetMPlans --npz   # add --c to resume
```

## Fetching Kaggle pages (no auth)
- Notebook source: `curl -L https://www.kaggle.com/kernels/scriptcontent/<scriptVersionId>/download`. To find the version id, use `r.jina.ai` with `X-Return-Format: html` and grep for `scriptVersionId`.
- Rendered pages: `https://r.jina.ai/<kaggle url>`. Anonymous access is sometimes rate-blocked for a few hours.
