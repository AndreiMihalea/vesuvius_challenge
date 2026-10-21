# State of the Art in 3D Segmentation for the Kaggle "Vesuvius Challenge – Surface Detection" (status as of October 2026)

The best approach to this competition is already known: a long-trained, self-configuring **nnU-Net (plain or ResEnc) ensemble over several patch sizes, followed by topology-repair post-processing**. That recipe won (1st place, private LB 0.627).\[1\]\[2\] The organizers wrote that "almost all of the top 10 submissions used large ensembles of our baseline, a ResEncUNet trained with the nn-UNet framework", while transformers, 2D/2.5D models, DINOv3-style foundation models and MONAI pipelines all lost. The competition **ended in February 2026** (writeups dated Feb 28, 2026; prizes announced March 2026),\[1\]\[3\] so the realistic goals now are (a) reproducing or beating the winning recipe as a late submission or research project, or (b) moving the result into the scroll-unwrapping pipeline, where the organizers say topology in densely packed regions is still unsolved.\[4\]

## TL;DR
- **Competition status and target:** The competition ran from Nov 13, 2025 to an extended end in mid-February 2026. The prize pool grew from the $100,000 in Kaggle's launch post to the $200,000 listed on scrollprize.org/winners. It was won by the "Vesuvius Team" (Tony Li, OzanM., Yiheng Wang and PaulG, i.e. Paul Geiger, per the 1st-place writeup and the organizers) with an nnU-Net ensemble plus post-processing (private 0.627, public 0.606). Gold solutions clustered at a private LB of about 0.613–0.627, so a well-executed nnU-Net pipeline should land around 0.61, and the remaining gains come from topology repair, not architecture.
- **What actually works:** nnU-Net v2 (default PlainConvUNet or ResEnc M/L) trained 2,000–8,000 epochs (vs. the default 1,000), patch sizes 128–256 (ensembling several), CE+Dice optionally plus Skeleton Recall / Medial-Surface Recall or clDice, light augmentation, probability-averaged ensembles with a low threshold (~0.2–0.26), then post-processing: remove small components, close/plug holes, height-map patching, Hessian/Frangi sheetness filtering or hysteresis thresholding. Post-processing alone added about +0.02–0.035.
- **What doesn't work / is still open:** pseudo-labeling, aggressive augmentation, SE modules, 2D/2.5D and transformer-only models, and metric "hacks" (Betti-2 trick: +0.007 public, +0.001 private). Separating **touching/merged sheets** was unsolved by every top team, including 1st place. It is the main frontier, and distance-transform targets (5th place), cascades/refinement networks (3rd, 10th, 11th) and diffeomorphic shape correction (10th) are the most promising directions.

## Key Findings

### 1. Competition understanding

**Task.** Segment the papyrus *recto surface* in 3D synchrotron micro-CT chunks of carbonized Herculaneum scrolls.\[3\] The masks are fed into the Vesuvius Challenge virtual-unwrapping pipeline (mesh tracing in VC3D, then flattening, then ink detection).\[3\] The hosts describe the data as "CT chunks + binary masks (smoothed sheet positions)" and the goal as reproducing those labels on held-out data.\[5\]

**Data.** A Japanese post-competition summary of the top solutions (18th-place author, Speaker Deck) gives the data as **786 training volumes from 6 scroll IDs, median size 320×320×320, uint8 input, labels {0 = background, 1 = surface, 2 = ignore}**.\[6\] Scans come from ESRF (Grenoble) and DLS (Oxford).\[3\] The ignore label matters: participants found that excluding label 2 from the loss is essential, and that patches with a very small valid ratio stall training.\[7\] Data quality varies widely: blurred, low-contrast, overly bright and densely compressed volumes are the hard cases.\[6\]

**Metric.** The official leaderboard score is a linear blend: **0.30 × TopoScore + 0.35 × SurfaceDice + 0.35 × VOI score**.\[6\]
- *TopoScore*: Betti-matching-based agreement of topological features (components, tunnels/holes, cavities); strict on mergers.\[6\]\[8\]
- *SurfaceDice*: surface-distance Dice with **2.0-voxel tolerance**, so exact voxel overlap is not required.\[6\]
- *VOI*: a variation-of-information score on connected components (26-connectivity); penalizes merges and splits.\[6\]

What this means in practice: (i) holes in a sheet cost TopoScore and VOI far more than they cost Dice; (ii) merging two adjacent sheets into one component is severely penalized by both TopoScore and VOI; (iii) small spurious components hurt VOI and Betti-0. The 2-voxel SurfaceDice tolerance lets you thin, smooth or slightly dilate predictions without much accuracy cost. An external diagnostic project notes that the VOI term "rewards predicted mass",\[9\] so a lower threshold (more recall) often helps.\[9\] The official metric library is `topological-metrics-kaggle` (C++ Betti matching with Python bindings), which participants installed offline in notebooks.\[10\] One external user reported it segfaults on at least one patch.\[9\]

**Constraints.** It is a code competition: submissions run in Kaggle notebooks with internet disabled, and the output is a zip of one `.tif` mask per test volume (`[image_id].tif`, same shape and dtype as the training masks).\[3\] The 12th-place author's blog lists the constraints they designed around as a **9-hour inference time limit and 16 GB GPU memory**.\[11\] Public notebooks used **2×T4**\[12\] (the 1st-place team's reference notebook is titled "surface-nnunet-training-inference-with-2xt4").\[1\]\[2\] The winning team said they hit GPU-memory and disk limits with even a 2-model ensemble and had to redesign their pipeline to fit 4 models.\[1\]

**Timeline.** Launched Nov 13, 2025 with a prize pool of $100,000 according to Kaggle's launch post; scrollprize.org/winners lists the final pool as $200,000. Entry deadline Feb 6, 2026. The final deadline was **extended two weeks** after a test-set fix,\[10\] to Feb 13, 2026 according to CompeteHub's listing (start 11/13/2025, end 2/13/2026). Writeups were posted Feb 28, 2026, and prizes were announced in March 2026 (1st Vesuvius Team $60K, 2nd "risk of overfitting" $40K, 3rd "W & A" $30K, 4th Starry $20K, 5th Dieter $15K, 6th #hui, 7th DECEM, 8th lingyundev, 9th 阿對對對對隊).\[13\] Team-count figures disagree across sources (1,329 / 1,391 / 1,427).\[11\]

**Known pitfalls and quirks.**
- **Test-set correction mid-competition.** The hosts found "a critical issue in the data", **updated the test set and rescored submissions, but did not update the training set**, and extended the deadline. Forum users called this unfair because the train labels stayed inconsistent with the corrected test labels.\[10\] The 24th-place author said their precision-focused strategy (Tversky α=0.7) stopped working after the fix "and the topo score was really taken into account".\[14\]
- **Train/test label-processing mismatch.** The 7th-place author found that the organizers' GitHub repo applies a specific processing pipeline to the *test* labels. Applying that same processing to the *training* labels was "probably the single most impactful thing I did".\[15\]
- **Public/private shake-up.** Public scores were systematically lower than private for top teams (1st: 0.606 → 0.627; 4th: post-processing gains larger on private).\[2\] The 9th-place team went 0.623 → 0.616.\[16\] Tuning post-processing on the public LB was risky.
- **Verso sheets and incomplete fragments.** Small partial sheets near volume corners are legitimate. Small interior fragments are often verso or noise (24th place).\[14\]

### 2. State of the art in 3D volumetric segmentation and how it played out here

**Benchmark evidence: "nnU-Net Revisited" (Isensee et al., MICCAI 2024, arXiv:2404.09556).** Under rigorous validation, the recipe for SOTA is "1) employing CNN-based U-Net models, including ResNet and ConvNeXt variants, 2) using the nnU-Net framework, and 3) scaling models to modern hardware resources". Transformer-based architectures (UNETR, SwinUNETR, nnFormer, etc.) "generally failed to match CNN performance, including against the original nnU-Net baseline".\[17\] Mamba models' reported gains were traced to their residual U-Net backbone rather than the Mamba blocks.\[18\] The paper released **ResEnc M/L/XL** presets (≈9 / 23 / 37 GB VRAM targets).\[17\] ResEnc L and XL and STU-Net were the robust top performers, MedNeXt was competitive, and MONAI Auto3DSeg/SegResNet trailed nnU-Net under default configs.\[17\]\[19\]

| Family | Representative models | Evidence (general 3D benchmarks) | Evidence in this competition | Verdict for this task |
|---|---|---|---|---|
| Self-configuring CNN U-Net | nnU-Net v2 (PlainConvUNet), **ResEnc M/L/XL** | Top tier in nnU-Net Revisited; scales with compute | Used by every gold team; 1st, 4th, 12th used near-default configs | **First choice** |
| Large/pretrained CNN | STU-Net, MedNeXt | STU-Net L robust; MedNeXt competitive | Not reported in gold writeups | Optional diversity |
| MONAI CNN | SegResNet (Auto3DSeg) | Below nnU-Net under defaults | 18th/27th: MONAI "far behind nnU-Net" | Low priority |
| Hybrid / ViT | UNETR, SwinUNETR, TransUNet | Generally fail to beat CNNs | Public TransUNet-ResNeXt50 notebook reached LB ~0.552–0.567 | Low priority |
| Pure-attention | **Primus-B/M** (DKFZ) | Closes most of the CNN gap; transformer actually used (removing it drops 87.98 → 84.48 avg DSC) | 10th place: Primus-B and Primus-B V2 complemented ResEnc-L in an ensemble | Good **ensemble diversity** member |
| Mamba | U-Mamba, SegMamba | Gains attributable to residual backbone | Not reported | Skip |
| Foundation / promptable | SAM-Med3D, SAM 2 adaptations, VISTA3D, MedDINOv3 | Built for organs/lesions with prompts | DINOv3-based and 2D methods "all lost" per the top-solution summary | Skip for leaderboard; consider for SSL pretraining |
| 2D / 2.5D | UNet++, SegFormer, Mask2Former slice models | Fast, but no 3D topology | 2.5D UNet++ notebooks and 18th-place's early Mask2Former underperformed; 3D needed | Only as cheap baseline |

**Why CNNs win on sheets.** Papyrus surfaces are thin (a few voxels), locally planar, closely stacked, and separated by gaps of a voxel or two. The decision depends on local texture and fine intensity edges, which convolutions with full-resolution skip connections capture well. Long-range context matters mainly for disambiguating which sheet a voxel belongs to. Large patches (192–256) supplied it more cheaply than attention, and that context matters because the organizers' launch post says the scoring rewards "surface connectivity (no gaps, holes, sheet‑switches or mergers)". Still, the 9th-place team reported a CV ranking of "192>224>256" in their writeup comments and said 224/256 patches raised Dice but *lowered* TopoScore, which is why winners ensemble several patch sizes rather than picking the largest.

**Training-length finding (most important practical lever).** Top teams trained far beyond nnU-Net's 1,000-epoch default; the organizers noted that the Kaggle experts found recipes "such as training for more than 4,000 epochs". Examples: 1st 4,000 epochs (patch 128, batch 2, then fine-tuned at 192 and 256 for 250 epochs; extra from-scratch 160/192/224 models),\[1\] 3rd 8,000 epochs (cascade, ~38 h on 2×H100), 7th 2,000, 12th 2,000–4,000. 12th place's ablation: **1,000 ep 0.552 → 2,000 ep 0.593 → 4,000 ep 0.613** private.\[6\] The organizers' own paper model for recto surfaces (nnU-Net-style ResEnc U-Net, patch 256³, batch 2, AMP fp16, Medial Surface Recall loss) was scheduled for 7,500 epochs and logged to epoch 3,864.\[20\]

**Optimizer.** 7th place, same config at 1,000 epochs: SGD (nnU-Net default) 0.590, AdamW 0.594, Muon+AdamW 0.595 private.\[15\] That is a small but consistent gain.\[15\] 9th place used a "MUSGD" variant.\[16\]

### 3. Thin-structure and topology methods

| Method | Idea | Cost | Evidence here | Recommendation |
|---|---|---|---|---|
| **Skeleton Recall Loss** (Kirchhoff et al., ECCV 2024; MIC-DKFZ/Skeleton-Recall) | Precompute tubed skeleton of GT on CPU (dilate with radius-2 diamond); soft-recall of prediction on it | Near-free (CPU, precomputed) | Host baseline; 7th, 9th, 18th, 24th (+0.004 for 24th); "more stable than clDice" | **Default add-on** |
| **Medial Surface Recall** (host trainer `nnUNetTrainerMedialSurfaceRecall`) | Skeleton Recall adapted to 2-manifolds (medial surface instead of centerline) | Near-free | Host baseline; 8th, 9th; organizers' paper model | **Use (best fit for sheets)** |
| clDice (Shit et al., CVPR 2021) | Differentiable soft-skeleton precision/recall | GPU-heavy in 3D | 3rd place used CE+Dice+clDice; 7th saw no gain over SkelRecall | Optional |
| cbDice (Shi et al., MICCAI 2024) | clDice + boundary/radius weighting for diameter balance | Moderate | Not reported | Low priority (vessel-oriented) |
| Betti-matching / persistent-homology losses (Stucki et al., ICML 2023; Betti-Matching-3D) | Match persistence intervals of pred vs GT | Expensive in 3D | Used as *diagnostic/post-processing* (4th: local Betti-1 detection in 20³ chunks) | Use for post-processing, not loss |
| Surface Dice loss (MemBrain v2 style) | Differentiable surface overlap | Moderate | 10th place (CE+Dice+SkelRecall+SurfDice) | Optional |
| Tversky / boundary DoU | Precision/recall or boundary weighting | Cheap | 24th's Tversky precision focus failed post-fix; 113th used BoundaryDoU | Not recommended |
| **Distance-transform regression** | Predict signed/unsigned distance to sheet (or skeleton) and threshold | Same as segmentation | 5th place ("Dieter"); organizers cite it as worth pursuing | **Promising for touching sheets** |

**Separating touching/merged sheets.** This was the unsolved problem. 1st place: "we did not find an effective solution to the problem of touching sheets. We just relied on nnu-Net to minimize their occurrence."\[1\] 9th place built a "de-adhesion" step (2D PCA normal ray-casting) but found it ineffective when high-confidence components were already merged.\[16\] 24th place applied a more conservative threshold near detected 2D bifurcations.\[14\] Approaches with the best prospects:
1. **Distance-transform targets (5th place).** Predicting distance to the sheet centre gives a valley between adjacent sheets that a threshold can split. The organizers explicitly recommend "using a Distance Transform rather than a Binary Segmentation".\[21\] (The 5th-place method details and score could not be verified.)
2. **Cascades and refinement networks.** 3rd and 11th used nnU-Net 3D cascades (low-res stage → full-res conditioned on it).\[6\] 10th stacked refinement networks that take [image, masks] and learn to fix errors, plus a final diffeomorphic network for "shape calibration and thickness modification", which the author called the biggest boost.\[22\] hengck23 (120th) got about +0.02 from a refinement U-Net on concat(image, prob).\[12\]
3. **Watershed and instance cues borrowed from connectomics.** The organizers' own fiber pipeline uses GPU watershed-from-minima to split touching fibers.\[21\] Applying the same idea to a distance-transform or probability map is a natural sheet splitter, though no top team reported success with it here.
4. **Hysteresis thresholding** with high seeds (T_high 0.8–0.9) and low growth (T_low 0.25–0.5), restricted along z (8th place),\[6\] avoids bridging neighbouring sheets while keeping recall.

### 4. Prior Vesuvius Challenge work

- **Volume Cartographer → VC3D** (ScrollPrize/villa): the team's semi-automatic surface tracer.\[23\] It grows quad meshes (`tifxyz` format) over a recto-surface prediction volume using normal grids, GrowPatch (Ceres optimization), Copy Out/In to adjacent wraps, and the newer **Lasagna** multi-sheet optimizer, which consumes predicted normals and a "fractional winding position" field built from distance transforms to sheet skeletons. VC3D works well "when the prediction is clean and the layers are separable" but fails when "predicted sheets incorrectly merge".\[21\]
- **ThaumatoAnakalyptor** (Julian Schilliger, part of the $700K 2023 Grand Prize team): an automatic pipeline combining classical gradient-based surface-point extraction with deep-learning instance segmentation of point clouds and sheet stitching. It is now marked deprecated and superseded by VC3D and Lasagna.\[23\]\[24\]
- **Khartes** (interactive segmentation tool, community) and **Spiral fitting** (Paul Henderson; global deformed-spiral prior)\[25\] are complementary geometry tools. They are useful context but not used in the Kaggle pipeline. (Khartes details were not re-verified for this report.)
- **Released checkpoints (Hugging Face `scrollprize`)**: `surface_recto_3dunet` (paper pipeline) and **`surface_m7_nnunet` (the nnU-Net component of the 1st-place Kaggle solution)**, plus `fiber_hz_vt`, DINO 3D SSL models (`dinovol_*`), and neural-tracing checkpoints. The organizers summarized the winning recipe as: "take an off-the-shelf nnU-Net model configured for a couple of patch sizes, train it for a very long time (well over the default 1,000 epochs), and average their predictions together".\[21\]
- **Post-competition progress prizes**: Scroll Fiesta (Ben Kyles, $10K, May 2026), "an automatic mesher of surface predictions that attempts to fix topological mistakes"; the Scroll Slab Viewer, a 3D viewer for the Kaggle data.\[13\] The organizers report they are "close to saturating the performance of models trained on the approximate surface labels", and that predictions "disappear or form blobs" in dense, twisted or broken regions.\[4\] **Label quality** is now named as one of the main bottlenecks.\[21\]
- **Kaggle Ink Detection (2023) and what transferred.** 1st place (ryches et al.) used a two-stage approach with a 9-model ensemble\[26\] of 3D CNN encoders and 2D decoders. 4th place relied on Kinetics-pretrained 3D ResNets ("a huge difference") and depth crop/paste augmentation.\[27\] The 2023 Grand Prize ink model used TimeSformer and 3D CNNs.\[28\] For surface detection the transfer was *partial*. The 2.5D "3D encoder → 2D decoder" ink designs do not fit a fully 3D output, and gold teams found 2D/2.5D inferior. What did carry over: heavy ensembling, flip/rotation TTA, and the lesson that precise data handling matters more than architecture. Kinetics-pretrained ResNet3D-50 remains the backbone of the current production ink model.\[21\]

### 5. Kaggle competition insights: top solutions

| Place | Private (public) | Model | Epochs / patch | Loss | Ensemble / TTA | Key post-processing |
|---|---|---|---|---|---|---|
| 1st (Vesuvius Team) | 0.627 (0.606) | nnU-Net (default), multi-patch | 4000 ep @128; FT @192/256 (250 ep); scratch 160/192/224 | Not specified (CE+Dice default reported by summaries) | 4-model weighted softmax fusion, thr 0.20–0.26 | Remove <20K-voxel CCs; 2×2×2 LUT 1-voxel hole plug; height-map gap patching; binary closing r=3; fill holes |
| 2nd (risk of overfitting) | 0.622 | (no public writeup found) | – | – | – | – |
| 3rd (W & A) | 0.620 | nnU-Net M **3D cascade** | 8000 ep; train 128 / infer 192 | CE+Dice+clDice | Cascade stages; ±15° rotation TTA | Hessian ridge filter + EDT dilation |
| 4th (Starry) | 0.619 | nnU-Net ResEnc (7-stage) | 2000 ep | Default | Single model | Dust removal; local Betti-1 detection + PCA-plane `binary_fill_holes` (0.609 → 0.622 private) |
| 5th (Dieter) | ~0.619 | Distance-transform formulation (details unverified) | – | – | – | – |
| 7th (DECEM) | 0.618 per summary* | nnU-Net ResEncM 3d_lowres | 2000 ep @128 | CE+Dice+SkelRecall | 2 models (raw vs processed labels) + TTA | Frangi sheetness → coherence-enhancing diffusion → anisotropic closing → dust removal |
| 8th | 0.616 | nnU-Net ResEnc L | 300–500 ep @224–288 | CE+Dice+MedialSurfRecall | 3-model mean | Anisotropic diffusion + z-only hysteresis |
| 9th | 0.616 (0.623) | Host MedialSurfaceRecall nnU-Net | @192 | MedialSurfRecall | Single + TTA | Skeletonize+EDT "line norm"; gap/diagonal fill looped ×5 (linear gains) |
| 10th | 0.614 | ResEnc-L + Primus-B + Primus-B V2 | ~400 ep @160 | Dice+CE+SkelRecall+SurfDice | 3 models → ensemble net → 2 refiners → diffeomorphic net | Learned |
| 12th | 0.613 | nnU-Net lowres + fullres | 2000–4000 @128 | – | 4-model weighted + TTA | Hysteresis + opening/closing |
| 18th | 0.609 | 3 "specialist" nnU-Nets (blur/bright/general) | 500 | CE+Dice+SkelRecall | Quality-routed weights | 3×3×3 median filter ×7 (+~0.02) |

*The 7th-place writeup itself states "Public LB 0.618 / Private LB 0.591",\[15\] which conflicts with its 7th rank between teams scoring 0.619 and 0.616. The 0.618 private figure from the summary is more consistent with the leaderboard, so treat 0.591 as likely a single-model or typo figure.

**1st-place post-processing ablation (cumulative, public / private):** none .572/.596 → remove small components .586/.614 → plug small holes .598/.622 → patch large holes .601/.625 → binary closing .606/.627 → fill holes .606/.627.\[2\] Most of the gain comes from removing small components and plugging holes.

**12th-place stepwise ablation (private):** 1000-ep baseline 0.552 → +hysteresis 0.570 → +2-fold ensemble 0.590 → +opening/closing 0.593 → +fullres models 0.598 → +4-model ensemble 0.606 → +TTA 0.607 → +4000 ep 0.613.\[6\]

**Host baseline.** About two months before the end, the hosts published a strong nnU-Net baseline (1,200 epochs, Skeleton/Medial-Surface Recall, strong augmentation, post-processing) that was in the gold zone when released. Most top teams built on it.\[6\]

**Public notebooks.** Early leaders were non-nnU-Net: "[Inference] Vesuvius Surface 3D Detection" (ipythonx; TransUNet/SegFormer-MiT-B2/ResNeXt models, best public 0.567)\[29\] and Tony Li's TransUNet-ResNeXt50 with SparseCenterlineDice+DiceCE (public 0.552).\[12\] A public hysteresis post-processing function was widely reused.\[14\] The 1st-place team credits "surface-nnunet-training-inference-with-2xt4" (Jirka Borovec) as their base.\[1\]

**What did not work (cross-team):** pseudo-labels (7th, 10th, 27th; "noise was too much"), stronger augmentation (7th used *lower* than default probabilities), SE modules, Mask2Former/MedDINOv3/MONAI/JAX pipelines,\[6\] smaller sliding-window step (24th: detected more verso sheets),\[14\] de-adhesion, and the Betti-2 metric hack (+0.007 public, +0.001 private).\[6\] Post-processing is model-specific: the 113th-place team found the 1st-place post-processing did not help their model.\[30\]

## Recommendations: a practical solution plan

**Stage 0 — Set up correctly (day 1).**
- Use nnU-Net v2 with the hosts' trainer (`nnUNetTrainerMedialSurfaceRecall`) or Skeleton Recall. Make sure label 2 is mapped to nnU-Net's ignore label.
- Apply the organizers' test-time label processing to training labels as one variant (7th place's key gain). Keep a raw-label variant for diversity.
- Install `topological-metrics-kaggle` locally and score validation with the exact leaderboard blend. Do not tune on Dice.

**Stage 1 — Validation without leakage.**
- Split by **scroll ID / spatial region**, not random chunk. Adjacent chunks from the same scroll region share sheets and label style. With 6 scroll IDs, use grouped folds by scroll (or by contiguous region within large scrolls) and check that CV ranks agree with the public LB.
- Hold out a fold weighted toward hard cases (blurred, low-contrast, compressed). 18th place showed that these converge much later (90%-convergence at about 54–96 epochs vs about 25–60 for normal data), which is why long training helps.\[6\]
- After choosing hyperparameters on folds, retrain on all data, as the 1st and 24th place teams did.\[14\]

**Stage 2 — Strong single model (expected ~0.59–0.61 private).**
- nnU-Net 3d_fullres (PlainConvUNet or ResEnc M; ResEnc L if VRAM allows), patch 128³ or 160³, batch 2, CE+Dice+Medial-Surface/Skeleton Recall, AdamW or Muon instead of SGD. Augmentation at or below default strength.
- Train **≥2,000, ideally 4,000 epochs**. This is the single biggest lever (+0.06 from 1,000 to 4,000 in 12th place's data).\[6\]

**Stage 3 — Topology post-processing tied to the metric (+0.02–0.035).** Apply in this order, tuning each step on CV with the official metric:
1. Low threshold (0.2–0.3) on averaged probabilities, or hysteresis (seed 0.8–0.9, grow 0.25–0.5, z-restricted).
2. Remove connected components below ~5K–20K voxels, but **keep small components touching volume faces or corners** (24th place: legitimate corner sheets can be 300–400 voxels).\[14\]
3. Plug 1-voxel holes (2×2×2 LUT, 1st place), then repair larger holes with height-map interpolation per sheet (1st)\[1\] or local PCA-plane `binary_fill_holes` driven by Betti-1 detection (4th).\[6\] Reject patches that increase the hole count.\[1\]
4. Light anisotropic closing (z-radius > xy-radius), *or* a Frangi-sheetness / coherence-enhancing-diffusion pass (7th),\[15\] *or* repeated 3×3×3 median filtering (18th).\[6\] Pick one family; they overlap.
5. Never dilate globally. It merges adjacent sheets, and VOI and TopoScore punish that more than the gain in SurfaceDice.

**Stage 4 — Ensemble and diversity (+0.01–0.02).**
- 3–4 nnU-Nets with different patch sizes (128/160/192/224–256), optionally one cascade and one Primus-B for architectural diversity. Fuse post-softmax probabilities with weights tuned on CV, then threshold.
- Fine-tuning a 4,000-epoch 128-patch model at 192/256 for 250 epochs is a cheap way to get large-context members.\[1\] The 1st-place fine-tuned 192 model scored 0.614 private alone.\[2\]

**Stage 5 — Research bets for touching sheets (highest upside, uncertain).**
- Add a distance-transform regression head (distance to sheet medial surface) and split merged components by watershed on it.
- Train a refinement U-Net on [image, ensemble probability] (10th/120th) with coarse-dropout corruption so it learns to fill holes and split bridges.
- Initialize from the organizers' `surface_m7_nnunet` or `surface_recto_3dunet` checkpoints, or from `dinovol` SSL features. Check license terms and leakage first.

**Inference within the 9-hour, 16 GB limit.**
- Use both T4s (one model or one half of the volumes per GPU), fp16 autocast, and nnU-Net's sliding window with step ~0.5. Reducing the step size did not help.\[14\]
- Limit TTA to flips (or ±15° rotation as in 3rd place). Train on 128 but infer on 192 for more context (3rd place's asymmetric patch).\[6\]
- Write probabilities to disk as uint8 or fp16 to avoid RAM and disk blow-ups, and run CPU post-processing in parallel with GPU inference.
- Benchmark per-volume time on a 320³ volume, multiply by the expected test count with a 30% safety margin, and only then add ensemble members.

**Ranked model choices (expected value under constraints):**
1. nnU-Net PlainConvUNet/ResEnc M, long-trained, multi-patch ensemble.
2. ResEnc L at larger patch size.
3. nnU-Net 3D cascade (strong but compute-heavy).
4. Primus-B as a diversity member.
5. Refinement/diffeomorphic second stage.
6. Everything else (SwinUNETR, UNETR, SegResNet, Mamba, SAM-type, 2.5D): low expected value.

## Caveats
- **Competition is closed.** Late submissions may still be possible on Kaggle but carry no prizes. Check the competition page for late-submission availability (not verified).
- **Unverified details.** I could not open the official Kaggle overview and data pages. The 9-hour/16 GB limits come from the 12th-place author's blog, and the data statistics (786 volumes, 6 scrolls, 320³ median) come from a community summary that its author says was partly AI-assisted.\[6\] The 5th-place distance-transform method and the 2nd- and 6th-place approaches could not be verified, and no public 2nd-place writeup was found.
- **Score conflicts.** The 7th-place writeup's private 0.591 conflicts with its rank (see table note). Team counts disagree across sources (1,329–1,427).
- **Label noise.** The organizers say the training labels are "approximate", may "drift slightly off the true surface",\[21\] and were changed for test but not for train.\[10\] CV-to-LB correlation is therefore imperfect, and conservative post-processing (as in 9th place's "SUPER-CONSERVATIVE" approach) is safer than aggressive LB-tuned hacks.\[16\]
- **Generalization of the benchmarks.** The nnU-Net Revisited results are from medical CT/MRI. Their transfer to scroll CT is supported by this competition's outcome but not by a controlled study on scroll data.

## Sources

1. [1st Place Solution for the Vesuvius Challenge - Surface Detection Competition](https://www.kaggle.com/c/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su)
2. [Tony Li](https://www.kaggle.com/tonylica)
3. [Vesuvius Challenge - Surface Detection - CompeteHub](https://www.competehub.dev/en/competitions/kagglevesuvius-challenge-surface-detection)
4. [The hard parts are getting clearer](https://scrollprize.substack.com/p/the-hard-parts-are-getting-clearer)
5. [Back to the Challenge: \$100K Kaggle Surface Detection](https://scrollprize.substack.com/p/back-to-the-challenge-100k-kaggle)
6. [kaggle Vesuvius Challenge - Surface Detection速報まとめ](https://speakerdeck.com/sugupoko/vesuvius2-combined)
7. [🏺【Kaggle Vesuvius】Training, Inference, and Submission with 2.5D UNet++, and What I Did to Improve LB｜Hachi｜業務効率化｜AI駆動開発](https://note.com/hachi4066/n/n722106e23ff3?hl=en)
8. [GitHub - nstucki/Betti-Matching-3D · GitHub](https://github.com/nstucki/Betti-Matching-3D)
9. [GitHub - Jinhojeong/vesuvius-surface-geometry-diagnostic: Geometry-stratified failure diagnostic for Vesuvius Challenge surface models (villa #191): two-model per-bin AUC vs curvature and sheet compression · GitHub](https://github.com/Jinhojeong/vesuvius-surface-geometry-diagnostic)
10. [Vesuvius Challenge - Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/discussion/672634)
11. [Kaggleの火山コンペで12位、金メダルを獲得しました](https://techblog.recruit.co.jp/article-6934/)
12. [Bronze Medal - ChatGPT Vibe Coding!](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/bronze-medal-chatgpt-vibe-coding)
13. [Winners](https://scrollprize.org/winners)
14. [24th Place Solution - Vesuvius Challenge](https://www.kaggle.com/c/vesuvius-challenge-surface-detection/writeups/24th-place-solution)
15. [7th Place Solution for the Vesuvius Challenge](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/7st-place-solution-for-the-vesuvius-challenge)
16. [9th Place Solution SUPER-CONSERVATIVE(0.623-\>0.616)](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/9th-place-solution)
17. [nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation](https://www.alphaxiv.org/abs/2404.09556)
18. [nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation \[Quick Review\]](https://liner.com/review/nnunet-revisited-call-for-rigorous-validation-in-3d-medical-image)
19. [nnU-Net Revisited: A Call for Rigorous Validation](https://arxiv.org/pdf/2404.09556)
20. [Complete virtual unwrapping and reading of a rolled Herculaneum papyrus](https://arxiv.org/pdf/2606.29085)
21. [Open Problems: Why Reading Every Herculaneum Scroll Is Still a Challenge](https://scrollprize.org/2026_open_problems)
22. [IAmParadox](https://www.kaggle.com/iamparadox/writeups)
23. [GitHub - ScrollPrize/villa: The Vesuvius Challenge monorepo · GitHub](https://github.com/ScrollPrize/villa)
24. [Community Projects](https://scrollprize.org/community_projects)
25. [Virtual Unwrapping](https://scrollprize.org/unwrapping)
26. [GitHub - ainatersol/Vesuvius-InkDetection: The Vesuvius Challenge is a machine learning and computer vision competition to read the Herculaneum Papyri. · GitHub](https://github.com/ainatersol/Vesuvius-InkDetection)
27. [4th place solution](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/writeups/posco-dx-heeyoung-ahn-4th-place-solution)
28. [Vesuvius Challenge 2023 Grand Prize awarded: we can read the scrolls!](https://scrollprize.org/grandprize)
29. [\[Inference\] Vesuvius Surface 3D Detection](https://www.kaggle.com/code/ipythonx/inference-vesuvius-surface-3d-detection/notebook?scriptVersionId=285843716)
30. [From Cell Segmentation to Ancient Scrolls: WNet3D + nnUNet (LB 113, 0.582)](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/vesuvius-wnet3d-nnunet)
