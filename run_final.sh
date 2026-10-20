#!/usr/bin/env bash
# Final model: the best fold-0 recipe (0.6045), retrained on all 786 volumes (fold "all", no validation).
# Base 1000 epochs -> Crossing 250 epochs -> CrossingLong 1000 epochs, about 30 s/epoch = about 19 h.
# Run in tmux. Re-running skips finished stages.
set -euo pipefail
cd "$(dirname "$0")"
source "$HOME/.local/bin/env"
export nnUNet_raw=$PWD/work/nnunet/raw nnUNet_preprocessed=$PWD/work/nnunet/preprocessed nnUNet_results=$PWD/work/nnunet/results

for tr in nnUNetTrainerBase nnUNetTrainerCrossing nnUNetTrainerCrossingLong; do
    out=$nnUNet_results/Dataset100_VesuviusSurface/${tr}__nnUNetResEncUNetMPlans__3d_fullres/fold_all
    if [ -f "$out/checkpoint_final.pth" ]; then echo "$tr done, skipping"; continue; fi
    uv run nnUNetv2_train 100 3d_fullres all -p nnUNetResEncUNetMPlans -tr "$tr" 2>&1 | tee "work/final_$tr.log"
done
