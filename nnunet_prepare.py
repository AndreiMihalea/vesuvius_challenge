"""nnU-Net step 1: build nnU-Net's raw dataset from symlinks (no copies) and write our fold split.
Label 2 is declared as nnU-Net's "ignore" label, so it is excluded from the loss automatically.
Run after prepare.py (needs work/split.csv). Next steps: see README, section "nnU-Net"."""
import json

import pandas as pd

from common import DATA, WORK
from prepare import nnunet_val_ids

NAME = "Dataset100_VesuviusSurface"
RAW = WORK / "nnunet" / "raw" / NAME
PREPROCESSED = WORK / "nnunet" / "preprocessed" / NAME
SPACING = json.dumps({"spacing": [1.0, 1.0, 1.0]})  # read by nnU-Net's Tiff3DIO

if __name__ == "__main__":
    ids = sorted(str(i) for i in pd.read_csv(WORK / "split.csv").id)  # the 786 non-deprecated volumes
    for d in ("imagesTr", "labelsTr"):
        (RAW / d).mkdir(parents=True, exist_ok=True)
    for i in ids:
        for src, dst in ((DATA / "train_images" / f"{i}.tif", RAW / "imagesTr" / f"{i}_0000.tif"),
                         (DATA / "train_labels" / f"{i}.tif", RAW / "labelsTr" / f"{i}.tif")):
            if not dst.is_symlink():
                dst.symlink_to(src.resolve())
        (RAW / "imagesTr" / f"{i}.json").write_text(SPACING)
        (RAW / "labelsTr" / f"{i}.json").write_text(SPACING)

    (RAW / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "surface": 1, "ignore": 2},  # "ignore" must be the highest label
        "numTraining": len(ids),
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "Tiff3DIO",
    }, indent=2))

    # same 5 folds as prepare.py (= nnU-Net's own default), written explicitly so nothing can drift
    vals = [nnunet_val_ids(ids, f) for f in range(5)]
    PREPROCESSED.mkdir(parents=True, exist_ok=True)
    (PREPROCESSED / "splits_final.json").write_text(json.dumps(
        [{"train": [i for i in ids if i not in v], "val": sorted(v)} for v in vals], indent=1))
    print(f"{len(ids)} cases -> {RAW}\nfold 0: {len(ids) - len(vals[0])} train / {len(vals[0])} val")
