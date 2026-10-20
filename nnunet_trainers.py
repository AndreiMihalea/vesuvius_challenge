"""nnU-Net trainers that fine-tune the ResEnc M model (checkpoint_final.pth of the same fold) for 250 more epochs.

nnUNetTrainerBase       nnU-Net's default trainer, minus the final validation on fold "all" (it would predict all 786 volumes)
nnUNetTrainerFinetune   control: nnU-Net's own Dice + CE, so extra training alone can be measured
nnUNetTrainerCrossing   same + our sharpened sheet-crossing loss (train.py crossing_loss)
nnUNetTrainerCrossingLong  continues nnUNetTrainerCrossing's final model for 1000 more epochs (longer training)

nnU-Net only finds trainers inside its own package, so link this file there once:
    ln -s $PWD/nnunet_trainers.py .venv/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/
Then: nnUNetv2_train 100 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerCrossing --npz
Final model: run_final.sh trains Base -> Crossing -> CrossingLong on fold "all".
"""
import sys
from pathlib import Path

import torch
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

ROOT = Path(__file__).resolve().parent  # resolves the symlink back to the project folder
sys.path.insert(0, str(ROOT))
from train import crossing_loss  # noqa: E402

RESULTS = ROOT / "work/nnunet/results/Dataset100_VesuviusSurface"
SURFACE, IGNORE = 1, 2


class nnUNetTrainerBase(nnUNetTrainer):
    def perform_actual_validation(self, save_probabilities: bool = False):
        if self.fold == "all":  # no held-out data: validation would only predict the training set
            self.print_to_log_file("fold all: skipping final validation")
        else:
            super().perform_actual_validation(save_probabilities)


class nnUNetTrainerFinetune(nnUNetTrainerBase):
    init_from = "nnUNetTrainer"  # trainer whose checkpoint_final.pth (same fold) we start from

    @property
    def init_ckpt(self):  # fold 0's base model came from nnU-Net's default trainer, fold "all"'s from nnUNetTrainerBase
        name = "nnUNetTrainerBase" if self.fold == "all" and self.init_from == "nnUNetTrainer" else self.init_from
        return RESULTS / f"{name}__nnUNetResEncUNetMPlans__3d_fullres/fold_{self.fold}/checkpoint_final.pth"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.initial_lr = 1e-3  # 10x below nnU-Net's default: we start from a trained model

    def initialize(self):
        super().initialize()
        # load ALL weights, including the output layer (-pretrained_weights would skip it)
        net = self.network._orig_mod if isinstance(self.network, OptimizedModule) else self.network
        net.load_state_dict(torch.load(self.init_ckpt, map_location=self.device, weights_only=False)["network_weights"])
        self.print_to_log_file(f"initialized from {self.init_ckpt}")


class CrossingLoss(torch.nn.Module):
    """nnU-Net's loss + weight * crossing loss on the full-resolution output, counted at the evaluation threshold."""

    def __init__(self, base, weight=1.0, temp=0.2, thr=0.26):
        super().__init__()
        self.base, self.weight, self.temp = base, weight, temp
        self.thr_logit = torch.logit(torch.tensor(thr)).item()

    def forward(self, output, target):
        loss = self.base(output, target)
        out, lab = (output[0], target[0]) if isinstance(output, (list, tuple)) else (output, target)  # deep supervision
        with torch.autocast(out.device.type, enabled=False):  # fp16 would overflow the sums
            out = out.float()
            valid = (lab != IGNORE).float()
            surface_logit = out[:, 1:2] - out[:, 0:1]  # 2-class softmax == sigmoid of the logit difference
            p = ((surface_logit - self.thr_logit) / self.temp).sigmoid() * valid
            return loss + self.weight * crossing_loss(p, (lab == SURFACE).float())


class nnUNetTrainerCrossing(nnUNetTrainerFinetune):
    def _build_loss(self):
        return CrossingLoss(super()._build_loss())


class nnUNetTrainerCrossingLong(nnUNetTrainerCrossing):
    init_from = "nnUNetTrainerCrossing"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):  # explicit: nnU-Net records the init arguments by name
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
