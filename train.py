"""Step 2: train a 3D U-Net on random 128^3 patches. Loss = BCE + soft Dice, both ignoring label 2."""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import IGNORE, PATCH, SURFACE, WORK, UNet3D, dice, load, normalize, predict


class RandomPatches(Dataset):
    """Each item: random volume -> random 128^3 crop -> random flips and axis permutation."""

    def __init__(self, ids, n):
        self.ids, self.n = list(ids), n

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        rng = np.random.default_rng()
        vid = rng.choice(self.ids)
        img, lab = load(vid, "images"), load(vid, "labels")
        for _ in range(10):  # re-draw crops that are mostly "ignore"
            win = tuple(slice(o, o + PATCH) for o in (rng.integers(0, s - PATCH + 1) for s in img.shape))
            if (lab[win] != IGNORE).mean() > 0.5:
                break
        x, y = normalize(img[win].astype(np.float32)), np.array(lab[win])
        for ax in range(3):
            if rng.random() < 0.5:
                x, y = np.flip(x, ax), np.flip(y, ax)
        perm = rng.permutation(3)  # voxels are isotropic, so any axis order is a valid sample
        x, y = x.transpose(perm), y.transpose(perm)
        return torch.from_numpy(x.copy())[None], torch.from_numpy(y.copy())[None]


def crossings(x, dim):
    """Soft number of times each line of voxels along `dim` enters a sheet: the sum of the positive steps.
    For a binary line this is exactly the number of foreground runs (the line's Betti-0)."""
    start = torch.zeros_like(x.narrow(dim, 0, 1))  # outside the crop counts as background
    return torch.diff(x, dim=dim, prepend=start).clamp(min=0).sum(dim)


def crossing_loss(p, target):
    """Sheet-crossing loss: on every axis-aligned line, the prediction must cross as many sheets as the label.
    A hole in a sheet or a bridge that fuses two wraps removes a crossing, and a stray blob adds one.
    A sheet that is a bit too thick or too thin keeps the count, so only structural errors are punished.
    Returns |count error| summed over all lines of all 3 axes, divided by the label's total count."""
    err = sum((crossings(p, d) - crossings(target, d)).abs().sum() for d in (2, 3, 4))
    total = sum(crossings(target, d).sum() for d in (2, 3, 4))
    return err / (total + 1)


def loss_fn(logits, lab, topo=0.0, topo_temp=None, topo_thr=0.26):
    valid = (lab != IGNORE).float()
    target = (lab == SURFACE).float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    p = logits.sigmoid() * valid  # ignore voxels count as background in both p and target, so lines are cut alike
    soft_dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    loss = bce + soft_dice
    if topo:
        if topo_temp:  # count crossings as the thresholded mask will see them: a faint bridge at p > thr is a bridge
            thr_logit = torch.logit(torch.tensor(topo_thr))
            p = ((logits - thr_logit) / topo_temp).sigmoid() * valid
        loss = loss + topo * crossing_loss(p, target)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iters", type=int, default=250, help="iterations per epoch")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--topo", type=float, default=0.0, help="weight of the sheet-crossing loss (0 = off)")
    ap.add_argument("--topo-temp", type=float, help="sharpen the crossing count around --topo-thr (logit units, e.g. 0.2)")
    ap.add_argument("--topo-thr", type=float, default=0.26, help="threshold used at evaluation")
    ap.add_argument("--tag", default="", help="checkpoint suffix: best<tag>.pt, last<tag>.pt")
    args = ap.parse_args()

    split = pd.read_csv(WORK / "split.csv")
    train_ids = split[split.split == "train"].id
    val = split[split.split == "val"].id
    val_ids = val.sample(min(8, len(val)), random_state=0)  # small subset for monitoring; empty with --fold all

    loader = DataLoader(RandomPatches(train_ids, args.iters * args.batch), args.batch,
                        num_workers=args.workers, pin_memory=True, persistent_workers=True)
    model = UNet3D().cuda()
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * args.iters)

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, losses = time.time(), []
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(x)
            loss = loss_fn(logits.float(), y, args.topo, args.topo_temp, args.topo_thr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            opt.step()
            sched.step()
            losses.append(loss.item())
        torch.save(model.state_dict(), WORK / f"last{args.tag}.pt")
        msg = f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  {time.time() - t0:.0f}s"

        if len(val_ids) and (epoch % 5 == 4 or epoch == args.epochs - 1):
            model.eval()
            scores = []
            for vid in val_ids:
                lab = np.asarray(load(vid, "labels"))
                prob = predict(model, load(vid, "images"))
                scores.append(dice(prob > 0.5, lab == SURFACE, lab != IGNORE))
            val_dice = float(np.mean(scores))
            if val_dice > best:
                best = val_dice
                torch.save(model.state_dict(), WORK / f"best{args.tag}.pt")
            msg += f"  val_dice {val_dice:.4f}  best {best:.4f}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
