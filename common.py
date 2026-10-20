"""Shared pieces: paths, 3D U-Net, sliding-window inference."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WORK = ROOT / "work"
NPY = WORK / "npy"

PATCH = 128         # training / inference window size (voxels per side)
SURFACE, IGNORE = 1, 2


def load(vid, kind):
    """Memory-mapped uint8 volume (z, y, x). kind is 'images' or 'labels'."""
    return np.load(NPY / kind / f"{vid}.npy", mmap_mode="r")


def normalize(x):
    """Z-score one patch. Works for numpy arrays and torch tensors."""
    return (x - x.mean()) / (x.std() + 1e-6)


def block(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01, True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01, True),
    )


class UNet3D(nn.Module):
    """Plain nnU-Net-style 3D U-Net: 5 resolution levels, skip connections, 1 output logit (surface)."""

    def __init__(self, ch=(32, 64, 128, 256, 320)):
        super().__init__()
        self.enc = nn.ModuleList(block(ch[i - 1] if i else 1, ch[i]) for i in range(len(ch)))
        self.up = nn.ModuleList(nn.ConvTranspose3d(ch[i + 1], ch[i], 2, stride=2) for i in range(len(ch) - 1))
        self.dec = nn.ModuleList(block(2 * ch[i], ch[i]) for i in range(len(ch) - 1))
        self.head = nn.Conv3d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(F.max_pool3d(x, 2) if i else x)
            skips.append(x)
        x = skips.pop()
        for i in reversed(range(len(self.dec))):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.head(x)


@torch.no_grad()
def predict(model, vol, step=PATCH // 2):
    """Slide a 128^3 window with 50% overlap and average the surface probabilities."""
    vol = torch.from_numpy(np.asarray(vol, dtype=np.float32)).cuda()
    prob, count = torch.zeros_like(vol), torch.zeros_like(vol)
    starts = [sorted({*range(0, s - PATCH, step), s - PATCH}) for s in vol.shape]
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                win = np.s_[z:z + PATCH, y:y + PATCH, x:x + PATCH]
                with torch.autocast("cuda", torch.bfloat16):
                    logit = model(normalize(vol[win])[None, None])
                prob[win] += logit.float().sigmoid()[0, 0]
                count[win] += 1
    return (prob / count).cpu().numpy()


def dice(pred, gt, valid):
    p, g = pred & valid, gt & valid
    return 2 * (p & g).sum() / (p.sum() + g.sum() + 1e-9)
