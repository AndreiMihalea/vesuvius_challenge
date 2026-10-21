"""Package the final model for Kaggle: work/kaggle/dataset (model + offline wheels) and work/kaggle/kernel (script).

The checkpoint keeps only the weights, and its trainer name is set to nnU-Net's default "nnUNetTrainer"
(our trainers only change the loss; the network is the plain ResEnc M from plans.json), so Kaggle needs no custom code.
Usage: uv run export_submission.py --user <kaggle username>
Then:  kaggle datasets create -p work/kaggle/dataset -r zip   (later: kaggle datasets version ... -m <msg>)
       kaggle kernels push -p work/kaggle/kernel"""
import argparse
import importlib.metadata as md
import json
import re
import shutil
import subprocess

import torch

from common import ROOT, WORK

RUN = WORK / "nnunet/results/Dataset100_VesuviusSurface/nnUNetTrainerCrossingLong__nnUNetResEncUNetMPlans__3d_fullres"
COMPETITION = "vesuvius-challenge-surface-detection"
SKIP = re.compile(r"torch|torchvision|triton|nvidia-.*|cuda-.*|argparse")  # Kaggle's own CUDA build of torch stays
PYTHONS = ["3.12", "3.13"]  # Kaggle ran Python 3.13 on 2026-10-20


def dependencies(root="nnunetv2"):
    """All packages nnU-Net needs (from this venv's metadata), with their installed versions."""
    seen, todo = {}, [root]
    while todo:
        name = todo.pop()
        key = name.lower().replace("_", "-")
        if key in seen or SKIP.fullmatch(key):
            continue
        try:
            seen[key] = md.version(name)
        except md.PackageNotFoundError:  # optional or platform-specific requirement
            continue
        todo += [re.split(r"[<>=!~; \[]", r)[0] for r in md.requires(name) or [] if "extra ==" not in r]
    return seen


def download_wheels(dest):
    """Wheels for every dependency: Kaggle installs only the ones it lacks. pip, not uv: it can fetch other platforms."""
    pip = ["uvx", "pip", "download", "-q", "--no-deps", "--only-binary=:all:", "-d", str(dest),
           "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_28_x86_64"]
    for name, version in sorted(dependencies().items()):
        ok = [subprocess.run(pip + ["--python-version", py, f"{name}=={version}"], capture_output=True).returncode == 0
              for py in PYTHONS]
        if not any(ok):  # source-only package (acvl-utils): build a wheel, since Kaggle unpacks uploaded .tar.gz
            subprocess.run(["uvx", "pip", "wheel", "-q", "--no-deps", "-w", str(dest), f"{name}=={version}"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username (owner of the dataset and the kernel)")
    ap.add_argument("--slug", default="vesuvius-crossing-nnunet", help="dataset name; the kernel is <slug>-submit")
    args = ap.parse_args()

    out = WORK / "kaggle"
    shutil.rmtree(out, ignore_errors=True)
    model, wheels, kernel = out / "dataset/model", out / "dataset/wheels", out / "kernel"
    (model / "fold_all").mkdir(parents=True)
    wheels.mkdir()
    kernel.mkdir()

    for f in ["plans.json", "dataset.json", "dataset_fingerprint.json"]:
        shutil.copy(RUN / f, model)
    ckpt = torch.load(RUN / "fold_all/checkpoint_final.pth", map_location="cpu", weights_only=False)
    keep = {k: ckpt[k] for k in ["network_weights", "init_args", "inference_allowed_mirroring_axes"]}
    torch.save({**keep, "trainer_name": "nnUNetTrainer"}, model / "fold_all/checkpoint_final.pth")

    download_wheels(wheels)

    meta = dict(title=args.slug, id=f"{args.user}/{args.slug}", licenses=[{"name": "CC0-1.0"}])
    (out / "dataset/dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    shutil.copy(ROOT / "kaggle_inference.py", kernel)
    name = f"{args.slug}-submit"  # a kernel cannot share the dataset's name (409 Conflict)
    meta = dict(id=f"{args.user}/{name}", title=name, code_file="kaggle_inference.py",
                language="python", kernel_type="script", is_private=True, enable_gpu=True, enable_internet=False,
                dataset_sources=[f"{args.user}/{args.slug}"], competition_sources=[COMPETITION])
    (kernel / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("\n".join(str(p.relative_to(out)) for p in sorted(out.rglob("*")) if p.is_file()))


if __name__ == "__main__":
    main()
