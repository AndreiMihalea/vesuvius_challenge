"""3D viewer server. Run on the GPU box, open it from your laptop.
Usage: uv run viewer.py [--host 0.0.0.0] [--port 8000]
Safer remote access: keep --host 127.0.0.1 and `ssh -L 8000:localhost:8000 root@<server>`."""
import argparse
import re
import threading
from pathlib import Path

import numpy as np
import tifffile
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response

from common import DATA, NPY, WORK

HERE = Path(__file__).resolve().parent
ID_RE = re.compile(r"^\d+$")
PRED_DIR = WORK / "pred"
gpu_lock = threading.Lock()  # one inference at a time
models = {}

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=3)


def check_id(vid):
    if not ID_RE.match(vid):
        raise HTTPException(400, "bad id")


def read_volume(vid, kind):
    """uint8 (z, y, x). kind is 'images' or 'labels'. Prefers the .npy cache over the TIFF."""
    npy = NPY / kind / f"{vid}.npy"
    if npy.exists():
        return np.ascontiguousarray(np.load(npy, mmap_mode="r"))
    tif = DATA / f"train_{kind}" / f"{vid}.tif"
    if not tif.exists():
        raise HTTPException(404, f"no {kind} for {vid}")
    return tifffile.imread(tif)


def raw(a):
    return Response(np.ascontiguousarray(a, dtype=np.uint8).tobytes(), media_type="application/octet-stream")


@app.get("/")
def index():
    return FileResponse(HERE / "viewer.html")


@app.get("/api/volumes")
def volumes():
    ids = {p.stem for p in (NPY / "images").glob("*.npy")} | {p.stem for p in (DATA / "train_images").glob("*.tif")}
    return {
        "ids": sorted(ids),
        "ckpts": sorted(p.name for p in WORK.glob("*.pt")),
    }


@app.get("/api/volume/{vid}/meta")
def meta(vid: str):
    check_id(vid)
    return {"shape": list(read_volume(vid, "images").shape)}


@app.get("/api/volume/{vid}/image.bin")
def image(vid: str):
    check_id(vid)
    return raw(read_volume(vid, "images"))


@app.get("/api/volume/{vid}/label.bin")
def label(vid: str):
    check_id(vid)
    return raw(read_volume(vid, "labels"))


@app.get("/api/volume/{vid}/pred.bin")
def pred(vid: str, ckpt: str = "best.pt", thr: float = 0.5, min_size: int = 1000):
    check_id(vid)
    if ckpt not in {p.name for p in WORK.glob("*.pt")}:
        raise HTTPException(404, "unknown checkpoint")
    key = PRED_DIR / f"{Path(ckpt).stem}_{vid}.npy"  # cached probabilities as float16
    if not key.exists():
        import torch
        if not torch.cuda.is_available():
            raise HTTPException(503, "no CUDA on server, cannot run the model")
        from common import UNet3D, predict
        with gpu_lock:
            if not key.exists():
                if ckpt not in models:
                    m = UNet3D().cuda().eval()
                    m.load_state_dict(torch.load(WORK / ckpt, map_location="cuda"))
                    models[ckpt] = m
                prob = predict(models[ckpt], read_volume(vid, "images"))
                PRED_DIR.mkdir(parents=True, exist_ok=True)
                np.save(key, prob.astype(np.float16))
    from evaluate import postprocess
    prob = np.load(key).astype(np.float32)
    return raw(postprocess(prob, thr, min_size))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
