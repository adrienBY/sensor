#!/usr/bin/env python3
"""Encode processed frames into latents with a frozen autoencoder (GPU step).

    python pipeline/encode_latents.py data/processed [--batch 64] [--device cuda]

Runs after preprocess.py. For every take folder that has frames.npy and no latents.npy yet:
  latents.npy  float16 [T, 4, H/8, W/8]   (256x144 frames -> 4 x 18 x 32)
  preview.png  first frames next to their decoded latents, to check the encoder visually

The autoencoder is TAESD ("madebyollin/taesd"), a tiny distilled version of Stable Diffusion's
VAE: it is fast enough to encode hours of footage on one GPU and is never trained here.
Training reads latents.npy instead of video, which is the biggest speed-up of the whole project.

Needs: torch (ROCm build on the RX 9070; on CUDA or CPU it also works), diffusers, numpy,
opencv-python. "cuda" is the device name PyTorch uses for AMD GPUs too.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderTiny

# WORLDMODEL_TAESD can point at a local copy of the model (offline use, tests).
MODEL_ID = os.environ.get("WORLDMODEL_TAESD", "madebyollin/taesd")


@torch.no_grad()
def encode_take(vae, take: Path, device, dtype, batch: int, force: bool):
    out = take / "latents.npy"
    if out.exists() and not force:
        print("  latents already there (use --force to redo)")
        return
    frames = np.load(take / "frames.npy", mmap_mode="r")   # [T, H, W, 3] uint8
    t = len(frames)
    h, w = frames.shape[1], frames.shape[2]
    if h % 8 or w % 8:
        raise ValueError(f"frame size {w}x{h} must be divisible by 8")
    latents = np.zeros((t, 4, h // 8, w // 8), dtype=np.float16)
    for i in range(0, t, batch):
        x = torch.from_numpy(np.array(frames[i:i + batch])).to(device)   # copy out of the memory map
        x = x.permute(0, 3, 1, 2).to(dtype) / 127.5 - 1.0     # [-1, 1], as diffusers expects
        latents[i:i + len(x)] = vae.encode(x).latents.float().cpu().numpy().astype(np.float16)
    np.save(out, latents)

    # Visual check: originals on top, round-tripped through the autoencoder below.
    n = min(6, t)
    pick = np.linspace(0, t - 1, n).astype(int)
    z = torch.from_numpy(latents[pick].astype(np.float32)).to(device, dtype)
    rec = vae.decode(z).sample.float().clamp(-1, 1)
    rec = ((rec + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    top = np.hstack([frames[k] for k in pick])
    bottom = np.hstack(list(rec))
    cv2.imwrite(str(take / "preview.png"), cv2.cvtColor(np.vstack([top, bottom]), cv2.COLOR_RGB2BGR))

    bad = int((~np.isfinite(latents)).sum())
    if bad:
        print(f"  WARNING: {bad} latent values are not finite (encoder overflow); "
              "this take may train badly")
    # float64 accumulation: summing millions of squares in float16 overflows to inf
    stats = {"latent_mean": float(latents.mean(dtype=np.float64)),
             "latent_std": float(latents.std(dtype=np.float64)),
             "latent_shape": list(latents.shape[1:]), "encoder": MODEL_ID}
    info_path = take / "take.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info["latents"] = stats
    info_path.write_text(json.dumps(info, indent=2))
    print(f"  {t} frames -> latents {list(latents.shape)}, std {stats['latent_std']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed", type=Path, help="output folder of preprocess.py")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Encoder {MODEL_ID} on {name}")
    vae = AutoencoderTiny.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()

    takes = sorted(p.parent for p in args.processed.glob("*/frames.npy"))
    if not takes:
        raise SystemExit(f"No processed takes in {args.processed}; run preprocess.py first")
    for take in takes:
        print(f"== {take.name}")
        encode_take(vae, take, device, dtype, args.batch, args.force)
    print("\nDone. Open a few preview.png files: the bottom row should look like the top row.")


if __name__ == "__main__":
    main()
