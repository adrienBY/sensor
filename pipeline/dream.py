#!/usr/bin/env python3
"""Make "dream" videos: the world model's rollout next to the real footage.

    python pipeline/dream.py data/processed runs/main/last.pt [--seconds 10] [--clips 3]

The model gets the real first frames of a validation take and the real camera motion, then
imagines every following frame on its own. Left = what really happened, right = the dream.
How long the right side stays recognisable is the main measure of progress.
"""
import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

import worldmodel as wm

TAESD = os.environ.get("WORLDMODEL_TAESD", "madebyollin/taesd")


def load_vae(device, dtype):
    from diffusers import AutoencoderTiny
    return AutoencoderTiny.from_pretrained(TAESD, torch_dtype=dtype).to(device).eval()


@torch.no_grad()
def decode(vae, z_norm, stats, dtype, chunk=32):
    out = []
    for i in range(0, len(z_norm), chunk):
        z = wm.denormalise(z_norm[i:i + chunk].float(), stats).to(dtype)
        img = vae.decode(z).sample.float().clamp(-1, 1)
        out.append(((img + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def rollout(model, take, start, horizon, device, steps=3, amp="fp16"):
    lat, act, ev = take["lat"], take["act"], take["ev"]
    ctx = torch.from_numpy(lat[start:start + wm.CONTEXT])[None].to(device)
    dreams = []
    for i in range(horizon):
        k = start + wm.CONTEXT - 1 + i
        a = torch.from_numpy(act[k])[None].to(device)
        e = torch.tensor([ev[k]], device=device)
        with wm.autocast(device, amp):
            z = wm.sample_next(model, ctx, a, e, steps=steps)
        dreams.append(z.float())
        ctx = torch.cat([ctx[:, 1:], z[:, None].to(ctx.dtype)], dim=1)
    return torch.cat(dreams)


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def make_dream_video(model, take, stats, vae, out_path: Path, fps, seconds=10, start=None,
                     device=torch.device("cpu"), dtype=torch.float32, steps=3, amp="fp16"):
    t_len = len(take["lat"])
    horizon = min(int(seconds * fps), t_len - wm.CONTEXT - 1)
    if horizon < 1:
        return None
    if start is None:
        start = random.randint(0, t_len - wm.CONTEXT - horizon)
    dream = rollout(model, take, start, horizon, device, steps, amp)
    real = torch.from_numpy(take["lat"][start + wm.CONTEXT:start + wm.CONTEXT + horizon]).to(device)
    dream_img = decode(vae, dream, stats, dtype)
    real_img = decode(vae, real, stats, dtype)

    h, w = real_img.shape[1:3]
    scale = 3
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (w * scale * 2 + 8, h * scale))
    for i in range(horizon):
        left = cv2.resize(real_img[i], (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(dream_img[i], (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
        label(left, "real")
        label(right, f"dream  {i / fps:4.1f} s")
        gap = np.zeros((h * scale, 8, 3), dtype=np.uint8)
        vw.write(cv2.cvtColor(np.hstack([left, gap, right]), cv2.COLOR_RGB2BGR))
    vw.release()

    metrics = {"take": take["name"], "start": int(start), "frames": int(horizon)}
    for sec in (1, 3, 10):
        k = int(sec * fps) - 1
        if k < horizon:
            metrics[f"psnr_{sec}s"] = round(psnr(real_img[k], dream_img[k]), 2)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3, help="denoising steps per frame")
    ap.add_argument("--amp", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, ck = wm.load_model(args.checkpoint, device)
    index = json.loads((args.processed / "index.json").read_text())
    data = wm.Dataset(args.processed, split="val", stats=ck["stats"])
    if len(data) == 0:
        data = wm.Dataset(args.processed, stats=ck["stats"])
    vae = load_vae(device, dtype)
    out = args.out or args.checkpoint.parent / "dreams"
    for c in range(args.clips):
        take = random.choice(data.takes)
        path = out / f"dream_step{ck['step']}_{c + 1}.mp4"
        m = make_dream_video(model, take, ck["stats"], vae, path, index["fps"], args.seconds,
                             device=device, dtype=dtype, steps=args.steps, amp=args.amp)
        print(f"{path.name}: {m}")


if __name__ == "__main__":
    main()
