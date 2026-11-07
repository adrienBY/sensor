#!/usr/bin/env python3
"""Make "dream" videos: the world model's rollout next to the real footage.

    python pipeline/dream.py data/processed runs/main/last.pt [--seconds 10] [--clips 3]

The model gets the real first frames of a validation take and the real camera motion, then
imagines every following frame on its own. Left = what really happened, right = the dream.
How long the right side stays recognisable is the main measure of progress.
With a memory model, memory comes from the same take but never from the clip being predicted
(nor 2 s around it), so the dream cannot copy the real future frames.
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
def find_revisit_start(take, context, horizon):
    """A clip where the camera turns away (more than 70 degrees) and comes back to within
    25 degrees of where it started: the test of whether the dream remembers.
    Returns (start, score) or None."""
    f = -take["R"][:, :, 2]
    n = len(f)
    best = None
    for s in range(0, n - context - horizon - 1, 5):
        a, b = s + context, s + context + horizon - 1
        back = float(f[a] @ f[b])
        away = float((f[a + 1:b] @ f[a]).min())
        if back > 0.9 and away < 0.35:
            score = back - away
            if best is None or score > best[1]:
                best = (s, score)
    return best


def rollout(model, take, start, horizon, device, steps=3, amp="fp16", guidance=1.0):
    lat, act, ev = take["lat"], take["act"], take["ev"]
    c = model.context
    ctx = torch.from_numpy(lat[start:start + c])[None].to(device)
    exclude = (start - wm.MEM_GAP, start + c + horizon + wm.MEM_GAP)
    dreams = []
    for i in range(horizon):
        k = start + c - 1 + i
        a = torch.from_numpy(act[k])[None].to(device)
        e = torch.tensor([ev[k]], device=device)
        mem = mem_pose = None
        if model.mem_k:                      # the real camera pose of the frame being dreamt
            mem, mem_pose = wm.memory_for(take, take["R"][k + 1], take["p"][k + 1],
                                          model.mem_k, device, exclude)
        with wm.autocast(device, amp):
            z = wm.sample_next(model, ctx, a, e, steps=steps, mem=mem, mem_pose=mem_pose,
                               guidance=guidance)
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
                     device=torch.device("cpu"), dtype=torch.float32, steps=3, amp="fp16",
                     guidance=1.0):
    t_len = len(take["lat"])
    c = model.context
    horizon = min(int(seconds * fps), t_len - c - 1)
    if horizon < 1:
        return None
    if start is None:
        start = random.randint(0, t_len - c - horizon)
    dream = rollout(model, take, start, horizon, device, steps, amp, guidance)
    real = torch.from_numpy(take["lat"][start + c:start + c + horizon]).to(device)
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
    ap.add_argument("--steps", default="3",
                    help="denoising steps per frame; a list like 3,10 renders each clip once "
                         "per value from the same starting point, to compare")
    ap.add_argument("--amp", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="action guidance (1 = off, 1.5-2 follows the camera moves more firmly)")
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
    steps_list = [int(x) for x in str(args.steps).split(",") if x.strip()]
    fps = index["fps"]
    for c in range(args.clips):
        take = random.choice(data.takes)
        horizon = min(int(args.seconds * fps), len(take["lat"]) - model.context - 1)
        if horizon < 1:
            continue
        start = random.randint(0, len(take["lat"]) - model.context - horizon)
        for steps in steps_list:
            tag = f"_{steps}steps" if len(steps_list) > 1 else ""
            path = out / f"dream_step{ck['step']}_{c + 1}{tag}.mp4"
            m = make_dream_video(model, take, ck["stats"], vae, path, fps, args.seconds,
                                 start=start, device=device, dtype=dtype, steps=steps,
                                 amp=args.amp, guidance=args.guidance)
            print(f"{path.name}: {m}")


if __name__ == "__main__":
    main()
