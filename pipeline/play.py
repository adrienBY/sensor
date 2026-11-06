#!/usr/bin/env python3
"""Walk around inside the dream with the keyboard.

    python pipeline/play.py data/processed runs/main/last.pt

Starts from real frames of one of your takes, then every frame comes from the world model.
  W / S   walk forward / back        A / D   step left / right
  J / L   turn left / right          I / K   look up / down
  1-9     trigger an interaction tag (listed at start)
  R       restart from a new real moment      Q or Esc   quit
Walking (W/S/A/D) only means something if the model was trained on AR-mode takes.
"""
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import worldmodel as wm
from dream import decode, load_vae

STEP_M = 0.035     # metres per frame when walking (~0.5 m/s at 15 fps)
TURN = 0.045       # radians per frame
LOOK = 0.03


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--steps", type=int, default=3, help="denoising steps (fewer = faster)")
    ap.add_argument("--scale", type=int, default=4, help="window size multiplier")
    ap.add_argument("--amp", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, ck = wm.load_model(args.checkpoint, device)
    data = wm.Dataset(args.processed, stats=ck["stats"])
    vae = load_vae(device, dtype)
    vocab = ck.get("vocab", [])
    print(__doc__.split("\n\n", 2)[2])
    for i, v in enumerate(vocab[:9]):
        print(f"  {i + 1}: {v}")

    def reset():
        t = random.choice(data.takes)
        s = random.randint(0, len(t["lat"]) - wm.CONTEXT - 1)
        return torch.from_numpy(t["lat"][s:s + wm.CONTEXT])[None].to(device)

    ctx = reset()
    win = "Dream (Q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    key = -1
    fps_t, frames = time.time(), 0
    fps_text = ""
    while True:
        rot = np.zeros(3, np.float32)
        trans = np.zeros(3, np.float32)
        event = 0
        k = chr(key).lower() if 0 <= key < 256 else ""
        if k == "w": trans[2] = -STEP_M
        if k == "s": trans[2] = STEP_M
        if k == "a": trans[0] = -STEP_M
        if k == "d": trans[0] = STEP_M
        if k == "j": rot[1] = TURN
        if k == "l": rot[1] = -TURN
        if k == "i": rot[0] = LOOK
        if k == "k": rot[0] = -LOOK
        if k.isdigit() and k != "0" and int(k) <= len(vocab):
            event = int(k)
        if k == "r":
            ctx = reset()
        if k == "q" or key == 27:
            break
        has_trans = 1.0 if np.any(trans) else 0.0
        a = torch.from_numpy(wm.action_vector(rot, trans, has_trans))[None].to(device)
        e = torch.tensor([event], device=device)
        with wm.autocast(device, args.amp):
            z = wm.sample_next(model, ctx, a, e, steps=args.steps)
        ctx = torch.cat([ctx[:, 1:], z[:, None].float()], dim=1)
        img = decode(vae, z, ck["stats"], dtype)[0]
        h, w = img.shape[:2]
        big = cv2.resize(img, (w * args.scale, h * args.scale), interpolation=cv2.INTER_LINEAR)
        frames += 1
        if time.time() - fps_t >= 1:
            fps_text = f"{frames / (time.time() - fps_t):.1f} fps"
            fps_t, frames = time.time(), 0
        bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, fps_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(win, bgr)
        key = cv2.waitKey(1)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
