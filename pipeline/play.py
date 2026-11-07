#!/usr/bin/env python3
"""Walk around inside the dream with the keyboard.

    python pipeline/play.py data/processed runs/main/last.pt

Starts from real frames of one of your takes, then every frame comes from the world model.
  W / S   walk forward / back        A / D   step left / right
  J / L   turn left / right          I / K   look up / down
  1-9     trigger an interaction tag (listed at start)
  R       restart from a new real moment      Q or Esc   quit
Walking (W/S/A/D) only means something if the model was trained on AR-mode takes.

It runs with the same settings as the dream videos: the same model copy (EMA), denoising
steps, context noise and memory, one frame per 1/15 s (the footage's frame rate), and walking
and turning speeds measured from your own footage, so each key press is a movement the model
has actually seen. On Windows several keys can be held at once (walk and turn together).
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

import json
import os


def footage_speeds(data):
    """Typical per-frame walking, turning and looking speeds while you were actually moving,
    measured from the takes, so key presses match the motion the model learned from."""
    act = np.concatenate([t["act"] for t in data.takes])
    rot, trans, has = act[:, :3] / wm.ROT_SCALE, act[:, 3:6] / wm.TRANS_SCALE, act[:, 6] > 0
    def typical(x, lo, hi, pct, default):
        x = np.abs(x)
        x = x[x > lo]                       # only frames where you were deliberately moving
        return float(np.clip(np.percentile(x, pct), lo, hi)) if len(x) > 50 else default
    step = typical(np.linalg.norm(trans[has], axis=1), 0.01, 0.08, 60, 0.035)
    turn = typical(rot[:, 1], 0.015, 0.12, 80, 0.045)
    look = typical(rot[:, 0], 0.01, 0.08, 80, 0.03)
    return step, turn, look


HOLD_KEYS = "WASDJLIK"


def held_keys():
    """Letters held down right now (Windows), so keys combine and movement is continuous like
    real footage instead of stuttering with keyboard auto-repeat. None elsewhere."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        get = ctypes.windll.user32.GetAsyncKeyState
        return {k.lower() for k in HOLD_KEYS if get(ord(k)) & 0x8000}
    except Exception:
        return None


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

    c = model.context
    fps = json.loads((args.processed / "index.json").read_text()).get("fps", 15)
    step_m, turn, look = footage_speeds(data)
    print(f"Matching the dream videos: {args.steps} denoising steps, {fps} frames per second, "
          f"walk {step_m * fps:.2f} m/s, turn {np.degrees(turn) * fps:.0f} deg/s, "
          f"look {np.degrees(look) * fps:.0f} deg/s (from your footage)")
    state = {}

    def reset():
        """Start from a real moment: its frames, and the camera pose of its last frame. Memory
        then comes from that take's real frames, found by where the dream camera is."""
        t = random.choice(data.takes)
        s = random.randint(0, len(t["lat"]) - c - 1)
        state.update(take=t, R=t["R"][s + c - 1].copy(), p=t["p"][s + c - 1].copy())
        return torch.from_numpy(t["lat"][s:s + c])[None].to(device)

    ctx = reset()
    if model.mem_k:
        print(f"Memory: {model.mem_k} frames of the current take, picked by where you look.")
    win = "Dream (Q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    key = -1
    fps_t, frames = time.time(), 0
    fps_text = ""
    frame_t = time.time()
    while True:
        rot = np.zeros(3, np.float32)
        trans = np.zeros(3, np.float32)
        event = 0
        k = chr(key).lower() if 0 <= key < 256 else ""
        held = held_keys()
        down = held if held is not None else {k}
        if "w" in down: trans[2] -= step_m
        if "s" in down: trans[2] += step_m
        if "a" in down: trans[0] -= step_m
        if "d" in down: trans[0] += step_m
        if "j" in down: rot[1] += turn
        if "l" in down: rot[1] -= turn
        if "i" in down: rot[0] += look
        if "k" in down: rot[0] -= look
        if k.isdigit() and k != "0" and int(k) <= len(vocab):
            event = int(k)
        if k == "r":
            ctx = reset()
        if k == "q" or key == 27:
            break
        has_trans = 1.0 if np.any(trans) else 0.0
        a = torch.from_numpy(wm.action_vector(rot, trans, has_trans))[None].to(device)
        e = torch.tensor([event], device=device)
        mem = mem_pose = None
        state["R"], state["p"] = wm.apply_action(state["R"], state["p"], rot, trans * has_trans)
        if model.mem_k:
            mem, mem_pose = wm.memory_for(state["take"], state["R"], state["p"], model.mem_k,
                                          device)
        with wm.autocast(device, args.amp):
            z = wm.sample_next(model, ctx, a, e, steps=args.steps, mem=mem, mem_pose=mem_pose)
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
        # one frame per 1/fps like the footage; if the GPU is slower, the dream just runs slower
        wait_ms = max(1, int((frame_t + 1 / fps - time.time()) * 1000))
        key = cv2.waitKey(wait_ms)
        frame_t = max(time.time(), frame_t + 1 / fps) if wait_ms > 1 else time.time()
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
