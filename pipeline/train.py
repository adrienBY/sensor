#!/usr/bin/env python3
"""Train the world model on encoded takes. Resumes automatically.

    python pipeline/train.py data/processed runs/main [--minutes 60]

Trains for a time budget, saves runs/main/last.pt every few minutes and at the end (Ctrl+C
also saves), and writes dream videos to runs/main/dreams/ so you can watch progress.
Running it again continues from last.pt, including on newly added footage.
"""
import argparse
import copy
import json
import signal
import time
from pathlib import Path

import torch

import worldmodel as wm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--save-every", type=float, default=10, help="minutes between checkpoints")
    ap.add_argument("--dream-every", type=float, default=20, help="minutes between dream videos")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-dreams", action="store_true", help="skip dream videos (testing)")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        print("WARNING: no GPU found, training on CPU will be very slow.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    args.run.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.run / "last.pt"
    index = json.loads((args.processed / "index.json").read_text())
    fps = index["fps"]

    ck = torch.load(ckpt_path, map_location=device, weights_only=False) if ckpt_path.exists() else None
    stats = ck["stats"] if ck else None
    train = wm.Dataset(args.processed, split="train", stats=stats)
    if len(train) == 0:
        train = wm.Dataset(args.processed, stats=stats)          # very first takes: no split yet
    if len(train) == 0:
        raise SystemExit("No encoded takes found. Run preprocess.py and encode_latents.py first.")
    stats = train.stats
    val = wm.Dataset(args.processed, split="val", stats=stats)
    if len(val) == 0:
        val = train
    print(f"Training on {len(train)} takes, {train.frames()} frames "
          f"({train.frames() / fps / 60:.1f} min); dreams from {len(val)} takes")

    if ck:
        cfg = dict(ck["config"])
        model = wm.WorldDiT(tuple(cfg.pop("latent_shape")), **cfg).to(device)
        model.load_state_dict(ck["model"])
        if tuple(model.config["latent_shape"]) != tuple(train.latent_shape):
            raise SystemExit("Latent size changed since the last run; start a new run folder.")
    else:
        model = wm.WorldDiT(train.latent_shape, dim=args.dim, depth=args.depth).to(device)
    ema = copy.deepcopy(model).eval()
    if ck:
        ema.load_state_dict(ck["ema"])
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    if ck and ck.get("opt"):
        opt.load_state_dict(ck["opt"])
    step = ck["step"] if ck else 0
    minutes_before = ck.get("minutes_trained", 0.0) if ck else 0.0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M parameters, starting at step {step} "
          f"({minutes_before:.0f} min trained before)")

    vae = None
    if not args.no_dreams:
        from dream import load_vae, make_dream_video
        vae = load_vae(device, torch.float16 if device.type == "cuda" else torch.float32)

    stop = {"now": False}

    def on_sigint(*_):
        print("\nStopping after this step, saving…")
        stop["now"] = True
    signal.signal(signal.SIGINT, on_sigint)

    use_amp = device.type == "cuda"
    t0 = time.time()
    last_save = last_dream = t0
    log_loss, log_n, log_t = 0.0, 0, t0
    metrics_path = args.run / "metrics.jsonl"

    def minutes_total():
        return minutes_before + (time.time() - t0) / 60

    def save():
        wm.save_checkpoint(ckpt_path, model, ema, opt, step, stats, index.get("event_vocab", []),
                           minutes_total())

    def dream():
        if vae is None:
            return
        take = val.takes[step % len(val.takes)]
        path = args.run / "dreams" / f"step{step:07d}.mp4"
        m = make_dream_video(ema, take, stats, vae, path, fps, 10, device=device,
                             dtype=torch.float16 if use_amp else torch.float32)
        if m:
            m.update({"step": step, "minutes": round(minutes_total(), 1)})
            with open(metrics_path, "a") as f:
                f.write(json.dumps(m) + "\n")
            print(f"  dream video {path.name}: " +
                  ", ".join(f"{k} {v}" for k, v in m.items() if k.startswith("psnr")))

    while not stop["now"] and time.time() - t0 < args.minutes * 60:
        warm = min(1.0, (step + 1) / 1000)
        for g in opt.param_groups:
            g["lr"] = args.lr * warm
        ctx, tgt, act, ev = train.batch(args.batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = wm.training_loss(model, ctx, tgt, act, ev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - 0.999)
        step += 1
        log_loss += loss.item()
        log_n += 1

        now = time.time()
        if now - log_t >= 30:
            print(f"step {step:7d}  loss {log_loss / log_n:.4f}  "
                  f"{log_n / (now - log_t):5.1f} steps/s  "
                  f"{(args.minutes * 60 - (now - t0)) / 60:5.1f} min left", flush=True)
            log_loss, log_n, log_t = 0.0, 0, now
        if now - last_save >= args.save_every * 60:
            save()
            last_save = now
        if now - last_dream >= args.dream_every * 60:
            dream()
            last_dream = time.time()

    save()
    print(f"Saved {ckpt_path} at step {step} ({minutes_total():.0f} min trained in total).")
    dream()


if __name__ == "__main__":
    main()
