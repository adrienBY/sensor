#!/usr/bin/env python3
"""Train the world model on encoded takes. Resumes automatically.

    python pipeline/train.py data/processed runs/main [--minutes 60] [--batch 64] [--amp fp16]

Trains for a time budget, saves runs/main/last.pt every few minutes and at the end (Ctrl+C
also saves), and writes dream videos to runs/main/dreams/ so you can watch progress.
Running it again continues from last.pt, including on newly added footage.

Tuned for AMD RDNA 4 (RX 9070) following AMD's ROCm guidance:
  * FP16 mixed precision with loss scaling (the precision AMD validates on Radeon);
  * the dataset lives on the GPU and batches are built there: no CPU work or copies per step;
  * no GPU-to-CPU syncs inside the loop (the loss is read every 30 s, not every step);
  * fused AdamW and a fused EMA update (fewer, larger kernels);
  * model sizes are multiples of 64 so matrix multiplies map onto the WMMA units;
  * TunableOp (AMD's GEMM auto-tuner) is switched on through environment variables by the
    launcher when bench.py finds it faster on this GPU.

New models (version 2) also get:
  * memory: frames of the take that saw the same place, retrieved by camera pose;
  * an 8-frame context instead of 4;
  * self-forcing rollouts: after a warm-up, part of each batch continues from frames the model
    generated itself, so it learns to correct its own blur instead of compounding it.
Older checkpoints keep training with the architecture they were made with.
"""
import argparse
import copy
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import torch

import worldmodel as wm


def make_optimizer(model, lr):
    kw = dict(lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    if next(model.parameters()).is_cuda:
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kw), "fused"
        except (RuntimeError, TypeError, ValueError):
            pass
    return torch.optim.AdamW(model.parameters(), foreach=True, **kw), "foreach"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate at batch 64")
    ap.add_argument("--amp", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--compile", action="store_true",
                    help="try torch.compile (Linux/WSL only; Triton is not on ROCm for Windows)")
    ap.add_argument("--save-every", type=float, default=10, help="minutes between checkpoints")
    ap.add_argument("--dream-every", type=float, default=20, help="minutes between dream videos")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-dreams", action="store_true", help="skip dream videos (testing)")
    ap.add_argument("--context", type=int, default=wm.NEW_CONTEXT, help="past frames (new models)")
    ap.add_argument("--memory", type=int, default=wm.NEW_MEM_K,
                    help="memory frames per prediction (new models; 0 = no memory)")
    ap.add_argument("--rollout-start", type=int, default=5000,
                    help="step at which self-forcing rollouts begin")
    ap.add_argument("--rollout-full", type=int, default=20000,
                    help="step at which rollouts reach their full share (half of the steps)")
    ap.add_argument("--no-rollout", action="store_true")
    ap.add_argument("--dream-steps", type=int, default=3, help="denoising steps in dream videos")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(wm.gpu_report())
    if device.type == "cpu":
        print("WARNING: no GPU found, training on CPU will be very slow.")
    tunable = os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1"
    print(f"precision {args.amp} · batch {args.batch} · TunableOp {'on' if tunable else 'off'}")

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
        model = wm.build_model(ck["config"]).to(device)
        model.load_state_dict(wm.upgrade_state_dict(ck["model"]))
        if tuple(model.config["latent_shape"]) != tuple(train.latent_shape):
            raise SystemExit("Latent size changed since the last run; start a new run folder.")
    else:
        if args.dim % 64:
            raise SystemExit("--dim must be a multiple of 64 (WMMA-friendly matrix sizes)")
        model = wm.WorldDiT(train.latent_shape, dim=args.dim, depth=args.depth,
                            heads=args.dim // 64, context=args.context,
                            mem_k=args.memory).to(device)
    rollout = 0 if args.no_rollout else wm.ROLLOUT
    t_dev = time.time()
    train.to_device(device, model.context, rollout, model.mem_k)
    print(f"context {model.context} frames · memory {model.mem_k} frames per prediction · "
          f"self-forcing rollouts {'off' if not rollout else f'from step {args.rollout_start}'}"
          f" · data on GPU in {time.time() - t_dev:.1f} s")
    ema = copy.deepcopy(model).eval()
    if ck:
        ema.load_state_dict(wm.upgrade_state_dict(ck["ema"]))
    for p in ema.parameters():
        p.requires_grad_(False)

    # Larger batches on a big GPU: scale the learning rate with sqrt(batch / 64).
    lr = args.lr * math.sqrt(args.batch / 64)
    opt, opt_kind = make_optimizer(model, lr)
    if ck and ck.get("opt"):
        try:
            opt.load_state_dict(ck["opt"])
            for g in opt.param_groups:
                g["lr"] = lr
            for p, st in opt.state.items():           # moments of a reshaped (upgraded) weight
                for k, v in st.items():
                    if torch.is_tensor(v) and v.dim() and v.shape != p.shape and v.numel() == p.numel():
                        st[k] = v.reshape(p.shape)
        except ValueError:
            print("Optimizer state does not match (settings changed); starting it fresh.")
    scaler = torch.amp.GradScaler(device.type, enabled=(args.amp == "fp16" and device.type == "cuda"))
    step = ck["step"] if ck else 0
    minutes_before = ck.get("minutes_trained", 0.0) if ck else 0.0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M parameters, optimizer {opt_kind}, lr {lr:.2e}, "
          f"starting at step {step} ({minutes_before:.0f} min trained before)")

    fwd_model = model
    if args.compile:
        if os.name == "nt":
            print("torch.compile skipped: Triton is not available for ROCm on Windows.")
        else:
            try:
                fwd_model = torch.compile(model)
                print("torch.compile enabled")
            except Exception as e:  # keep training even if the compiler is unavailable
                print(f"torch.compile unavailable ({e}); running eagerly")

    vae = None
    if not args.no_dreams:
        from dream import load_vae, make_dream_video
        vae = load_vae(device, torch.float16 if device.type == "cuda" else torch.float32)

    stop = {"now": False}

    def on_sigint(*_):
        print("\nStopping after this step, saving…")
        stop["now"] = True
    signal.signal(signal.SIGINT, on_sigint)

    ema_params = list(ema.parameters())
    model_params = list(model.parameters())
    t0 = time.time()
    last_save = last_dream = t0
    log_t = t0
    log_loss = torch.zeros((), device=device)   # summed on the GPU, read every 30 s
    log_n = 0
    metrics_path = args.run / "metrics.jsonl"

    def minutes_total():
        return minutes_before + (time.time() - t0) / 60

    def save():
        wm.save_checkpoint(ckpt_path, model, ema, opt, step, stats, index.get("event_vocab", []),
                           minutes_total())
        if tunable:                       # keep TunableOp's results even if the window is closed
            try:
                torch.cuda.tunable.write_file()
            except Exception:
                pass

    def dream():
        if vae is None:
            return
        take = val.takes[step % len(val.takes)]
        path = args.run / "dreams" / f"step{step:07d}.mp4"
        m = make_dream_video(ema, take, stats, vae, path, fps, 10, device=device,
                             dtype=torch.float16 if device.type == "cuda" else torch.float32,
                             steps=args.dream_steps, amp=args.amp)
        if m:
            m.update({"step": step, "minutes": round(minutes_total(), 1)})
            with open(metrics_path, "a") as f:
                f.write(json.dumps(m) + "\n")
            print(f"  dream video {path.name}: " +
                  ", ".join(f"{k} {v}" for k, v in m.items() if k.startswith("psnr")))

    while not stop["now"] and time.time() - t0 < args.minutes * 60:
        if step < 1000:                                   # warm-up
            for g in opt.param_groups:
                g["lr"] = lr * (step + 1) / 1000
        b = train.gpu_batch(args.batch)
        C, R = model.context, rollout
        win, acts, evs = b["win"], b["act"], b["ev"]
        mem, mem_pose = b.get("mem"), b.get("mem_pose")
        ctx, tgt = win[:, R:R + C], win[:, R + C]
        act, ev = acts[:, R + C - 1], evs[:, R + C - 1]
        # Self-forcing: a quarter of the batch replaces its last k context frames with frames
        # the model generates itself (2 quick denoising steps each, no gradient), so it learns
        # to predict the real next frame from its own imperfect output.
        p_roll = 0.0 if not R else min(0.5, max(0.0, 0.5 * (step - args.rollout_start)
                                                / max(1, args.rollout_full - args.rollout_start)))
        if p_roll > 0 and random.random() < p_roll:
            nb, k = max(1, args.batch // 4), random.randint(1, R)
            sub = (lambda x: None if x is None else x[:nb])
            with wm.autocast(device, args.amp):
                c = win[:nb, R - k:R - k + C]
                for j in range(k):
                    m = R - k + C + j                        # frame being generated
                    z = wm.sample_next(model, c, acts[:nb, m - 1], evs[:nb, m - 1], steps=2,
                                       mem=sub(mem), mem_pose=sub(mem_pose))
                    c = torch.cat([c[:, 1:], z[:, None].float()], dim=1)
            ctx = torch.cat([c, ctx[nb:]], dim=0)
        with wm.autocast(device, args.amp):
            loss = wm.training_loss(fwd_model, ctx, tgt, act, ev, mem, mem_pose)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model_params, 1.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, model_params, 1 - 0.999)
        step += 1
        log_loss += loss.detach()
        log_n += 1

        now = time.time()
        if now - log_t >= 30:
            # Read the loss (this waits for the GPU to finish the queued steps) BEFORE taking
            # the time, so each window is charged for its own GPU work, not the previous one's.
            loss_avg = log_loss.item() / log_n
            now = time.time()
            print(f"step {step:7d}  loss {loss_avg:.4f}  "
                  f"{log_n / (now - log_t):6.1f} steps/s  "
                  f"{log_n * args.batch / (now - log_t):7.0f} samples/s  "
                  f"{(args.minutes * 60 - (now - t0)) / 60:5.1f} min left", flush=True)
            log_loss.zero_()
            log_n, log_t = 0, now
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
