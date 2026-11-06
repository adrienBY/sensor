#!/usr/bin/env python3
"""Find the fastest training settings on this GPU: precision, batch size and TunableOp.

    python pipeline/bench.py [--seconds 10] [--out bench.json] [--cache-dir .]

Runs short synthetic training bursts with the real model and latent size, each in a fresh
process (TunableOp is configured through environment variables at start-up):
  1. FP16 and BF16 at batch 64, 128 and 256, TunableOp off;
  2. the best of those again with TunableOp on (AMD's GEMM auto-tuner). It tunes every matrix
     shape on first use, which happens during the warm-up and is not measured; the results
     are saved in --cache-dir (tunableop_results0.csv) and reused by training.
Writes the winner to --out as JSON, e.g. {"amp": "fp16", "batch": 128, "tunableop": true}.
FP16 is the precision AMD validates on Radeon, so BF16 has to be clearly faster (>5%) to win.
A bigger batch has to be clearly faster (>10%), since on a small dataset smaller batches
learn more per sample.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LATENT = (4, 18, 32)                     # 256x144 frames through TAESD
TUNABLE_FILE = "tunableop_results.csv"   # PyTorch inserts the GPU index: ..._results0.csv


def tunable_env(enabled, cache_dir):
    if not enabled:
        return {"PYTORCH_TUNABLEOP_ENABLED": "0"}
    return {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1",
            "PYTORCH_TUNABLEOP_FILENAME": str(Path(cache_dir) / TUNABLE_FILE)}


def worker(amp, batch, seconds, warmup):
    import torch
    import worldmodel as wm
    from train import make_optimizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = {"gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
            "report": wm.gpu_report()}

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    try:
        torch.manual_seed(0)
        model = wm.WorldDiT(LATENT).to(dev)
        opt, opt_kind = make_optimizer(model, 1e-4)
        scaler = torch.amp.GradScaler(dev.type, enabled=(amp == "fp16" and dev.type == "cuda"))
        ctx = torch.randn(batch, wm.CONTEXT, *LATENT, device=dev)
        tgt = torch.randn(batch, *LATENT, device=dev)
        act = torch.randn(batch, wm.ACTION_DIM, device=dev)
        ev = torch.zeros(batch, dtype=torch.long, device=dev)
        params = list(model.parameters())

        def one_step():
            with wm.autocast(dev, amp):
                loss = wm.training_loss(model, ctx, tgt, act, ev)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()

        # Warm-up: at least 3 steps (TunableOp tunes each new matrix shape the first time it
        # appears, which can take minutes) and at least `warmup` seconds (clocks ramp up).
        t_end, k = time.time() + warmup, 0
        while k < 3 or time.time() < t_end:
            one_step()
            k += 1
        sync()
        n, t0 = 0, time.time()
        while time.time() - t0 < seconds:
            one_step()
            n += 1
        sync()
        dt = time.time() - t0
        if os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1":
            try:
                torch.cuda.tunable.write_file()   # also written at exit; this is a safety net
            except Exception:
                pass
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        return {"ok": True, "samples_per_s": n * batch / dt, "steps_per_s": n / dt,
                "peak_gb": round(mem, 2), "optimizer": opt_kind, **info}
    except RuntimeError as e:   # out of memory or an unsupported kernel
        return {"ok": False, "error": str(e).splitlines()[0][:200], **info}


def run_config(amp, batch, seconds, tunable, cache_dir):
    env = dict(os.environ)
    env.update(tunable_env(tunable, cache_dir))
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--amp", amp,
           "--batch", str(batch), "--seconds", str(seconds), "--warmup", "5"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env,
                             cwd=str(Path(__file__).resolve().parent),
                             timeout=1800 if tunable else 600)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timed out"}
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    err = (out.stderr or "").strip().splitlines()
    return {"ok": False, "error": (err[-1] if err else f"exit code {out.returncode}")[:200]}


def describe(r):
    if not r["ok"]:
        return f"failed ({r['error']})"
    return f"{r['samples_per_s']:8.0f} samples/s, {r['peak_gb']} GB"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=10, help="measured seconds per setting")
    ap.add_argument("--out", type=Path, default=Path("bench.json"))
    ap.add_argument("--cache-dir", type=Path, default=Path("."))
    ap.add_argument("--batches", default="64,128,256")
    ap.add_argument("--amps", default="fp16,bf16")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--amp", default="fp16", help=argparse.SUPPRESS)
    ap.add_argument("--batch", type=int, default=64, help=argparse.SUPPRESS)
    ap.add_argument("--warmup", type=float, default=5, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        print(json.dumps(worker(args.amp, args.batch, args.seconds, args.warmup)), flush=True)
        return

    amps = args.amps.split(",")
    batches = [int(b) for b in args.batches.split(",")]
    print(f"Testing {len(amps) * len(batches)} settings for {args.seconds:.0f} s each, "
          "then TunableOp on the best one.", flush=True)
    results = []
    for amp in amps:
        for batch in batches:
            r = run_config(amp, batch, args.seconds, False, args.cache_dir)
            if len(results) == 0 and r.get("report"):
                print(r["report"], flush=True)
            r.update({"amp": amp, "batch": batch, "tunableop": False})
            results.append(r)
            print(f"  {amp} batch {batch:4d}: {describe(r)}", flush=True)

    ok = [r for r in results if r["ok"]]
    if not ok:
        raise SystemExit("Every setting failed; training will use the defaults.")
    gpu = ok[0]["gpu"]

    def pick_batch(amp):
        pick = None
        for r in sorted((r for r in ok if r["amp"] == amp), key=lambda r: r["batch"]):
            if pick is None or r["samples_per_s"] > pick["samples_per_s"] * 1.10:
                pick = r
        return pick

    picks = [p for p in (pick_batch(a) for a in amps) if p]
    best = picks[0]                       # FP16 first: AMD's validated precision on Radeon
    for p in picks[1:]:
        if p["samples_per_s"] > best["samples_per_s"] * 1.05:
            best = p

    if gpu != "CPU":
        print(f"  {best['amp']} batch {best['batch']} with TunableOp "
              "(it tunes every matrix shape first; this can take a few minutes)…", flush=True)
        r = run_config(best["amp"], best["batch"], args.seconds, True, args.cache_dir)
        r.update({"amp": best["amp"], "batch": best["batch"], "tunableop": True})
        results.append(r)
        print(f"    {describe(r)}", flush=True)
        if r["ok"] and r["samples_per_s"] > best["samples_per_s"] * 1.03:
            best = r

    for r in results:
        r.pop("report", None)
    choice = {"amp": best["amp"], "batch": best["batch"], "tunableop": best["tunableop"],
              "samples_per_s": round(best["samples_per_s"]), "gpu": gpu, "results": results}
    args.out.write_text(json.dumps(choice, indent=2))
    print(f"Best: {choice['amp']}, batch {choice['batch']}, TunableOp "
          f"{'on' if choice['tunableop'] else 'off'}: {choice['samples_per_s']} samples/s")


if __name__ == "__main__":
    main()
