"""The world model: data loading, the denoiser network and dream sampling.

A small diffusion transformer predicts the next latent frame from the previous CONTEXT latent
frames and the action taken between them (camera rotation + translation + interaction tag).
It follows DIAMOND's recipe (EDM diffusion, a few Euler steps, noise added to past frames during
training so the model learns to recover from its own mistakes), on TAESD latents instead of
pixels so it trains fast on one GPU.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT = 4          # past frames the model sees
ACTION_DIM = 7       # rot xyz, trans xyz, has_trans
ROT_SCALE = 10.0     # bring per-step radians / metres to roughly unit range
TRANS_SCALE = 10.0
SIGMA_DATA = 1.0     # latents are normalised to unit variance
P_MEAN, P_STD = -0.4, 1.2
CTX_SIGMA_MAX = 0.6  # noise added to context frames during training


# ---------------------------------------------------------------------------------------
# Data

def action_vector(rot, trans, has_trans):
    """Per-step action, scaled. Shapes [..., 3], [..., 3], [...]."""
    h = np.asarray(has_trans, dtype=np.float32)
    return np.concatenate([
        np.asarray(rot, dtype=np.float32) * ROT_SCALE,
        np.asarray(trans, dtype=np.float32) * TRANS_SCALE * h[..., None],
        h[..., None],
    ], axis=-1)


class Dataset:
    """All encoded takes of one split, held in RAM (an hour of footage is ~250 MB)."""

    def __init__(self, processed: Path, split=None, stats=None):
        index = json.loads((processed / "index.json").read_text())
        self.vocab = index.get("event_vocab", [])
        self.takes = []
        for e in index["takes"]:
            if split and e.get("split") != split:
                continue
            d = processed / e["take"]
            if not (d / "latents.npy").exists():
                continue
            lat = np.load(d / "latents.npy").astype(np.float32)
            if len(lat) < CONTEXT + 2:
                continue
            a = np.load(d / "actions.npz")
            self.takes.append({
                "name": e["take"],
                "dir": d,
                "lat": lat,
                "act": action_vector(a["rot"], a["trans"], a["has_trans"]),
                "ev": a["event"].astype(np.int64),
            })
        if not self.takes:
            return
        if stats is None:
            allz = np.concatenate([t["lat"] for t in self.takes])
            stats = {"mean": allz.mean(axis=(0, 2, 3)).tolist(),
                     "std": (allz.std(axis=(0, 2, 3)) + 1e-6).tolist()}
        self.stats = stats
        m = np.array(stats["mean"], dtype=np.float32)[None, :, None, None]
        s = np.array(stats["std"], dtype=np.float32)[None, :, None, None]
        for t in self.takes:
            t["lat"] = (t["lat"] - m) / s
        self.weights = np.array([len(t["lat"]) - CONTEXT for t in self.takes], dtype=np.float64)
        self.weights /= self.weights.sum()
        self.latent_shape = self.takes[0]["lat"].shape[1:]

    def __len__(self):
        return len(self.takes)

    def frames(self):
        return sum(len(t["lat"]) for t in self.takes)

    def batch(self, n, device):
        ctx, tgt, act, ev = [], [], [], []
        picks = np.random.choice(len(self.takes), size=n, p=self.weights)
        for k in picks:
            t = self.takes[k]
            s = random.randint(0, len(t["lat"]) - CONTEXT - 1)
            ctx.append(t["lat"][s:s + CONTEXT])
            tgt.append(t["lat"][s + CONTEXT])
            act.append(t["act"][s + CONTEXT - 1])   # the step from the last context frame
            ev.append(t["ev"][s + CONTEXT - 1])
        f = lambda x: torch.from_numpy(np.stack(x)).to(device)
        return f(ctx), f(tgt), f(act), f(ev)


# ---------------------------------------------------------------------------------------
# Network: a small DiT with adaLN-Zero conditioning

def fourier(x, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=x.device) / half)
    a = x[:, None].float() * freqs[None] * 10.0
    return torch.cat([a.sin(), a.cos()], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"),
                                 nn.Linear(dim * 4, dim))
        self.mod = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.mod(F.silu(c))[:, None].chunk(6, dim=-1)
        x = x + g1 * self.attn(self.n1(x) * (1 + sc1) + s1)
        x = x + g2 * self.mlp(self.n2(x) * (1 + sc2) + s2)
        return x


class WorldDiT(nn.Module):
    def __init__(self, latent_shape, dim=384, depth=8, heads=6, patch=2, n_events=64):
        super().__init__()
        ch, h, w = latent_shape
        assert h % patch == 0 and w % patch == 0, "latent size must be divisible by the patch"
        self.config = dict(latent_shape=list(latent_shape), dim=dim, depth=depth, heads=heads,
                           patch=patch, n_events=n_events)
        self.ch, self.h, self.w, self.p = ch, h, w, patch
        self.embed = nn.Conv2d(ch * (CONTEXT + 1), dim, patch, patch)
        self.pos = nn.Parameter(torch.randn(1, (h // patch) * (w // patch), dim) * 0.02)
        self.noise_mlp = nn.Sequential(nn.Linear(128, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ctx_noise_mlp = nn.Sequential(nn.Linear(128, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act_mlp = nn.Sequential(nn.Linear(ACTION_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.event = nn.Embedding(n_events + 1, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, patch * patch * ch)
        for m in (self.final_mod, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, ctx, c_noise, ctx_sigma, action, event):
        b = x.shape[0]
        inp = torch.cat([ctx.flatten(1, 2), x], dim=1)
        t = self.embed(inp).flatten(2).transpose(1, 2) + self.pos
        c = (self.noise_mlp(fourier(c_noise)) + self.ctx_noise_mlp(fourier(ctx_sigma))
             + self.act_mlp(action) + self.event(event.clamp(0, self.event.num_embeddings - 1)))
        for blk in self.blocks:
            t = blk(t, c)
        shift, scale = self.final_mod(F.silu(c))[:, None].chunk(2, dim=-1)
        t = self.out(self.final_norm(t) * (1 + scale) + shift)
        hp, wp, p = self.h // self.p, self.w // self.p, self.p
        t = t.view(b, hp, wp, p, p, self.ch).permute(0, 5, 1, 3, 2, 4)
        return t.reshape(b, self.ch, self.h, self.w)


# ---------------------------------------------------------------------------------------
# EDM diffusion

def precond(sigma):
    s = sigma[:, None, None, None]
    c_skip = SIGMA_DATA ** 2 / (s ** 2 + SIGMA_DATA ** 2)
    c_out = s * SIGMA_DATA / (s ** 2 + SIGMA_DATA ** 2).sqrt()
    c_in = 1 / (s ** 2 + SIGMA_DATA ** 2).sqrt()
    c_noise = sigma.log() / 4
    return c_skip, c_out, c_in, c_noise


def noisy_context(ctx, sigma_ctx):
    return ctx + sigma_ctx[:, None, None, None, None] * torch.randn_like(ctx)


def training_loss(model, ctx, tgt, act, ev):
    b = tgt.shape[0]
    sigma = (torch.randn(b, device=tgt.device) * P_STD + P_MEAN).exp()
    sigma_ctx = torch.rand(b, device=tgt.device) * CTX_SIGMA_MAX
    ctx = noisy_context(ctx, sigma_ctx)
    c_skip, c_out, c_in, c_noise = precond(sigma)
    x = tgt + sigma[:, None, None, None] * torch.randn_like(tgt)
    pred = model(c_in * x, ctx, c_noise, sigma_ctx, act, ev)
    target = (tgt - c_skip * x) / c_out
    return F.mse_loss(pred.float(), target.float())


def denoise(model, x, sigma, ctx, sigma_ctx, act, ev):
    c_skip, c_out, c_in, c_noise = precond(sigma)
    return c_skip * x + c_out * model(c_in * x, ctx, c_noise, sigma_ctx, act, ev).float()


@torch.no_grad()
def sample_next(model, ctx, act, ev, steps=3, ctx_noise=0.05, sigma_min=0.002, sigma_max=5.0,
                rho=7.0):
    """Generate the next latent frame (normalised) given context [B, CONTEXT, C, H, W]."""
    b = ctx.shape[0]
    dev = ctx.device
    sigma_ctx = torch.full((b,), ctx_noise, device=dev)
    ctx = noisy_context(ctx, sigma_ctx)
    ramp = torch.linspace(0, 1, steps, device=dev)
    sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1, device=dev)])
    x = torch.randn(b, *ctx.shape[2:], device=dev) * sigmas[0]
    for i in range(steps):
        s = sigmas[i].expand(b)
        d = (x - denoise(model, x, s, ctx, sigma_ctx, act, ev)) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x


# ---------------------------------------------------------------------------------------
# Checkpoints

def save_checkpoint(path: Path, model, ema, opt, step, stats, vocab, minutes):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "config": model.config, "model": model.state_dict(), "ema": ema.state_dict(),
        "opt": opt.state_dict() if opt is not None else None, "step": step, "stats": stats,
        "vocab": vocab, "minutes_trained": minutes,
    }, tmp)
    tmp.replace(path)


def load_model(path: Path, device, use_ema=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck["config"])
    model = WorldDiT(tuple(cfg.pop("latent_shape")), **cfg).to(device)
    model.load_state_dict(ck["ema"] if use_ema else ck["model"])
    model.eval()
    return model, ck


def denormalise(z, stats):
    m = torch.tensor(stats["mean"], device=z.device)[None, :, None, None]
    s = torch.tensor(stats["std"], device=z.device)[None, :, None, None]
    return z * s + m
