"""The world model: data loading, the denoiser network, memory and dream sampling.

A small diffusion transformer predicts the next latent frame from the previous `context` latent
frames and the action taken between them (camera rotation + translation + interaction tag).
It follows DIAMOND's recipe (EDM diffusion, a few Euler steps, noise added to past frames during
training so the model learns to recover from its own mistakes), on TAESD latents instead of
pixels so it trains fast on one GPU.

Memory (models made from version 2 on): every frame of a take has a camera pose, integrated
from the actions. For each prediction, the frames of the take that looked at the same place
(similar viewing direction, close position) are retrieved and given to the network as extra
tokens, tagged with where they were seen from relative to the camera. So when the camera turns
back towards the desk, the network sees real frames of the desk instead of having to invent it.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT = 4          # past frames of models made before memory (old checkpoints still load)
NEW_CONTEXT = 8      # past frames for new models
NEW_MEM_K = 4        # memory frames per prediction for new models
MEM_CANDIDATES = 8   # best-matching frames kept per frame; training picks NEW_MEM_K of them
MEM_GAP = 30         # training: memory may not come from within 2 s (at 15 fps) of the target
MEM_MIN_COS = 0.5    # memory frames must look within 60 degrees of the target's direction
MEM_DIST_W = 0.3     # score penalty per metre between the two camera positions
MEM_DROPOUT = 0.1    # training: share of samples with no memory, so the model works without
ROLLOUT = 3          # training: up to this many context frames come from the model itself
POSE_DIM = 8         # relative rotation xyz, relative position xyz, has_position, valid
ACTION_DIM = 7       # rot xyz, trans xyz, has_trans
MAX_ROT_STEP = 0.6   # rad per frame (~500 deg/s at 15 fps): faster turns are tracking glitches
MAX_TRANS_STEP = 0.5 # m per frame (7.5 m/s): longer jumps are tracking glitches
# Sampling weights (v2.1): which training windows to show more or less often
W_EVENT = 4.0        # the target step has an interaction tag
W_FAST_TURN = 2.0    # the camera turns fast in the target step
FAST_TURN = 0.15     # rad per frame (~130 deg/s at 15 fps)
W_BLURRY = 0.3       # the target frame is much blurrier than the take's median
BLURRY = 0.5         # "much blurrier": sharpness under this share of the median
# Mirror augmentation: flipping left-right negates these action / memory-pose components
# (rotation about Y and Z, movement along X; the camera looks along -Z with +X to the right).
MIRROR_ACT = (1, -1, -1, -1, 1, 1, 1)
MIRROR_POSE = (1, -1, -1, -1, 1, 1, 1, 1)
ROT_SCALE = 10.0     # bring per-step radians / metres to roughly unit range
TRANS_SCALE = 10.0
SIGMA_DATA = 1.0     # latents are normalised to unit variance
P_MEAN, P_STD = -0.4, 1.2
CTX_SIGMA_MAX = 0.6  # noise added to context frames during training


# ---------------------------------------------------------------------------------------
# Camera poses (world-from-camera rotation R and position p, camera looks along -Z)

def skew(v):
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1),
                     np.stack([v[..., 2], z, -v[..., 0]], -1),
                     np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def so3_exp(v):
    """Axis-angle [..., 3] -> rotation matrix [..., 3, 3] (Rodrigues)."""
    v = np.asarray(v, np.float64)
    th = np.linalg.norm(v, axis=-1)
    k = skew(v / np.maximum(th, 1e-12)[..., None])
    th = th[..., None, None]
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)


def so3_log(r):
    """Rotation matrix [..., 3, 3] -> axis-angle [..., 3]."""
    r = np.asarray(r, np.float64)
    th = np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    w = np.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0],
                  r[..., 1, 0] - r[..., 0, 1]], -1)
    s = np.sin(th)
    scale = np.where(s > 1e-6, th / (2 * np.maximum(s, 1e-12)), 0.5)
    return w * scale[..., None]


def integrate_poses(rot, trans):
    """Per-step actions -> per-frame poses, the inverse of preprocess.pose_steps:
    rot_t = log(R_t^T R_t+1) and trans_t = R_t^T (p_t+1 - p_t). The first frame is the origin."""
    n = len(rot) + 1
    r = np.empty((n, 3, 3))
    p = np.zeros((n, 3))
    r[0] = np.eye(3)
    steps = so3_exp(rot)
    for t in range(n - 1):
        p[t + 1] = p[t] + r[t] @ trans[t]
        r[t + 1] = r[t] @ steps[t]
    return r, p


def apply_action(r, p, rot, trans):
    """Pose after one step of (rot, trans), in the same convention as integrate_poses."""
    return r @ so3_exp(rot), p + r @ np.asarray(trans, np.float64)


def view_scores(rq, pq, rb, pb):
    """How well bank frames match query views: [Q, B] (-inf = unusable).
    Viewing directions must be within 60 degrees; closer positions score higher."""
    fq, fb = -rq[..., 2], -rb[..., 2]
    cos = fq @ fb.T
    dist = np.linalg.norm(pq[:, None] - pb[None], axis=-1)
    score = cos - MEM_DIST_W * dist
    score[cos < MEM_MIN_COS] = -np.inf
    return score


def relative_pose(rq, pq, rm, pm, has_pos):
    """Where a memory frame was seen from, relative to the query camera: [..., POSE_DIM]."""
    rqt = np.swapaxes(rq, -1, -2)
    rot = so3_log(rqt @ rm)
    pos = (rqt @ (pm - pq)[..., None])[..., 0]
    has = np.broadcast_to(np.asarray(has_pos, np.float64), rot.shape[:-1])[..., None]
    return np.concatenate([rot, pos * has, has, np.ones_like(has)], -1).astype(np.float32)


def memory_table(r, p, m, gap, has_pos):
    """For every frame of a take, the m best memory frames (local indices, -1 = none) and their
    relative poses, never within `gap` frames of that frame."""
    n = len(r)
    idx = np.full((n, m), -1, np.int64)
    pose = np.zeros((n, m, POSE_DIM), np.float32)
    frames = np.arange(n)
    for a in range(0, n, 1024):
        b = min(n, a + 1024)
        s = view_scores(r[a:b], p[a:b], r, p)
        s[np.abs(frames[a:b, None] - frames[None]) <= gap] = -np.inf
        k = min(m, n)
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        top_s = np.take_along_axis(s, top, 1)
        order = np.argsort(-top_s, axis=1)
        top, top_s = np.take_along_axis(top, order, 1), np.take_along_axis(top_s, order, 1)
        ok = np.isfinite(top_s)
        idx[a:b, :k] = np.where(ok, top, -1)
        q = np.arange(a, b)[:, None].repeat(k, 1)
        rp = relative_pose(r[q], p[q], r[top], p[top], has_pos)
        rp[~ok] = 0
        pose[a:b, :k] = rp
    return idx, pose


def memory_for(take, rq, pq, k, device, exclude=None):
    """Memory for one query view from a take's frames: tensors [1, k, C, H, W] and
    [1, k, POSE_DIM]; empty slots are zeros with valid = 0. `exclude` = (first, last) frames
    not to use (dream videos exclude the clip they are predicting)."""
    lat = take["lat"]
    s = view_scores(np.asarray(rq)[None], np.asarray(pq)[None], take["R"], take["p"])[0]
    if exclude is not None:
        s[max(0, exclude[0]):max(0, exclude[1])] = -np.inf
    mem = np.zeros((k,) + lat.shape[1:], np.float32)
    pose = np.zeros((k, POSE_DIM), np.float32)
    for slot, j in enumerate(np.argsort(-s)[:k]):
        if not np.isfinite(s[j]):
            break
        mem[slot] = lat[j]
        pose[slot] = relative_pose(np.asarray(rq), np.asarray(pq), take["R"][j], take["p"][j],
                                   float(take["has"]))
    return torch.from_numpy(mem)[None].to(device), torch.from_numpy(pose)[None].to(device)


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


def clean_actions(rot, trans, has_trans):
    """Cap physically impossible per-frame jumps (ARCore relocalisation glitches). A single
    1.8 rad step was enough to overflow FP16 inside the network and turn the loss to NaN."""
    rot = np.asarray(rot, np.float64).copy()
    trans = np.asarray(trans, np.float64)
    has = np.asarray(has_trans, bool).copy()
    rn = np.linalg.norm(rot, axis=1)
    big = rn > MAX_ROT_STEP
    rot[big] *= (MAX_ROT_STEP / rn[big])[:, None]
    has[np.linalg.norm(trans, axis=1) > MAX_TRANS_STEP] = False      # movement unknown there
    return rot, np.where(has[:, None], trans, 0.0), has


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
            rot, trans, has = clean_actions(a["rot"], a["trans"], a["has_trans"])
            r, p = integrate_poses(rot, trans * has[:, None])
            sharp = None
            if (d / "sharpness.npy").exists():
                sharp = np.load(d / "sharpness.npy").astype(np.float32)
                if len(sharp) != len(lat):
                    sharp = None
            self.takes.append({
                "name": e["take"],
                "dir": d,
                "lat": lat,
                "act": action_vector(rot, trans, has),
                "ev": a["event"].astype(np.int64),
                "R": r, "p": p, "has": bool(has.any()),
                "turn": np.linalg.norm(rot, axis=1), "sharp": sharp,
            })
        if not self.takes:
            return
        if stats is None:
            allz = np.concatenate([t["lat"] for t in self.takes])
            stats = {"mean": allz.mean(axis=(0, 2, 3), dtype=np.float64).tolist(),
                     "std": (allz.std(axis=(0, 2, 3), dtype=np.float64) + 1e-6).tolist()}
        self.stats = stats
        m = np.array(stats["mean"], dtype=np.float32)[None, :, None, None]
        s = np.array(stats["std"], dtype=np.float32)[None, :, None, None]
        for t in self.takes:
            t["lat"] = (t["lat"] - m) / s
        self.latent_shape = self.takes[0]["lat"].shape[1:]

    def __len__(self):
        return len(self.takes)

    def frames(self):
        return sum(len(t["lat"]) for t in self.takes)

    def to_device(self, device, context=NEW_CONTEXT, rollout=ROLLOUT, mem_k=0,
                  candidates=MEM_CANDIDATES, weighted=True, mirror=False):
        """Move the whole split to the GPU once, so batches are built there with no CPU work
        or host-to-device copies per step (an hour of footage is ~125 MB in FP16). Each window
        holds rollout + context + 1 frames; with memory, the best memory candidates of every
        frame are precomputed."""
        self.context, self.rollout, self.mem_k, self.mirror = context, rollout, mem_k, mirror
        lat, act, ev, starts, mem_idx, mem_pose, weights = [], [], [], [], [], [], []
        offset = 0
        for t in self.takes:
            n = len(t["lat"])
            lat.append(t["lat"].astype(np.float16))
            a = np.zeros((n, ACTION_DIM), np.float32)
            a[:n - 1] = t["act"]                       # action i = step from frame i to i+1
            act.append(a)
            e = np.zeros(n, np.int64)
            e[:n - 1] = t["ev"]
            ev.append(e)
            if n - context - rollout > 0:              # windows stay inside one take
                local = np.arange(n - context - rollout)
                starts.append(local + offset)
                w = np.ones(len(local))
                if weighted:                           # the step / frame each window teaches
                    step_i, frame_i = local + context + rollout - 1, local + context + rollout
                    w[t["ev"][step_i] > 0] *= W_EVENT
                    w[t["turn"][step_i] > FAST_TURN] *= W_FAST_TURN
                    if t["sharp"] is not None:
                        w[t["sharp"][frame_i] < BLURRY * np.median(t["sharp"])] *= W_BLURRY
                weights.append(w)
            if mem_k:
                idx, pose = memory_table(t["R"], t["p"], candidates, MEM_GAP, float(t["has"]))
                mem_idx.append(np.where(idx >= 0, idx + offset, -1))
                mem_pose.append(pose)
            offset += n
        if not starts:
            raise SystemExit("The takes are too short for this model's context.")
        self.g_lat = torch.from_numpy(np.concatenate(lat)).to(device)
        self.g_act = torch.from_numpy(np.concatenate(act)).to(device)
        self.g_ev = torch.from_numpy(np.concatenate(ev)).to(device)
        self.g_starts = torch.from_numpy(np.concatenate(starts)).to(device)
        w = np.concatenate(weights)
        self.g_weights = torch.from_numpy(w / w.sum()).float().to(device)
        self.weight_stats = {"events": float((w >= W_EVENT).mean()),
                             "boosted_share": float(w[w > 1].sum() / w.sum())}
        self.g_window = torch.arange(context + rollout + 1, device=device)
        self.g_steps = torch.arange(context + rollout, device=device)
        if mem_k:
            self.g_mem_idx = torch.from_numpy(np.concatenate(mem_idx)).to(device)
            self.g_mem_pose = torch.from_numpy(np.concatenate(mem_pose)).to(device)
        return self

    def gpu_batch(self, n):
        """Random training windows, built entirely on the device. Returns a dict:
        win [n, R+C+1, ...] latents (the last one is the target), act / ev for each step of
        the window, and with memory: mem [n, K, ...] and mem_pose [n, K, POSE_DIM]."""
        dev = self.g_starts.device
        s = self.g_starts[torch.multinomial(self.g_weights, n, replacement=True)]
        steps = s[:, None] + self.g_steps
        out = {"win": self.g_lat[s[:, None] + self.g_window].float(),
               "act": self.g_act[steps], "ev": self.g_ev[steps]}
        if self.mem_k:
            target = s + self.context + self.rollout
            cand, pose = self.g_mem_idx[target], self.g_mem_pose[target]
            # a random K of the candidates, valid ones first; some samples get no memory
            pick = (torch.rand(cand.shape, device=dev) + (cand >= 0)).topk(self.mem_k, 1).indices
            ci = cand.gather(1, pick)
            cp = pose.gather(1, pick[..., None].expand(-1, -1, POSE_DIM))
            valid = (ci >= 0) & (torch.rand(n, 1, device=dev) >= MEM_DROPOUT)
            out["mem"] = self.g_lat[ci.clamp(min=0)].float() * valid[..., None, None, None]
            out["mem_pose"] = cp * valid[..., None]
        if self.mirror:                                # flip half the samples left-right
            flip = torch.rand(n, device=dev) < 0.5
            f5 = flip[:, None, None, None, None]
            out["win"] = torch.where(f5, out["win"].flip(-1), out["win"])
            sa = torch.tensor(MIRROR_ACT, device=dev, dtype=out["act"].dtype)
            out["act"] = torch.where(flip[:, None, None], out["act"] * sa, out["act"])
            if self.mem_k:
                out["mem"] = torch.where(f5, out["mem"].flip(-1), out["mem"])
                sp = torch.tensor(MIRROR_POSE, device=dev, dtype=out["mem_pose"].dtype)
                out["mem_pose"] = torch.where(flip[:, None, None], out["mem_pose"] * sp,
                                              out["mem_pose"])
        return out


# ---------------------------------------------------------------------------------------
# Network: a small DiT with adaLN-Zero conditioning

def fourier(x, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=x.device) / half)
    a = x[:, None].float() * freqs[None] * 10.0
    return torch.cat([a.sin(), a.cos()], dim=-1)


def patchify(x, ph, pw):
    """[B, C, H, W] -> [B, tokens, C*ph*pw], in the same order as a stride-(ph, pw) conv."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // ph, ph, w // pw, pw).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // ph) * (w // pw), c * ph * pw)


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
    """The defaults (context 4, no memory) rebuild models saved before memory existed; new
    models are created with context=NEW_CONTEXT and mem_k=NEW_MEM_K."""

    def __init__(self, latent_shape, dim=384, depth=8, heads=6, patch=2, n_events=64,
                 context=CONTEXT, mem_k=0, mem_patch=(3, 4)):
        super().__init__()
        ch, h, w = latent_shape
        assert h % patch == 0 and w % patch == 0, "latent size must be divisible by the patch"
        self.config = dict(latent_shape=list(latent_shape), dim=dim, depth=depth, heads=heads,
                           patch=patch, n_events=n_events, context=context, mem_k=mem_k,
                           mem_patch=list(mem_patch))
        self.ch, self.h, self.w, self.p = ch, h, w, patch
        self.context, self.mem_k, self.mem_patch = context, mem_k, tuple(mem_patch)
        # Patch embedding as reshape + Linear (the same maths as a stride-`patch` conv): every
        # matrix multiply in the network is then a GEMM for hipBLASLt, which TunableOp tunes,
        # and none goes through MIOpen's convolution solvers.
        self.embed = nn.Linear(ch * (context + 1) * patch * patch, dim)
        self.pos = nn.Parameter(torch.randn(1, (h // patch) * (w // patch), dim) * 0.02)
        if mem_k:
            mph, mpw = self.mem_patch
            assert h % mph == 0 and w % mpw == 0, "latent size must be divisible by mem_patch"
            # Memory frames use coarser 3x4 patches (48 tokens each) to keep attention cheap.
            self.mem_embed = nn.Linear(ch * mph * mpw, dim)
            self.mem_pos = nn.Parameter(torch.randn(1, (h // mph) * (w // mpw), dim) * 0.02)
            self.mem_pose_mlp = nn.Sequential(nn.Linear(POSE_DIM, dim), nn.SiLU(),
                                              nn.Linear(dim, dim))
            self.mem_empty = nn.Parameter(torch.zeros(dim))
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

    def forward(self, x, ctx, c_noise, ctx_sigma, action, event, mem=None, mem_pose=None):
        b = x.shape[0]
        inp = torch.cat([ctx.flatten(1, 2), x], dim=1)
        t = self.embed(patchify(inp, self.p, self.p)) + self.pos
        n_main = t.shape[1]
        if self.mem_k:
            k = self.mem_k
            if mem is None:
                mem = x.new_zeros(b, k, self.ch, self.h, self.w)
                mem_pose = x.new_zeros(b, k, POSE_DIM)
            mph, mpw = self.mem_patch
            mt = self.mem_embed(patchify(mem.flatten(0, 1), mph, mpw)) + self.mem_pos
            valid = mem_pose[..., -1:]
            cond = self.mem_pose_mlp(mem_pose) * valid + self.mem_empty * (1 - valid)
            mt = mt.view(b, k, -1, mt.shape[-1]) + cond[:, :, None]
            t = torch.cat([t, mt.flatten(1, 2).to(t.dtype)], dim=1)
        c = (self.noise_mlp(fourier(c_noise)) + self.ctx_noise_mlp(fourier(ctx_sigma))
             + self.act_mlp(action) + self.event(event.clamp(0, self.event.num_embeddings - 1)))
        for blk in self.blocks:
            t = blk(t, c)
        t = t[:, :n_main]
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


def training_loss(model, ctx, tgt, act, ev, mem=None, mem_pose=None):
    b = tgt.shape[0]
    sigma = (torch.randn(b, device=tgt.device) * P_STD + P_MEAN).exp()
    sigma_ctx = torch.rand(b, device=tgt.device) * CTX_SIGMA_MAX
    ctx = noisy_context(ctx, sigma_ctx)
    c_skip, c_out, c_in, c_noise = precond(sigma)
    x = tgt + sigma[:, None, None, None] * torch.randn_like(tgt)
    pred = model(c_in * x, ctx, c_noise, sigma_ctx, act, ev, mem, mem_pose)
    target = (tgt - c_skip * x) / c_out
    return F.mse_loss(pred.float(), target.float())


def null_event(model):
    """The last tag slot is never a real tag: training uses it, with a zero action, to mean
    "action unknown", which is what action guidance compares against."""
    m = getattr(model, "model", model)
    return m.event.num_embeddings - 1


def denoise(model, x, sigma, ctx, sigma_ctx, act, ev, mem=None, mem_pose=None, guidance=1.0):
    c_skip, c_out, c_in, c_noise = precond(sigma)
    out = model(c_in * x, ctx, c_noise, sigma_ctx, act, ev, mem, mem_pose).float()
    if guidance != 1.0:
        # Action guidance: push the prediction away from "action unknown" towards "this action",
        # so the dream follows the keys more firmly (classifier-free guidance on the action).
        un = model(c_in * x, ctx, c_noise, sigma_ctx, torch.zeros_like(act),
                   torch.full_like(ev, null_event(model)), mem, mem_pose).float()
        out = un + guidance * (out - un)
    return c_skip * x + c_out * out


@torch.no_grad()
def sample_next(model, ctx, act, ev, steps=3, ctx_noise=0.05, sigma_min=0.002, sigma_max=5.0,
                rho=7.0, mem=None, mem_pose=None, guidance=1.0):
    """Generate the next latent frame (normalised) given context [B, context, C, H, W]."""
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
        d = (x - denoise(model, x, s, ctx, sigma_ctx, act, ev, mem, mem_pose, guidance)) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x


class GraphedModel:
    """Play mode runs the network at batch 1, where launching hundreds of small GPU kernels
    costs more than the maths. This records one forward pass as a HIP/CUDA graph per input
    signature and replays it as a single launch. Any failure falls back to the normal model."""

    def __init__(self, model, amp="fp16"):
        self.model, self.amp = model, amp
        self.graphs, self.failed = {}, False
        self.context, self.mem_k, self.event = model.context, model.mem_k, model.event

    def __call__(self, *args):
        dev = args[0].device
        if self.failed or dev.type != "cuda":
            return self.model(*args)
        key = tuple(None if a is None else (tuple(a.shape), a.dtype) for a in args)
        try:
            if key not in self.graphs:
                self.graphs[key] = self._capture(args)
            g, ins, out = self.graphs[key]
            for buf, a in zip(ins, args):
                if buf is not None:
                    buf.copy_(a)
            g.replay()
            return out.clone()
        except Exception as e:  # graphs unsupported here: run normally from now on
            print(f"GPU graph replay unavailable ({type(e).__name__}); running normally.")
            self.failed = True
            return self.model(*args)

    def _capture(self, args):
        ins = [None if a is None else a.clone() for a in args]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.autocast("cuda", dtype=AMP_DTYPES.get(self.amp) or torch.float32,
                                                     enabled=AMP_DTYPES.get(self.amp) is not None,
                                                     cache_enabled=False):
            for _ in range(3):
                self.model(*ins)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.autocast("cuda", dtype=AMP_DTYPES.get(self.amp) or torch.float32,
                                                 enabled=AMP_DTYPES.get(self.amp) is not None,
                                                 cache_enabled=False):
            out = self.model(*ins)
        return g, ins, out


# ---------------------------------------------------------------------------------------
# Precision. AMD's ROCm support matrix for Radeon validates FP32, FP16 and FP16 mixed
# precision, so FP16 autocast is the default; BF16 also runs on RDNA 4 and is kept as an option.

AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}


def autocast(device, amp="fp16"):
    dtype = AMP_DTYPES.get(amp)
    enabled = device.type == "cuda" and dtype is not None
    return torch.autocast(device.type, dtype=dtype or torch.float32, enabled=enabled)


def gpu_report():
    """One line describing the GPU stack, for logs."""
    if not torch.cuda.is_available():
        return f"torch {torch.__version__}, no GPU"
    p = torch.cuda.get_device_properties(0)
    arch = getattr(p, "gcnArchName", "")
    hip = getattr(torch.version, "hip", None)
    stack = f"ROCm/HIP {hip}" if hip else f"CUDA {torch.version.cuda}"
    return (f"{p.name} {arch} · {p.total_memory / 2**30:.1f} GB · {p.multi_processor_count} CUs · "
            f"torch {torch.__version__} · {stack}")


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


def upgrade_state_dict(sd):
    """Checkpoints from before the patch embedding became a Linear layer: same weights,
    flattened from conv layout [dim, C, p, p] to [dim, C*p*p]."""
    w = sd.get("embed.weight")
    if w is not None and w.dim() == 4:
        sd = dict(sd)
        sd["embed.weight"] = w.reshape(w.shape[0], -1)
    return sd


def build_model(config):
    cfg = dict(config)
    return WorldDiT(tuple(cfg.pop("latent_shape")), **cfg)


def load_model(path: Path, device, use_ema=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["config"]).to(device)
    model.load_state_dict(upgrade_state_dict(ck["ema"] if use_ema else ck["model"]))
    model.eval()
    return model, ck


def denormalise(z, stats):
    m = torch.tensor(stats["mean"], device=z.device)[None, :, None, None]
    s = torch.tensor(stats["std"], device=z.device)[None, :, None, None]
    return z * s + m
