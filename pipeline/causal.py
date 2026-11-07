"""World model v3 / v4: a frame-causal diffusion transformer trained on whole clips.

v4 ("world", the permanent architecture) adds: flow matching instead of EDM (straight paths
from noise to frame, the basis for few-step distillation later), 512x288 frames through 4x4
latent patches (same token count as v3 at 256x144), room / lighting / clip-speed conditioning
with reserved slots, and 256 tag slots. Every choice that cannot change after training starts
is made generously, so the same model can keep training as footage grows.


What changes from v2 (worldmodel.WorldDiT), and why:
  * Whole-clip training. The network sees a clip of CLIP frames and predicts every frame at
    once; frame-causal attention lets each frame see only itself and the past. One example
    gives CLIP lessons instead of one (v2 stacked past frames as channels and predicted one).
  * Diffusion forcing. Every frame has its own noise level, so the network learns to work from
    any mix of sharp and blurry history, which is what its own dreams look like.
  * Stable at scale. RMSNorm, QK-norm (normalised attention queries and keys keep FP16 from
    overflowing), 3D rotary positions (time, height, width) instead of learned position tables.
  * Geometry. Each frame carries its camera pose relative to the newest frame; memory frames
    (same place, other moments of the take) are prefix tokens with their relative pose.
  * Action guidance. A null tag means "action unknown" (trained by dropping 10% of actions);
    when dreaming, predictions are pushed away from it towards the pressed action.
  * Size. ~120M parameters by default ("large"); gradient checkpointing keeps memory in 16 GB.
When dreaming, the last WINDOW frames are re-run each step (history at a small noise level,
the new frame being denoised), so no cache is needed and any window length works.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import worldmodel as wm

# name: (width, blocks, heads); heads are 64 wide
SIZES = {"tiny": (128, 2, 2), "base": (512, 12, 8), "large": (768, 16, 12), "xl": (1024, 20, 16)}
CLIP = 16          # frames per training clip
WINDOW = 12        # frames used when dreaming: 11 of history + the one being generated
SELF_SIGMA = (0.3, 1.0)   # self-forcing: noise range for the model's own frames in training.
# Their error is about this size, so labelling them with a tiny noise level would demand huge
# corrections from the network (EDM divides the target by the noise level).


class RMSNorm(nn.Module):
    def __init__(self, dim, affine=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(x.dtype)


def rope_tables(t_pos, y_pos, x_pos, head_dim, device):
    """cos/sin [N, head_dim/2] for 3D rotary positions: 1/4 of each head for time, 3/8 for
    height, 3/8 for width. Rows with NaN position (memory tokens) get no rotation."""
    dt = head_dim // 4
    dy = (head_dim - dt) // 2
    dx = head_dim - dt - dy

    def ang(pos, d, base):
        inv = base ** (-torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)
        return pos[:, None] * inv[None]
    a = torch.cat([ang(t_pos, dt, 100.0), ang(y_pos, dy, 100.0), ang(x_pos, dx, 100.0)], dim=1)
    a = torch.nan_to_num(a, nan=0.0)
    return a.cos(), a.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask, cos, sin):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos.to(q.dtype), sin.to(q.dtype)), apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2 = RMSNorm(dim, affine=False), RMSNorm(dim, affine=False)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"),
                                 nn.Linear(dim * 4, dim))
        self.mod = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, cond, group, mask, cos, sin):
        # adaLN-Zero per frame / memory slot (computed once per group, then spread to tokens)
        s1, sc1, g1, s2, sc2, g2 = self.mod(F.silu(cond))[:, group].chunk(6, dim=-1)
        x = x + g1 * self.attn(self.n1(x) * (1 + sc1) + s1, mask, cos, sin)
        return x + g2 * self.mlp(self.n2(x) * (1 + sc2) + s2)


class CausalWorldDiT(nn.Module):
    def __init__(self, latent_shape, size="large", patch=2, n_events=64, mem_k=wm.NEW_MEM_K,
                 mem_patch=(3, 4), clip=CLIP, window=WINDOW, arch="causal", objective="edm",
                 n_rooms=0, n_lights=0, max_stride=1):
        super().__init__()
        ch, h, w = latent_shape
        dim, depth, heads = SIZES[size]
        self.config = dict(arch=arch, latent_shape=list(latent_shape), size=size, patch=patch,
                           n_events=n_events, mem_k=mem_k, mem_patch=list(mem_patch), clip=clip,
                           window=window, objective=objective, n_rooms=n_rooms,
                           n_lights=n_lights, max_stride=max_stride)
        self.objective, self.max_stride = objective, max_stride
        self.room = nn.Embedding(n_rooms + 1, dim) if n_rooms else None
        self.light = nn.Embedding(n_lights + 1, dim) if n_lights else None
        self.stride = nn.Embedding(max_stride + 1, dim) if max_stride > 1 else None
        self.ch, self.h, self.w, self.p = ch, h, w, patch
        self.dim, self.heads, self.mem_k, self.mem_patch = dim, heads, mem_k, tuple(mem_patch)
        self.clip, self.window = clip, window
        self.context = window - 1               # history frames when dreaming (same API as v2)
        self.grad_ckpt = False
        self.embed = nn.Linear(ch * patch * patch, dim)
        self.noise_mlp = nn.Sequential(nn.Linear(128, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act_mlp = nn.Sequential(nn.Linear(wm.ACTION_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.pose_mlp = nn.Sequential(nn.Linear(wm.POSE_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.event = nn.Embedding(n_events + 1, dim)
        if mem_k:
            mph, mpw = self.mem_patch
            self.mem_embed = nn.Linear(ch * mph * mpw, dim)
            self.mem_pos = nn.Parameter(torch.randn(1, (h // mph) * (w // mpw), dim) * 0.02)
            self.mem_pose_mlp = nn.Sequential(nn.Linear(wm.POSE_DIM, dim), nn.SiLU(),
                                              nn.Linear(dim, dim))
            self.mem_empty = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.final_norm = RMSNorm(dim, affine=False)
        self.final_mod = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, patch * patch * ch)
        for m in (self.final_mod, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self._cache = {}

    def _layout(self, t, k, device):
        """Attention mask, rotary tables and token->group index for t frames and k memory
        slots; cached. Memory tokens come first; they attend only to memory. A frame's tokens
        attend to memory and to every token of frames up to and including their own."""
        key = (t, k, str(device))
        if key in self._cache:
            return self._cache[key]
        hp, wp = self.h // self.p, self.w // self.p
        P = hp * wp
        if k:
            mph, mpw = self.mem_patch
            Pm = (self.h // mph) * (self.w // mpw)
        else:
            Pm = 0
        nm, nf = k * Pm, t * P
        frame_of = torch.cat([torch.full((nm,), -1, device=device),
                              torch.arange(t, device=device).repeat_interleave(P)])
        q, kk = frame_of[:, None], frame_of[None, :]
        mask = (kk == -1) | ((q >= 0) & (kk >= 0) & (kk <= q))
        yy = torch.arange(hp, device=device).repeat_interleave(wp).float().repeat(t)
        xx = torch.arange(wp, device=device).repeat(hp).float().repeat(t)
        tt = torch.arange(t, device=device).repeat_interleave(P).float()
        nan = torch.full((nm,), float("nan"), device=device)
        cos, sin = rope_tables(torch.cat([nan, tt]), torch.cat([nan, yy]), torch.cat([nan, xx]),
                               self.dim // self.heads, device)
        group = torch.cat([t + torch.arange(k, device=device).repeat_interleave(Pm),
                           torch.arange(t, device=device).repeat_interleave(P)])
        self._cache[key] = (mask, cos, sin, group, nm)
        return self._cache[key]

    def forward(self, x, c_noise, act, ev, pose, mem=None, mem_pose=None, room=None, light=None,
                stride=None):
        """x [B, T, C, H, W] (noisy, already scaled by c_in), c_noise [B, T], act [B, T, 7]
        (action into each frame), ev [B, T], pose [B, T, POSE_DIM] (relative to the newest
        frame) -> denoiser output [B, T, C, H, W]."""
        b, t = x.shape[:2]
        tok = self.embed(wm.patchify(x.flatten(0, 1), self.p, self.p)).view(b, -1, self.dim)
        cond = (self.noise_mlp(wm.fourier(c_noise.flatten())).view(b, t, -1) + self.act_mlp(act)
                + self.event(ev.clamp(0, self.event.num_embeddings - 1)) + self.pose_mlp(pose))
        for emb, ids in ((self.room, room), (self.light, light), (self.stride, stride)):
            if emb is not None:                       # per clip, shared by all its frames
                if ids is None:
                    ids = torch.full((b,), 1 if emb is self.stride else 0, device=x.device)
                cond = cond + emb(ids.clamp(0, emb.num_embeddings - 1))[:, None]
        k = self.mem_k
        if k:
            if mem is None:
                mem = x.new_zeros(b, k, self.ch, self.h, self.w)
                mem_pose = x.new_zeros(b, k, wm.POSE_DIM)
            mph, mpw = self.mem_patch
            mt = self.mem_embed(wm.patchify(mem.flatten(0, 1), mph, mpw)) + self.mem_pos
            valid = mem_pose[..., -1:]
            mcond = self.mem_pose_mlp(mem_pose) * valid + self.mem_empty * (1 - valid)
            mt = (mt.view(b, k, -1, self.dim) + mcond[:, :, None]).flatten(1, 2)
            tok = torch.cat([mt.to(tok.dtype), tok], dim=1)
            cond = torch.cat([cond, mcond.to(cond.dtype)], dim=1)
        mask, cos, sin, group, nm = self._layout(t, k, x.device)
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                tok = checkpoint(blk, tok, cond, group, mask, cos, sin, use_reentrant=False)
            else:
                tok = blk(tok, cond, group, mask, cos, sin)
        tok = tok[:, nm:]
        P = (self.h // self.p) * (self.w // self.p)
        shift, scale = self.final_mod(F.silu(cond[:, :t])).repeat_interleave(P, dim=1).chunk(2, -1)
        y = self.out(self.final_norm(tok) * (1 + scale) + shift)
        hp, wp, p = self.h // self.p, self.w // self.p, self.p
        y = y.view(b * t, hp, wp, p, p, self.ch).permute(0, 5, 1, 3, 2, 4)
        return y.reshape(b, t, self.ch, self.h, self.w)


# ---------------------------------------------------------------------------------------
# Poses on the GPU

def so3_log_t(r):
    cos = ((r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1) / 2).clamp(-1, 1)
    th = torch.arccos(cos)
    w = torch.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0],
                     r[..., 1, 0] - r[..., 0, 1]], -1)
    s = torch.sin(th)
    scale = torch.where(s > 1e-6, th / (2 * s.clamp_min(1e-12)), torch.full_like(s, 0.5))
    return w * scale[..., None]


def relative_pose_t(rq, pq, rm, pm, has):
    """Torch version of worldmodel.relative_pose: [..., POSE_DIM]."""
    rqt = rq.transpose(-1, -2)
    rot = so3_log_t(rqt @ rm)
    pos = (rqt @ (pm - pq)[..., None])[..., 0]
    h = has[..., None].float().expand(rot.shape[:-1] + (1,))
    return torch.cat([rot, pos * h, h, torch.ones_like(h)], -1)


# ---------------------------------------------------------------------------------------
# Diffusion

def precond(sigma):
    s = sigma[..., None, None, None]
    c_skip = wm.SIGMA_DATA ** 2 / (s ** 2 + wm.SIGMA_DATA ** 2)
    c_out = s * wm.SIGMA_DATA / (s ** 2 + wm.SIGMA_DATA ** 2).sqrt()
    c_in = 1 / (s ** 2 + wm.SIGMA_DATA ** 2).sqrt()
    return c_skip, c_out, c_in, sigma.log() / 4


FLOW_CTX_T = 0.05     # flow matching: noise level of history frames when dreaming
FLOW_SHIFT = 3.0      # flow matching: sampling schedule spends more steps at high noise


def _objective(model):
    m = getattr(model, "_orig_mod", getattr(model, "model", model))
    return getattr(m, "objective", "edm")


def flow_loss(model, x0, act, ev, pose, mem=None, mem_pose=None, own=None, own_mask=None,
              **scene):
    """Flow matching with per-frame noise (diffusion forcing): x_t = (1-t) x0 + t noise, the
    network predicts the velocity (x_t - x0) / t. Half the clips keep a nearly clean history,
    like dreaming; those history frames are context only (predicting the noise of an almost
    clean frame teaches nothing). Self-forced frames: the model's own drawings stand in for
    the real frames as inputs, and the real frames stay the targets."""
    b, t = x0.shape[:2]
    dev = x0.device
    tt = torch.sigmoid(torch.randn(b, t, device=dev))          # logit-normal noise levels
    hist = torch.rand(b, device=dev) < 0.5
    n_new = torch.randint(1, 4, (b,), device=dev)
    is_hist = (torch.arange(t, device=dev)[None] < (t - n_new)[:, None]) & hist[:, None]
    tt = torch.where(is_hist, torch.rand(b, t, device=dev) * 3 * FLOW_CTX_T, tt)
    base = x0
    if own is not None:
        lo, hi = SELF_SIGMA
        tt = torch.where(own_mask, 0.25 + torch.rand(b, t, device=dev) * 0.5, tt)
        base = torch.where(own_mask[..., None, None, None], own, x0)
    t5 = tt[..., None, None, None]
    xt = (1 - t5) * base + t5 * torch.randn_like(x0)
    target = (xt - x0) / t5.clamp_min(0.02)
    pred = model(xt, (tt - 0.5) * 2, act, ev, pose, mem, mem_pose, **scene)
    w = (~is_hist).float()
    if own is not None:
        w = torch.where(own_mask, torch.ones_like(w), w)
    err = (pred.float() - target.float()).pow(2).mean(dim=(2, 3, 4))
    return (err * w).sum() / w.sum().clamp_min(1)


def training_loss(model, x0, act, ev, pose, mem=None, mem_pose=None, own=None, own_mask=None,
                  **scene):
    if _objective(model) == "flow":
        return flow_loss(model, x0, act, ev, pose, mem, mem_pose, own, own_mask, **scene)
    return edm_loss(model, x0, act, ev, pose, mem, mem_pose, own, own_mask)


def edm_loss(model, x0, act, ev, pose, mem=None, mem_pose=None, own=None, own_mask=None):
    """Diffusion forcing over a clip x0 [B, T, ...]: every frame gets its own noise level and
    every frame is a target. For half the clips, the history is kept nearly clean (as when
    dreaming) and only the newest frames are heavily noised. `own` / `own_mask` [B, T]:
    frames the model generated itself (self-forcing) are used as inputs, lightly noised, and
    the real frames remain the targets."""
    b, t = x0.shape[:2]
    dev = x0.device
    sigma = (torch.randn(b, t, device=dev) * wm.P_STD + wm.P_MEAN).exp()
    hist = torch.rand(b, device=dev) < 0.5
    n_new = torch.randint(1, 4, (b,), device=dev)
    is_hist = torch.arange(t, device=dev)[None] < (t - n_new)[:, None]
    low = torch.rand(b, t, device=dev) * wm.CTX_SIGMA_MAX
    sigma = torch.where(hist[:, None] & is_hist, low, sigma)
    base = x0
    if own is not None:
        lo, hi = SELF_SIGMA
        sigma = torch.where(own_mask, lo + torch.rand(b, t, device=dev) * (hi - lo), sigma)
        base = torch.where(own_mask[..., None, None, None], own, x0)
    c_skip, c_out, c_in, c_noise = precond(sigma)
    x = base + sigma[..., None, None, None] * torch.randn_like(x0)
    pred = model(c_in * x, c_noise, act, ev, pose, mem, mem_pose)
    target = (x0 - c_skip * x) / c_out
    return F.mse_loss(pred.float(), target.float())


@torch.no_grad()
def sample_flow(model, hist, act, ev, pose, mem=None, mem_pose=None, steps=4, guidance=1.0,
                **scene):
    b, n = hist.shape[:2]
    dev = hist.device
    hist = (1 - FLOW_CTX_T) * hist + FLOW_CTX_T * torch.randn_like(hist)
    u = torch.linspace(1, 0, steps + 1, device=dev)
    ts = FLOW_SHIFT * u / (1 + (FLOW_SHIFT - 1) * u)
    x = torch.randn(b, *hist.shape[2:], device=dev)
    null = wm.null_event(model)
    for i in range(steps):
        tv = torch.cat([torch.full((b, n), FLOW_CTX_T, device=dev), ts[i].expand(b, 1)], 1)
        inp = torch.cat([hist, x[:, None]], 1)
        v = model(inp, (tv - 0.5) * 2, act, ev, pose, mem, mem_pose, **scene)[:, -1].float()
        if guidance != 1.0:
            act_u, ev_u = act.clone(), ev.clone()
            act_u[:, -1] = 0
            ev_u[:, -1] = null
            vu = model(inp, (tv - 0.5) * 2, act_u, ev_u, pose, mem, mem_pose, **scene)[:, -1].float()
            v = vu + guidance * (v - vu)
        x = x + (ts[i + 1] - ts[i]) * v
    return x


@torch.no_grad()
def sample_next(model, hist, act, ev, pose, mem=None, mem_pose=None, steps=3, ctx_noise=0.05,
                guidance=1.0, sigma_min=0.002, sigma_max=5.0, rho=7.0, **scene):
    """hist [B, W-1, ...] past frames; act / ev / pose [B, W] for every frame of the window
    including the new one (pose relative to the new frame). Returns the new frame [B, ...]."""
    if _objective(model) == "flow":
        return sample_flow(model, hist, act, ev, pose, mem, mem_pose, steps, guidance, **scene)
    b, n = hist.shape[:2]
    dev = hist.device
    hist = hist + ctx_noise * torch.randn_like(hist)
    ramp = torch.linspace(0, 1, steps, device=dev)
    sig = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros(1, device=dev)])
    x = torch.randn(b, *hist.shape[2:], device=dev) * sig[0]
    null = wm.null_event(model)
    for i in range(steps):
        sigma = torch.cat([torch.full((b, n), ctx_noise, device=dev), sig[i].expand(b, 1)], 1)
        c_skip, c_out, c_in, c_noise = precond(sigma)
        inp = torch.cat([hist, x[:, None]], 1) * c_in
        out = model(inp, c_noise, act, ev, pose, mem, mem_pose)[:, -1].float()
        if guidance != 1.0:
            act_u, ev_u = act.clone(), ev.clone()
            act_u[:, -1] = 0
            ev_u[:, -1] = null
            un = model(inp, c_noise, act_u, ev_u, pose, mem, mem_pose)[:, -1].float()
            out = un + guidance * (out - un)
        den = c_skip[:, -1] * x + c_out[:, -1] * out
        x = x + (x - den) / sig[i] * (sig[i + 1] - sig[i])
    return x


# ---------------------------------------------------------------------------------------
# One dreaming API for both architectures

class Rollout:
    """Dream frame by frame from a real starting moment of a take, with either the v2 model
    (worldmodel.WorldDiT) or v3 (CausalWorldDiT). `net` is the model or a GraphedModel."""

    def __init__(self, model, take, start, device, net=None):
        self.model, self.net, self.dev = model, net or model, device
        self.causal = model.config.get("arch") in ("causal", "world")
        self.scene = {}
        if model.config.get("n_rooms"):
            self.scene = {"room": torch.tensor([take.get("room", 0)], device=device),
                          "light": torch.tensor([take.get("light", 0)], device=device)}
        self.take, self.has = take, float(take["has"])
        c = model.context
        lat = take["lat"]
        self.frames = [torch.from_numpy(lat[j]).to(device) for j in range(start, start + c)]
        self.acts = [take["act"][j - 1] if j > 0 else np.zeros(wm.ACTION_DIM, np.float32)
                     for j in range(start, start + c)]
        self.evs = [int(take["ev"][j - 1]) if j > 0 else 0 for j in range(start, start + c)]
        self.poses = [(take["R"][j], take["p"][j]) for j in range(start, start + c)]

    @property
    def pose(self):
        return self.poses[-1]

    def step(self, action, event, r_new, p_new, steps=3, guidance=1.0, mem=None, mem_pose=None):
        a = torch.from_numpy(np.asarray(action, np.float32))[None].to(self.dev)
        if not self.causal:
            ctx = torch.stack(self.frames[-self.model.context:])[None]
            e = torch.tensor([event], device=self.dev)
            z = wm.sample_next(self.net, ctx, a, e, steps=steps, mem=mem, mem_pose=mem_pose,
                               guidance=guidance)[0]
        else:
            n = self.model.window - 1
            hist = torch.stack(self.frames[-n:])[None]
            acts = np.stack(self.acts[-n:] + [np.asarray(action, np.float32)])
            evs = self.evs[-n:] + [int(event)]
            rel = [wm.relative_pose(r_new, p_new, r, p, self.has) for r, p in self.poses[-n:]]
            rel.append(wm.relative_pose(r_new, p_new, r_new, p_new, self.has))
            z = sample_next(self.net, hist, torch.from_numpy(acts)[None].to(self.dev),
                            torch.tensor([evs], device=self.dev),
                            torch.from_numpy(np.stack(rel))[None].to(self.dev),
                            mem, mem_pose, steps=steps, guidance=guidance, **self.scene)[0]
        self.frames.append(z.float())
        self.acts.append(np.asarray(action, np.float32))
        self.evs.append(int(event))
        self.poses.append((r_new, p_new))
        keep = self.model.context
        self.frames, self.acts = self.frames[-keep:], self.acts[-keep:]
        self.evs, self.poses = self.evs[-keep:], self.poses[-keep:]
        return z
