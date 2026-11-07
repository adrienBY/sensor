"""AMD FidelityFX Super Resolution 1 (FSR 1) in PyTorch, for upscaling dream frames.

Two passes, ported from AMD's open-source ffx_fsr1.h (MIT licence):
  * EASU (Edge-Adaptive Spatial Upsampling): for every output pixel, reads a 12-tap
    neighbourhood, estimates the local edge direction and strength, and filters with a
    Lanczos-like kernel stretched along the edge, so edges stay crisp instead of blurring.
    The result is clamped to the 4 nearest input pixels to avoid ringing.
  * RCAS (Robust Contrast-Adaptive Sharpening): a 5-tap sharpen whose strength adapts to local
    contrast so it does not create halos.
Runs on the GPU on whole frames at once. Input/output: uint8 HxWx3 numpy (RGB).
"""
import numpy as np
import torch

# 12 taps around the sample's top-left input pixel f = (0, 0), as (dx, dy):
#        b c
#      e f g h
#      i j k l
#        n o
TAPS = {"b": (0, -1), "c": (1, -1), "e": (-1, 0), "f": (0, 0), "g": (1, 0), "h": (2, 0),
        "i": (-1, 1), "j": (0, 1), "k": (1, 1), "l": (2, 1), "n": (0, 2), "o": (1, 2)}


def _luma(c):
    """FSR's luma approximation on [3, ...] RGB: 0.5 R + G + 0.5 B (only ratios matter)."""
    return 0.5 * c[0] + c[1] + 0.5 * c[2]


def _set(dir_x, dir_y, length, w, la, lb, lc, ld, le):
    """FsrEasuSetF: accumulate direction and edge length from a plus-shaped 5-tap pattern
    (a up, b left, c centre, d right, e down) with bilinear weight w."""
    lenx = torch.maximum((ld - lc).abs(), (lc - lb).abs())
    dirx = ld - lb
    lenx = ((dirx.abs() / lenx.clamp_min(1e-5)).clamp(0, 1)) ** 2
    leny = torch.maximum((le - lc).abs(), (lc - la).abs())
    diry = le - la
    leny = ((diry.abs() / leny.clamp_min(1e-5)).clamp(0, 1)) ** 2
    return dir_x + dirx * w, dir_y + diry * w, length + (lenx + leny) * w


@torch.no_grad()
def easu(img, out_w, out_h):
    """img: float [3, H, W] in 0..1 on any device -> [3, out_h, out_w]."""
    _, h, w = img.shape
    dev = img.device
    pad = torch.nn.functional.pad(img[None], (1, 2, 1, 2), mode="replicate")[0]
    ys = (torch.arange(out_h, device=dev, dtype=torch.float32) + 0.5) * (h / out_h) - 0.5
    xs = (torch.arange(out_w, device=dev, dtype=torch.float32) + 0.5) * (w / out_w) - 0.5
    py, px = torch.meshgrid(ys, xs, indexing="ij")
    fy, fx = py.floor(), px.floor()
    ty, tx = py - fy, px - fx                        # fraction inside the f-g-j-k square
    iy, ix = fy.long().clamp(-1, h - 1) + 1, fx.long().clamp(-1, w - 1) + 1   # into padded

    col = {n: pad[:, iy + dy, ix + dx] for n, (dx, dy) in TAPS.items()}
    lum = {n: _luma(c) for n, c in col.items()}

    z = torch.zeros_like(tx)
    dx_, dy_, ln = z, z, z
    dx_, dy_, ln = _set(dx_, dy_, ln, (1 - tx) * (1 - ty), lum["b"], lum["e"], lum["f"], lum["g"], lum["j"])
    dx_, dy_, ln = _set(dx_, dy_, ln, tx * (1 - ty), lum["c"], lum["f"], lum["g"], lum["h"], lum["k"])
    dx_, dy_, ln = _set(dx_, dy_, ln, (1 - tx) * ty, lum["f"], lum["i"], lum["j"], lum["k"], lum["n"])
    dx_, dy_, ln = _set(dx_, dy_, ln, tx * ty, lum["g"], lum["j"], lum["k"], lum["l"], lum["o"])

    d2 = dx_ * dx_ + dy_ * dy_
    flat = d2 < 1.0 / 32768.0
    inv = torch.where(flat, torch.ones_like(d2), torch.rsqrt(d2.clamp_min(1e-12)))
    dx_ = torch.where(flat, torch.ones_like(dx_), dx_) * inv
    dy_ = dy_ * inv
    ln = (ln * 0.5) ** 2                              # 0 = flat area, 1 = strong edge
    stretch = 1.0 / torch.maximum(dx_.abs(), dy_.abs()).clamp_min(1e-5)
    len_x = 1 + (stretch - 1) * ln                    # widen along the edge
    len_y = 1 - 0.5 * ln                              # narrow across it
    lob = 0.5 + (0.25 - 0.04 - 0.5) * ln              # negative lobe strength
    clp = 1.0 / lob

    acc = torch.zeros_like(img[:, :1, :1]).expand(3, out_h, out_w).clone()
    wsum = torch.zeros_like(tx)
    for n, (ox, oy) in TAPS.items():
        offx, offy = ox - tx, oy - ty
        vx = (offx * dx_ + offy * dy_) * len_x       # rotate into the edge frame, then scale
        vy = (offx * -dy_ + offy * dx_) * len_y
        dd = torch.minimum(vx * vx + vy * vy, clp)
        wb = (0.4 * dd - 1) ** 2
        wa = (lob * dd - 1) ** 2
        wgt = (25.0 / 16.0 * wb - (25.0 / 16.0 - 1)) * wa
        acc += col[n] * wgt
        wsum += wgt
    out = acc / wsum.clamp_min(1e-6)
    lo = torch.minimum(torch.minimum(col["f"], col["g"]), torch.minimum(col["j"], col["k"]))
    hi = torch.maximum(torch.maximum(col["f"], col["g"]), torch.maximum(col["j"], col["k"]))
    return torch.minimum(torch.maximum(out, lo), hi)  # de-ring


@torch.no_grad()
def rcas(img, sharpness=0.25):
    """img: float [3, H, W] 0..1. sharpness in stops: 0 = strongest, 2 = mild."""
    p = torch.nn.functional.pad(img[None], (1, 1, 1, 1), mode="replicate")[0]
    b, d, e, f, hh = p[:, :-2, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 1:-1], p[:, 1:-1, 2:], p[:, 2:, 1:-1]
    mn4 = torch.minimum(torch.minimum(b, d), torch.minimum(f, hh))
    mx4 = torch.maximum(torch.maximum(b, d), torch.maximum(f, hh))
    hit_min = torch.minimum(mn4, e) / (4 * mx4).clamp_min(1e-5)
    hit_max = (1 - torch.maximum(mx4, e)) / (4 * mn4 - 4).clamp_max(-1e-5)
    lobe_rgb = torch.maximum(-hit_min, hit_max)
    lobe = lobe_rgb.max(dim=0, keepdim=True).values.clamp(-0.1875, 0) * (2.0 ** -sharpness)
    return ((lobe * (b + d + f + hh) + e) / (4 * lobe + 1)).clamp(0, 1)


def upscale(frame, out_w=1280, out_h=720, device=None, sharpness=0.25):
    """uint8 HxWx3 RGB numpy -> uint8 out_h x out_w x 3, with FSR 1 (EASU + RCAS)."""
    t = torch.from_numpy(frame).to(device or "cpu").permute(2, 0, 1).float() / 255.0
    t = rcas(easu(t, out_w, out_h), sharpness)
    return (t * 255 + 0.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
