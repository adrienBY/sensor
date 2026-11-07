#!/usr/bin/env python3
"""Put all takes of a room into one shared map, so memory can use every take.

    python pipeline/align_takes.py data/processed_512x288

Each take's camera path starts at its own origin. To join them:
  1. Moments where two takes saw the same view are found by comparing their encoded frames
     (a small pooled descriptor of each latent, best matches in both directions).
  2. Each matched pair gives an estimate of how one take's coordinates sit relative to the
     other: if frame i of take A and frame j of take B show the same view, the two cameras
     were in about the same place, so T(B->A) = pose_A(i) * pose_B(j)^-1.
  3. A robust vote (RANSAC) keeps the rotation most pairs agree with (within 15 degrees) and
     discards false matches such as blank walls. Rotation = average of the agreeing pairs;
     position = median of the most similar quarter of them (similar-looking views can be
     taken a metre apart, the most similar ones rarely are).
  4. Per room, takes are chained to the best-connected take along the strongest links.
The result goes to map.json next to index.json. Training, dream videos and play mode read it:
poses are moved into the shared map and memory searches all aligned takes of the room.
Takes that cannot be linked keep their own coordinates (memory then stays within the take).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import worldmodel as wm

MIN_INLIERS = 8
MIN_RATIO = 0.2
ANGLE_TOL = np.radians(15)
EVERY = 3                    # use every 3rd frame for matching
MATCH = 0.6                  # descriptor similarity for a candidate match


def descriptors(lat):
    """[T, C, H, W] latents -> unit descriptors [T, D] and a texture score per frame."""
    x = torch.from_numpy(np.asarray(lat, np.float32))
    x = F.adaptive_avg_pool2d(x, (9, 16)).flatten(1)
    x = x - x.mean(1, keepdim=True)
    tex = x.std(1)
    return F.normalize(x, dim=1).numpy(), tex.numpy()


def mean_rotation(rs):
    u, _, vt = np.linalg.svd(np.sum(rs, axis=0))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def relate(ta, tb, rng):
    """Transform taking take B's coordinates into take A's, or None. Returns (R, p, inliers,
    pairs)."""
    ia, ib = np.arange(0, len(ta["lat"]), EVERY), np.arange(0, len(tb["lat"]), EVERY)
    da, xa = descriptors(ta["lat"][ia])
    db, xb = descriptors(tb["lat"][ib])
    keep_a, keep_b = xa > np.percentile(xa, 30), xb > np.percentile(xb, 30)   # skip blank walls
    s = da @ db.T
    s[~keep_a] = -1
    s[:, ~keep_b] = -1
    # best match in each direction (not mutual: neighbouring frames look alike, so a best
    # match rarely points back to the exact same frame); RANSAC removes the wrong ones
    best_b, best_a = s.argmax(1), s.argmax(0)
    pairs = {(i, int(best_b[i])) for i in range(len(ia)) if s[i, best_b[i]] > MATCH}
    pairs |= {(int(best_a[j]), j) for j in range(len(ib)) if s[best_a[j], j] > MATCH}
    pairs = sorted(pairs)
    if len(pairs) < MIN_INLIERS:
        return None
    use_t = ta["has"] and tb["has"]
    sims = np.array([s[i, j] for i, j in pairs])
    cands = []
    for i, j in pairs:
        fa, fb = ia[i], ib[j]
        r = ta["R"][fa] @ tb["R"][fb].T
        p = ta["p"][fa] - r @ tb["p"][fb] if use_t else np.zeros(3)
        cands.append((r, p))
    rs = np.stack([c[0] for c in cands])
    ps = np.stack([c[1] for c in cands])
    best = None
    for k in rng.choice(len(cands), size=min(300, len(cands)), replace=False):
        ang = np.arccos(np.clip((np.einsum("ij,nij->n", rs[k], rs) - 1) / 2, -1, 1))
        ok = ang < ANGLE_TOL                  # the vote is on rotation, which views pin down well
        if best is None or ok.sum() > best.sum():
            best = ok
    n = int(best.sum())
    if n < MIN_INLIERS or n < MIN_RATIO * len(cands):
        return None
    # Position: a wall looks alike from 1 m or 3 m away, so only the most similar agreeing
    # matches are trusted (top quarter, median). On a test with a known answer this cut the
    # error from 1.7 m (all matches) to 0.36 m.
    top = best & (sims >= np.percentile(sims[best], 75))
    if top.sum() < 3:
        top = best
    return mean_rotation(rs[best]), np.median(ps[top], axis=0), n, len(cands)


def build_map(data):
    rng = np.random.default_rng(0)
    rooms = {}
    for t in data.takes:
        rooms.setdefault(t["room"], []).append(t)
    out = {"version": 1, "rooms": {}}
    for room, takes in rooms.items():
        name = wm.ROOMS[room - 1] if room else "unknown room"
        print(f"== {name}: {len(takes)} takes")
        links = {}
        for a in range(len(takes)):
            for b in range(a + 1, len(takes)):
                rel = relate(takes[a], takes[b], rng)
                if rel:
                    r, p, n, m = rel
                    links[(a, b)] = (r, p, n)
                    links[(b, a)] = (r.T, -r.T @ p, n)
                    print(f"  {takes[a]['name'][-34:]} <-> {takes[b]['name'][-34:]}: "
                          f"{n} agreeing matches of {m}")
        weight = [sum(n for (x, _), (_, _, n) in links.items() if x == a) for a in range(len(takes))]
        ref = int(np.argmax(weight)) if takes else 0
        placed = {ref: (np.eye(3), np.zeros(3), None, 0)}
        while True:                                   # strongest link from the placed set
            cand = [(n, a, b) for (a, b), (_, _, n) in links.items() if a in placed and b not in placed]
            if not cand:
                break
            n, a, b = max(cand)
            ra, pa = placed[a][:2]
            r, p, _ = links[(a, b)]                   # b -> a
            placed[b] = (ra @ r, ra @ p + pa, takes[a]["name"], n)
        entry = {"reference": takes[ref]["name"], "takes": {}, "unaligned": []}
        for i, t in enumerate(takes):
            if i in placed:
                r, p, via, n = placed[i]
                entry["takes"][t["name"]] = {"R": r.tolist(), "p": p.tolist(), "via": via,
                                             "matches": n}
            else:
                entry["unaligned"].append(t["name"])
        print(f"  map: {len(entry['takes'])} takes aligned, {len(entry['unaligned'])} on their own")
        out["rooms"][name] = entry
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("processed", type=Path)
    args = ap.parse_args()
    data = wm.Dataset(args.processed, use_map=False)
    if not data.takes:
        raise SystemExit("No encoded takes yet.")
    m = build_map(data)
    (args.processed / "map.json").write_text(json.dumps(m, indent=1))
    print(f"Saved {args.processed / 'map.json'}")


if __name__ == "__main__":
    main()
