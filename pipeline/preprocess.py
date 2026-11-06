#!/usr/bin/env python3
"""Turn raw Sensor Capture takes into training-ready arrays (CPU step).

    python pipeline/preprocess.py RAW_DIR OUT_DIR [--width 256] [--height 144] [--fps 15]

RAW_DIR  the SensorCapture folder copied from the phone (one folder per take)
OUT_DIR  e.g. data/processed. Gets one folder per take, plus index.json.

What it does for each take:
  1. Decodes the video once: keeps frames at the target fps and size, and measures blur.
  2. Finds each video frame's exact timestamp on the phone clock by matching motion blur
     to gyroscope speed (blur appears exactly when the phone turns).
  3. Computes the action between consecutive kept frames: camera rotation (ARCore pose or
     gyroscope) and, in AR mode, camera translation in metres.
  4. Maps interaction tags onto the steps where they happened.

Output per take (OUT_DIR/<take id>/), T = number of kept frames:
  frames.npy      uint8   [T, H, W, 3]  RGB
  timestamps.npy  int64   [T]           phone clock (elapsedRealtime ns)
  sharpness.npy   float32 [T]           Laplacian variance; low = blurry
  actions.npz     per step t -> t+1, T-1 rows:
      rot        float32 [T-1, 3]  rotation, axis-angle (rad) in the camera frame of frame t
      trans      float32 [T-1, 3]  translation (m) in the camera frame of frame t (AR only, else 0)
      has_trans  bool    [T-1]     translation is valid (AR mode, tracked at both ends)
      event      int16   [T-1]     interaction tag during the step (0 = none, see index.json)
  take.json       source metadata, alignment offset and confidence, counts

Camera frame convention (same as ARCore): +X right, +Y up, +Z backwards (the camera looks
along -Z).

Needs: numpy, opencv-python.
"""
import argparse
import csv
import json
import math
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

FORMAT = "sensor-capture/1"
PIPELINE_VERSION = 1


# ---------------------------------------------------------------------------------------
# Small rotation helpers (quaternions are x, y, z, w)

def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_rotvec(r):
    cos = np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)
    angle = math.acos(cos)
    if angle < 1e-8:
        return np.zeros(3)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    s = math.sin(angle)
    if s < 1e-6:  # ~180 degrees, not expected between consecutive frames
        return np.zeros(3)
    return v / (2 * s) * angle


# ---------------------------------------------------------------------------------------
# Reading a take

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_imu(path):
    gyro_t, gyro, grav = [], [], []
    for r in read_csv(path):
        if r["sensor"] == "gyro":
            gyro_t.append(int(r["timestamp_ns"]))
            gyro.append((float(r["x"]), float(r["y"]), float(r["z"])))
        elif r["sensor"] == "grav":
            grav.append((float(r["x"]), float(r["y"]), float(r["z"])))
    order = np.argsort(gyro_t)
    return (np.array(gyro_t, dtype=np.int64)[order],
            np.array(gyro, dtype=np.float64)[order],
            np.array(grav, dtype=np.float64))


def load_poses(path):
    rows = read_csv(path)
    t = np.array([int(r["timestamp_ns"]) for r in rows], dtype=np.int64)
    tracked = np.array([r["tracking"] == "TRACKING" for r in rows])
    p = np.array([[float(r[k]) for k in ("tx", "ty", "tz")] for r in rows])
    q = np.array([[float(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in rows])
    return t, tracked, p, q


def crop_to_aspect(img, aspect):
    h, w = img.shape[:2]
    if w / h > aspect:          # too wide: crop the sides
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = int(round(w / aspect))  # too tall: crop top and bottom
    y0 = (h - nh) // 2
    return img[y0:y0 + nh]


def decode(video_path, out_w, out_h, fps):
    """One pass over the video: blur score for every frame, resized RGB for kept frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    pts, sharp, kept_idx, kept = [], [], [], []
    step = 1.0 / fps
    next_t = 0.0
    i = 0
    aspect = out_w / out_h
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if i > 0 and t <= pts[-1]:            # some files report 0 for the last frames
            t = pts[-1] + 1.0 / 30
        pts.append(t)
        small = cv2.resize(frame, (640, 360) if frame.shape[1] >= frame.shape[0] else (360, 640),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        if t + 1e-6 >= next_t:
            img = crop_to_aspect(frame, aspect)
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            kept.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            kept_idx.append(i)
            next_t += step
            while next_t <= t:
                next_t += step
        i += 1
    cap.release()
    return np.array(pts), np.array(sharp), np.array(kept_idx), np.stack(kept) if kept else None


# ---------------------------------------------------------------------------------------
# Alignment: video frame times -> phone clock

def align(pts, sharp, gyro_t, gyro, guess_ns, window_s=3.0):
    """Find offset (ns) so that phone_time = offset + pts, by matching blur to rotation speed."""
    speed = np.linalg.norm(gyro, axis=1)
    # Smooth over ~one exposure (30 ms) so the speed matches what blurs a frame.
    k = 12
    speed = np.convolve(speed, np.ones(k) / k, mode="same")
    blur = -np.log(np.maximum(sharp, 1e-3))
    blur = (blur - blur.mean()) / (blur.std() + 1e-9)
    pts_ns = (pts * 1e9).astype(np.int64)

    def score(offset):
        t = offset + pts_ns
        inside = (t >= gyro_t[0]) & (t <= gyro_t[-1])
        if inside.mean() < 0.8:
            return -1.0
        s = np.interp(t[inside], gyro_t, speed)
        b = blur[inside]
        if s.std() < 1e-9:
            return -1.0
        return float(np.corrcoef(s, b)[0, 1])

    best_off, best = guess_ns, -2.0
    for step_ms, half_s in ((10, window_s), (1, 0.05)):
        centre = best_off
        n = int(half_s * 1000 / step_ms)
        for j in range(-n, n + 1):
            off = centre + j * step_ms * 1_000_000
            c = score(off)
            if c > best:
                best, best_off = c, off
    return int(best_off), best


# ---------------------------------------------------------------------------------------

def camera_from_device(grav):
    """Rotation taking device-frame vectors into the camera frame (+X right, +Y up, +Z back)."""
    up = grav.mean(axis=0)            # gravity sensor points 'up' in device coordinates
    z = np.array([0.0, 0.0, 1.0])     # back camera looks along -z of the device
    y = up - up.dot(z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z])        # rows = camera axes expressed in device coordinates


def gyro_rotations(t_keep, gyro_t, gyro, r_cd):
    """Integrated gyroscope rotation between consecutive kept frames, in the camera frame."""
    out = np.zeros((len(t_keep) - 1, 3))
    dt = np.diff(gyro_t) / 1e9
    w_mid = (gyro[1:] + gyro[:-1]) / 2
    cum = np.vstack([np.zeros(3), np.cumsum(w_mid * dt[:, None], axis=0)])
    for c in range(3):
        a = np.interp(t_keep, gyro_t, cum[:, c])
        out[:, c] = np.diff(a)
    return out @ r_cd.T


def pose_steps(t_keep, poses, tol_ns=25_000_000):
    pt, tracked, p, q = poses
    n = len(t_keep) - 1
    if len(pt) < 2:
        return np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n, dtype=bool)
    idx = np.clip(np.searchsorted(pt, t_keep), 1, len(pt) - 1)
    left = idx - 1
    nearest = np.where(np.abs(pt[left] - t_keep) <= np.abs(pt[idx] - t_keep), left, idx)
    ok = (np.abs(pt[nearest] - t_keep) <= tol_ns) & tracked[nearest]
    rot, trans, valid = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n, dtype=bool)
    for i in range(n):
        a, b = nearest[i], nearest[i + 1]
        if not (ok[i] and ok[i + 1]):
            continue
        ra, rb = quat_to_mat(q[a]), quat_to_mat(q[b])
        trans[i] = ra.T @ (p[b] - p[a])
        rot[i] = mat_to_rotvec(ra.T @ rb)
        valid[i] = True
    return rot, trans, valid


def process_take(take_dir, out_root, args, vocab):
    meta = json.loads((take_dir / "meta.json").read_text())
    take_id = meta.get("session", take_dir.name)
    if meta.get("format", FORMAT) != FORMAT:
        print(f"  skip: unknown format {meta.get('format')}")
        return None
    out = out_root / take_id
    if (out / "take.json").exists() and not args.force:
        info = json.loads((out / "take.json").read_text())
        if info.get("pipeline_version") == PIPELINE_VERSION:
            print("  already processed (use --force to redo)")
            return info

    gyro_t, gyro, grav = load_imu(take_dir / "imu.csv")
    if len(gyro_t) < 100:
        print("  skip: no gyroscope data")
        return None

    pts, sharp, kept_idx, frames = decode(take_dir / "video.mp4", args.width, args.height, args.fps)
    if frames is None or len(kept_idx) < 2:
        print("  skip: video has no frames")
        return None

    guess = meta.get("video_start_ns", -1)
    if guess is None or guess <= 0:
        guess = meta.get("record_call_ns", int(gyro_t[0]))
    offset, conf = align(pts, sharp, gyro_t, gyro, int(guess))
    if conf < 0.3:
        print(f"  WARNING: weak alignment (corr {conf:.2f}); using it anyway, check this take")

    t_keep = offset + (pts[kept_idx] * 1e9).astype(np.int64)

    r_cd = camera_from_device(grav) if len(grav) else np.eye(3)
    rot = gyro_rotations(t_keep, gyro_t, gyro, r_cd)
    trans = np.zeros_like(rot)
    has_trans = np.zeros(len(rot), dtype=bool)
    pose_agreement = None
    if (take_dir / "poses.csv").exists():
        prot, ptrans, valid = pose_steps(t_keep, load_poses(take_dir / "poses.csv"))
        if valid.any():
            # Diagnostic: gyro and ARCore should agree on rotation.
            a, b = rot[valid].ravel(), prot[valid].ravel()
            if a.std() > 1e-6 and b.std() > 1e-6:
                pose_agreement = float(np.corrcoef(a, b)[0, 1])
            rot[valid] = prot[valid]
        trans, has_trans = ptrans, valid

    # Interaction tags -> step index
    event = np.zeros(len(rot), dtype=np.int16)
    ev_path = take_dir / "events.json"
    if ev_path.exists():
        for e in json.loads(ev_path.read_text()).get("events", []):
            key = f"{e.get('object')}:{e.get('action')}"
            if key not in vocab:
                vocab.append(key)
            step = int(np.searchsorted(t_keep, int(e["t_ns"]), side="right")) - 1
            if 0 <= step < len(event):
                event[step] = vocab.index(key) + 1

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frames.npy", frames)
    np.save(out / "timestamps.npy", t_keep)
    np.save(out / "sharpness.npy", sharp[kept_idx].astype(np.float32))
    np.savez(out / "actions.npz", rot=rot.astype(np.float32), trans=trans.astype(np.float32),
             has_trans=has_trans, event=event)

    blurry = sharp[kept_idx] < 0.35 * np.median(sharp)
    info = {
        "pipeline_version": PIPELINE_VERSION,
        "take": take_id,
        "room": meta.get("room"),
        "lighting": meta.get("lighting"),
        "take_type": meta.get("take_type"),
        "mode": meta.get("mode", "video"),
        "frames": int(len(kept_idx)),
        "fps": args.fps,
        "size": [args.width, args.height],
        "duration_s": float(pts[-1] - pts[0]),
        "alignment": {"offset_ns": offset, "correlation": round(conf, 3),
                      "guess_ns": int(guess), "shift_ms": round((offset - guess) / 1e6, 1)},
        "translation_steps": int(has_trans.sum()),
        "gyro_vs_arcore_rotation_corr": pose_agreement,
        "blurry_frames": int(blurry.sum()),
        "events": int((event > 0).sum()),
        "camera_from_device": r_cd.round(4).tolist(),
    }
    (out / "take.json").write_text(json.dumps(info, indent=2))
    print(f"  {info['frames']} frames @ {args.fps} fps, aligned (corr {conf:.2f}, "
          f"shift {info['alignment']['shift_ms']} ms), {info['translation_steps']} steps with "
          f"translation, {info['blurry_frames']} blurry, {info['events']} tags")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", type=Path, help="SensorCapture folder from the phone")
    ap.add_argument("out", type=Path, help="output folder, e.g. data/processed")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=144)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--val-every", type=int, default=10,
                    help="put roughly 1 take in N into the validation split (by take id)")
    ap.add_argument("--force", action="store_true", help="reprocess takes already done")
    args = ap.parse_args()

    takes = sorted({p.parent for p in args.raw.rglob("meta.json")})
    if not takes:
        sys.exit(f"No takes found under {args.raw}")
    args.out.mkdir(parents=True, exist_ok=True)

    index_path = args.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    vocab = index.get("event_vocab", [])

    entries = []
    for t in takes:
        print(f"== {t.name}")
        try:
            info = process_take(t, args.out, args, vocab)
        except Exception as e:  # keep going; one bad take should not stop the batch
            print(f"  FAILED: {e}")
            continue
        if info:
            split = "val" if zlib.crc32(info["take"].encode()) % args.val_every == 0 else "train"
            entries.append({k: info[k] for k in
                            ("take", "room", "lighting", "take_type", "mode", "frames",
                             "duration_s")} | {"split": split})

    index = {
        "format": FORMAT,
        "pipeline_version": PIPELINE_VERSION,
        "fps": args.fps,
        "size": [args.width, args.height],
        "event_vocab": vocab,
        "takes": entries,
    }
    index_path.write_text(json.dumps(index, indent=2))
    total = sum(e["duration_s"] for e in entries) / 60
    n_val = sum(e["split"] == "val" for e in entries)
    print(f"\n{len(entries)} takes ready ({n_val} validation), {total:.1f} min. Index: {index_path}")


if __name__ == "__main__":
    main()
