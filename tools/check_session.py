#!/usr/bin/env python3
"""Sanity-check Sensor Capture sessions after copying them to the PC.

Usage:
    python tools/check_session.py path/to/SensorCapture            # every session in the folder
    python tools/check_session.py path/to/SensorCapture/20261107_...  # one session

Needs only the Python standard library. If ffprobe (from FFmpeg) is on the PATH it also
compares the video's frame count with the camera frame log.
"""
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REQUIRED = ["video.mp4", "imu.csv", "frames.csv", "events.json", "meta.json"]


def video_info(path: Path):
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=width,height,nb_read_packets,avg_frame_rate,duration",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
        return json.loads(out)["streams"][0]
    except Exception:
        return None


def check(session: Path) -> bool:
    print(f"\n== {session.name}")
    missing = [f for f in REQUIRED if not (session / f).exists()]
    if missing:
        print(f"  MISSING: {', '.join(missing)}")
        return False

    meta = json.loads((session / "meta.json").read_text())
    print(f"  room={meta.get('room')}  light={meta.get('lighting')}  take={meta.get('take_type')}")
    dur = meta.get("video_duration_s", 0)
    print(f"  video duration {dur:.1f} s, error code {meta.get('video_error', 0)}")
    ok = meta.get("video_error", 0) == 0

    # Camera frames
    ts = []
    with open(session / "frames.csv", newline="") as f:
        for row in csv.DictReader(f):
            ts.append(int(row["timestamp_ns"]))
    if len(ts) > 2:
        dts = [(b - a) / 1e6 for a, b in zip(ts, ts[1:])]
        med = statistics.median(dts)
        drops = sum(1 for d in dts if d > 1.5 * med)
        print(f"  camera frames {len(ts)}, median interval {med:.1f} ms "
              f"({1000 / med:.1f} fps), gaps {drops}")
        if drops > 0.02 * len(dts):
            print("  WARNING: more than 2% of frames have gaps")
    else:
        print("  WARNING: frame log is empty")
        ok = False

    # IMU
    per = defaultdict(list)
    with open(session / "imu.csv", newline="") as f:
        for row in csv.DictReader(f):
            per[row["sensor"]].append(int(row["timestamp_ns"]))
    for name, t in sorted(per.items()):
        if len(t) < 2:
            continue
        span = (t[-1] - t[0]) / 1e9
        rate = (len(t) - 1) / span if span > 0 else 0
        gaps = sum(1 for a, b in zip(t, t[1:]) if (b - a) > 50e6)
        print(f"  imu {name:5s} {len(t):7d} samples  {rate:6.0f} Hz  gaps>50ms: {gaps}")
    if "gyro" not in per:
        print("  WARNING: no gyroscope samples")
        ok = False

    # Clock overlap between camera and IMU
    if ts and per.get("gyro"):
        g = per["gyro"]
        if g[0] > ts[0] + 1e9 or g[-1] < ts[-1] - 1e9:
            print("  WARNING: IMU does not cover the whole video (clock mismatch?)")
    src = meta.get("camera", {}).get("timestamp_source")
    print(f"  camera timestamp source: {src}")

    # ARCore poses (AR mode only)
    poses_path = session / "poses.csv"
    if poses_path.exists():
        rows = list(csv.DictReader(open(poses_path, newline="")))
        tracked = [r for r in rows if r["tracking"] == "TRACKING"]
        pct = 100 * len(tracked) / max(1, len(rows))
        path_m = 0.0
        prev = None
        for r in tracked:
            p = (float(r["tx"]), float(r["ty"]), float(r["tz"]))
            if prev:
                path_m += math.dist(prev, p)
            prev = p
        print(f"  ARCore poses {len(rows)}, tracking {pct:.0f}%, walked {path_m:.1f} m")
        if pct < 90:
            reasons = defaultdict(int)
            for r in rows:
                if r["tracking"] != "TRACKING":
                    reasons[r["failure_reason"]] += 1
            print(f"  WARNING: tracking lost on {100 - pct:.0f}% of frames: {dict(reasons)}")
        if tracked:
            xs = [float(r["tx"]) for r in tracked]
            zs = [float(r["tz"]) for r in tracked]
            print(f"  floor extent {max(xs) - min(xs):.1f} m x {max(zs) - min(zs):.1f} m")
    elif meta.get("mode") == "ar":
        print("  WARNING: AR session without poses.csv")
        ok = False

    # Events
    ev = json.loads((session / "events.json").read_text())
    tags = ev.get("events", [])
    if tags:
        kinds = defaultdict(int)
        for e in tags:
            kinds[f"{e['object']}:{e['action']}"] += 1
        top = ", ".join(f"{k}x{v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:8])
        print(f"  {len(tags)} interaction tags: {top}")

    # Video file vs frame log
    info = video_info(session / "video.mp4")
    if info:
        n = int(info.get("nb_read_packets", 0))
        print(f"  video {info.get('width')}x{info.get('height')}, {n} frames, "
              f"rate {info.get('avg_frame_rate')}")
        if ts and abs(n - len(ts)) > 15:
            print(f"  note: video has {n} frames vs {len(ts)} logged "
                  "(frames logged before the encoder started are expected)")
    print("  OK" if ok else "  PROBLEMS FOUND")
    return ok


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1])
    # Sessions can sit at any depth (room/take-type/session), so find them by their meta.json.
    sessions = sorted({p.parent for p in root.rglob("meta.json")})
    if not sessions:
        print(f"No sessions found under {root}")
        sys.exit(1)
    results = [check(s) for s in sessions]

    # Minutes per room and take type
    table = defaultdict(lambda: defaultdict(float))
    takes = set()
    total = 0.0
    for s in sessions:
        try:
            m = json.loads((s / "meta.json").read_text())
        except Exception:
            continue
        minutes = m.get("video_duration_s", 0) / 60
        table[m.get("room", "?")][m.get("take_type", "?")] += minutes
        takes.add(m.get("take_type", "?"))
        total += minutes
    cols = sorted(takes)
    print("\nMinutes of footage per room and take type")
    print(f"  {'room':16s}" + "".join(f"{c[:12]:>13s}" for c in cols) + f"{'total':>9s}")
    for room in sorted(table):
        row = table[room]
        print(f"  {room:16s}" + "".join(f"{row.get(c, 0):13.1f}" for c in cols)
              + f"{sum(row.values()):9.1f}")
    print(f"\n{sum(results)}/{len(results)} sessions OK, {total:.1f} min of footage")


if __name__ == "__main__":
    main()
