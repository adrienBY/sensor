# Sensor Capture

Android app that records training data for a learned world model of an apartment:
video, IMU, per-frame camera metadata and timestamped interaction tags, all on one clock.

Built by GitHub Actions on every push to `main`. The APK is attached to the latest release:
**https://github.com/adrienBY/sensor/releases/latest/download/sensor-capture.apk**

## What one take produces

Each take is saved to `Download/SensorCapture/<session>/` on the phone:

| File | Contents |
| --- | --- |
| `video.mp4` | 1080p H.264, ~20 Mbps, 30 fps fixed when supported, electronic stabilization off, no audio |
| `imu.csv` | `timestamp_ns,sensor,x,y,z,w` for `gyro` (rad/s), `accel` (m/s²), `grav` (m/s²), `rot` (game rotation vector quaternion x,y,z,w) at the fastest rate the phone allows |
| `frames.csv` | One row per camera frame: frame number, sensor timestamp, exposure, ISO, focus distance, focal length and per-frame intrinsics (fx, fy, cx, cy, skew) when the phone reports them |
| `events.json` | Interaction tags after undo: `{t_ns, t_video_s, object, action}` |
| `events.log.jsonl` | Raw tag log written as you tap, including undos (crash-safe) |
| `meta.json` | Room, lighting, take type, device, camera calibration, sensor list, clock anchors, counts |

Session names look like `20261107_101530_kitchen_walkthrough`.

## Clocks and alignment

Everything uses Android `elapsedRealtimeNanos`:

- IMU timestamps are `SensorEvent.timestamp` (elapsedRealtime on modern Android).
- Camera timestamps are `SENSOR_TIMESTAMP`; `meta.json → camera.timestamp_source` says `realtime`
  when they share the IMU clock.
- `meta.json → video_start_ns` is when CameraX reported the video started.

The camera frame log starts slightly before the encoder's first frame, so the mapping from
`frames.csv` rows to video frames is approximate (a frame or two). The PC pipeline refines it by
matching gyro rotation to visual rotation.

## On the PC

```
python tools/check_session.py D:/path/to/SensorCapture
```

checks every session (missing files, frame gaps, IMU rates, clock overlap, tag counts) and
prints total minutes. Install FFmpeg to also compare the video's frame count.

## Building

No local setup needed: push to `main`, the workflow builds `assembleRelease` with Gradle 8.11.1
and publishes `sensor-capture.apk`. The signing key in `signing/` is committed on purpose so each
build installs over the previous one; it signs only this personal tool.
