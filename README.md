# Sensor Capture

Android app that records training data for a learned world model of an apartment:
video, IMU, per-frame camera metadata and timestamped interaction tags, all on one clock.

Built by GitHub Actions on every push to `main`. The APK is attached to the latest release:
**https://github.com/adrienBY/sensor/releases/latest/download/sensor-capture.apk**

## Dataset format (`sensor-capture/1`)

Every take is one flat folder directly in `Download/SensorCapture/`. Copy that whole folder to
the PC as the dataset's raw input (e.g. `data/raw/`); the training pipeline treats each
sub-folder as one take and never needs anything outside it.

```
Download/SensorCapture/
  20261107-101530_whole-apartment_walkthrough_daylight_ar/
  20261107-103204_kitchen_interactions_daylight_ar/
  20261107-183002_corridor_transitions_evening-lights-on_video/
```

**Take id** = folder name = `meta.json → session`:
`<yyyyMMdd-HHmmss>_<room>_<take type>_<lighting>_<mode>`, lowercase, no spaces. Names sort by
time and stay unique, so the id is safe to use as the key for every derived file later
(`data/processed/<take id>/…`). The name is only for humans; code reads attributes from
`meta.json`, never by parsing the folder name.

**Rooms**: `corridor`, `living-room`, `kitchen`, `my-room`, `bathroom`, `parents-room`,
`whole-apartment`, `other`. **Take types**: `walkthrough`, `room-sweep`, `transitions`,
`interactions`, `free`. **Lighting**: `daylight`, `evening-lights-on`, `night-lights-on`,
`night-lights-off`, `mixed`. **Mode**: `video` or `ar`.

Inside each take, always the same file names:

| File | Contents |
| --- | --- |
| `video.mp4` | Camera video, H.264, 30 fps, no audio. Video mode: 1080p, electronic stabilization off. AR mode: written by ARCore at its largest 30 fps stream |
| `imu.csv` | `timestamp_ns,sensor,x,y,z,w` for `gyro` (rad/s), `accel` (m/s²), `grav` (m/s²), `rot` (game rotation vector quaternion x,y,z,w) at the fastest rate the phone allows |
| `frames.csv` | One row per camera frame: frame number, sensor timestamp, exposure, ISO, focus distance, focal length and per-frame intrinsics (fx, fy, cx, cy, skew) when the phone reports them |
| `events.json` | Interaction tags after undo: `{t_ns, t_video_s, object, action}` |
| `events.log.jsonl` | Raw tag log written as you tap, including undos (crash-safe) |
| `meta.json` | `format`, room, lighting, take type, mode, device, camera calibration, sensor list, clock anchors, counts, and `files` (the list of files in the take) |
| `poses.csv` | **AR mode only.** One row per camera frame: `timestamp_ns,tracking,failure_reason,tx,ty,tz,qx,qy,qz,qw` |

## Two modes

- **Video only (CameraX)**: the original mode. 1080p, stabilization off, rich per-frame camera
  metadata (exposure, ISO, focus). Camera positions are computed later on the PC.
- **AR (ARCore)**: ARCore owns the camera and tracks the phone in 3D while recording. The video is
  written by ARCore's Recording API at the largest 30 fps resolution ARCore offers (see
  `meta.json → ar_camera.texture_size`). Its MP4 also contains ARCore's own data tracks; the
  camera image is the first video stream (`ffmpeg -map 0:v:0`).

ARCore poses use `Camera.getPose()`: world-from-camera, OpenGL camera axes (+X right, +Y up,
−Z looking forward), metres, in a gravity-aligned world (+Y up) whose origin is where the AR
session started. Each take therefore has its own origin; the PC pipeline registers takes together.
`frames.csv` in AR mode holds the frame index, timestamp and texture intrinsics (fx, fy, cx, cy).

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
