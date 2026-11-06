# PC pipeline

Turns the `SensorCapture` folder from the phone into training-ready data. Three steps, each
safe to re-run: takes that are already done are skipped.

```
D:\worldmodel\
  sensor\                      this repo (git clone https://github.com/adrienBY/sensor)
  data\
    raw\SensorCapture\         copied from the phone, one folder per take
    processed\                 written by the pipeline
      index.json               every take, its split (train/val) and the tag vocabulary
      <take id>\
        frames.npy  timestamps.npy  sharpness.npy  actions.npz  take.json
        latents.npy  preview.png   (after step 3)
```

## One-time setup (Windows 11)

1. Install Python 3.12 from python.org (tick "Add to PATH").
2. Install AMD's PyTorch-on-Windows driver (Adrenalin 26.1.1 or newer), then PyTorch for ROCm
   following AMD's guide:
   https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
3. Then, in PowerShell:

```
cd D:\worldmodel
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy opencv-python diffusers
```

(Install PyTorch into this same venv in step 2.) Check the GPU is visible:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `True` and the RX 9070. PyTorch calls AMD GPUs "cuda" too.

## Every time you bring footage over

```
cd D:\worldmodel
.\venv\Scripts\Activate.ps1

# 1. Check the takes (missing files, frame gaps, IMU rates, AR tracking)
python sensor\tools\check_session.py data\raw\SensorCapture

# 2. Decode, align, compute actions (CPU, ~10 s per minute of footage)
python sensor\pipeline\preprocess.py data\raw\SensorCapture data\processed

# 3. Encode frames into latents (GPU)
python sensor\pipeline\encode_latents.py data\processed
```

## What step 2 checks for you

- **alignment correlation** in `take.json`: how well motion blur matched the gyroscope.
  Above ~0.5 is solid; under 0.3 prints a warning (usually a take with almost no turning).
- **gyro_vs_arcore_rotation_corr** (AR takes): the gyroscope and ARCore must agree on how the
  camera turned. Expect above 0.9; a low or negative value means a convention problem, so tell me.
- **blurry_frames**: frames much blurrier than the take's median. Training can down-weight them.

## Actions, in one line

For each step from frame t to t+1: `rot` (3 numbers, how the camera turned, radians) and, for AR
takes, `trans` (3 numbers, how it moved, metres), both in the camera's own frame at t
(+X right, +Y up, looking along −Z). `event` is the interaction tag that happened in that step.
That 6-number action plus the tag is what the world model is conditioned on.
