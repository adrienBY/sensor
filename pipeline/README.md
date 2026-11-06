# PC pipeline

## The easy way: WorldModel.exe

1. Download `WorldModel.exe` from the latest release and put it in its own folder, e.g.
   `D:\WorldModel\`. Everything it makes stays in that folder.
2. Copy `SensorCapture` from the phone to the PC (drag-and-drop straight from the phone over
   USB does not give Windows a real path, so copy it first).
3. Drag the `SensorCapture` folder onto `WorldModel.exe`.

The first run installs Python 3.12 and AMD's PyTorch for ROCm 7.2.1 into `.runtime\` (a few GB,
10–30 min, once). It needs AMD Adrenalin driver 26.2.2 or newer. Then it imports the new takes,
checks, aligns and encodes them, trains for `train_minutes` (60 by default, edit
`settings.json`), and opens the folder of dream videos. Run it again with more footage and it
continues training from where it stopped. Double-click it (no folder) for a menu: train more,
make dream videos, or walk in the dream with the keyboard (`play.py`). Windows may warn that the
.exe is unrecognised (it is not code-signed): click "More info" → "Run anyway".

## The manual way

The steps below are what the .exe runs. Each is safe to re-run: takes already done are skipped.

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

## The model (train.py)

`worldmodel.py` holds a small diffusion transformer (about 22M parameters) that predicts the next
latent frame from the previous 4 and the action between them. It follows DIAMOND's recipe: EDM
diffusion, 3 denoising steps per frame, and noise added to the past frames during training so
the model learns to recover from its own mistakes when it dreams. Training is time-budgeted and
resumable (`runs/main/last.pt`), keeps an EMA copy for dreaming, and every 20 minutes writes a
video to `runs/main/dreams/`: real footage on the left, the model's own rollout on the right.
`metrics.jsonl` logs PSNR between them at 1, 3 and 10 seconds.

```
python sensor\pipeline\train.py data\processed runs\main --minutes 60
python sensor\pipeline\dream.py data\processed runs\main\last.pt
python sensor\pipeline\play.py data\processed runs\main\last.pt
```
