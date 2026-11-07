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
2. Install the AMD Adrenalin driver 26.2.2 or newer (needed by ROCm 7.2.1), then PyTorch for
   ROCm following AMD's guide:
   https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
3. Then, in PowerShell:

```
cd D:\worldmodel
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy opencv-python diffusers safetensors accelerate
```

(Install PyTorch into this same venv in step 2.) Check the GPU is visible:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `True` and the RX 9070. PyTorch calls AMD GPUs "cuda" too. If it prints
`False` or a .dll error, AMD's troubleshooting steps are: driver 26.2.2 or newer, the Microsoft
Visual C++ 2015-2022 Redistributable (x64), and Smart App Control and Microsoft Defender
Application Guard turned off.

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

## Tuned for RDNA 4 (RX 9070)

What AMD's ROCm guidance recommends, and what the code does about it:

- **FP16 mixed precision with loss scaling** (`--amp fp16`, the default). AMD's support matrix
  for Radeon validates FP32, FP16 and FP16 mixed precision. RDNA 4's matrix units (WMMA) also run
  BF16, so `--amp bf16` exists, but the benchmark only picks it if it is clearly faster (>5%).
- **TunableOp**, PyTorch's GEMM auto-tuner for ROCm, which AMD recommends: for every matrix
  shape the model uses, it times the available hipBLASLt and rocBLAS kernels and keeps the
  fastest. Results are cached in `.runtime\tunableop_results0.csv` and reused.
- **Only GEMMs**: the patch embedding is a reshape + Linear layer (the same maths as the
  stride-2 convolution it replaces), so every matrix multiply goes to hipBLASLt, where
  TunableOp can tune it, and none goes through MIOpen. Older checkpoints still load.
- **Sizes that tile well**: width 384 (6 heads of 64), MLP 1536, 144 tokens per frame, batches
  of 64 or more. All are multiples of WMMA's 16x16 tiles. `--dim` must be a multiple of 64.
- **No CPU in the loop**: the whole dataset sits on the GPU in FP16 (an hour of footage is about
  125 MB) and batches are sampled there. The loss is read back every 30 s, not every step, so
  the GPU never waits for Python.
- **Fewer, bigger kernels**: fused AdamW, which also lets the FP16 loss scaler skip its
  per-step GPU-to-CPU check, and one multi-tensor EMA update.
- **Settings measured on your card**: `bench.py` tries FP16 and BF16 at batch 64, 128 and 256,
  then the best with TunableOp on, and WorldModel.exe stores the winner in `settings.json`
  (`amp`, `batch`, `tunableop`). It runs once per GPU; delete `tuned_gpu` there to re-run it.
  The learning rate scales with the square root of the batch size.
- **torch.compile** needs Triton, which ROCm for Windows does not ship, so `--compile` only
  does something on Linux or WSL2.
- **Native Windows vs WSL2**: AMD officially supports the RX 9070 with PyTorch 2.9.1 and
  ROCm 7.2.1 on WSL2, training included. On native Windows, AMD lists training as not yet
  supported. WorldModel.exe trains natively; if training misbehaves there, WSL2 is the
  officially supported fallback.

Running the same thing by hand:

```
python sensor\pipeline\bench.py --out bench.json --cache-dir .
$env:PYTORCH_TUNABLEOP_ENABLED = "1"; $env:PYTORCH_TUNABLEOP_TUNING = "1"
$env:PYTORCH_TUNABLEOP_FILENAME = "D:\worldmodel\tunableop_results.csv"
python sensor\pipeline\train.py data\processed runs\main --minutes 60 --amp fp16 --batch 128
```

(Use the `amp` and `batch` that `bench.json` reports, and drop the three `$env:` settings if it
says TunableOp did not help.)

## Model v3 (default for new runs)

`causal.py`: a frame-causal diffusion transformer trained on whole 16-frame clips, where every
frame is a target and can only see the past. Each frame has its own noise level (diffusion
forcing). It uses RMSNorm and QK-norm for stable FP16, 3D rotary positions, a per-frame camera
pose relative to the newest frame, memory frames as prefix tokens, action dropout for action
guidance, and gradient checkpointing. Sizes (`"model_size"` in `settings.json`): base 58M,
large 174M (default), xl 384M. When dreaming it re-runs the last 12 frames each step. New runs
go to `runs\v3`; earlier models stay in their folders (`"run": "memory"` uses the v2 model).
