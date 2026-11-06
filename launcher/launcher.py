"""WorldModel.exe: one program for the whole PC side.

Drop SensorCapture folders (or single take folders) onto WorldModel.exe and it will:
  1. copy the new takes into its data folder,
  2. check them, align them and compute actions,
  3. encode the frames into latents on the GPU,
  4. train the world model (continuing from last time),
  5. save dream videos and open the folder.
Double-click it instead to get a menu (add footage, train more, walk in the dream, ...).

Everything lives next to the .exe:
  WorldModel.exe
  data/raw/SensorCapture/   your takes
  data/processed/           aligned frames, actions, latents
  runs/main/                the trained model (last.pt) and dreams/
  logs/                     one log file per run
  .runtime/                 Python + PyTorch for AMD GPUs (installed on first run, a few GB)
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

LAUNCHER_VERSION = 1
ROCM = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
ROCM_SDK = [
    ROCM + "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    ROCM + "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    ROCM + "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    ROCM + "rocm-7.2.1.tar.gz",
]
TORCH = [ROCM + "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"]
EXTRA = ["numpy", "opencv-python", "diffusers", "safetensors", "accelerate"]
DEFAULT_SETTINGS = {"train_minutes": 60, "fps": 15, "width": 256, "height": 144}

FROZEN = getattr(sys, "frozen", False)
BUNDLE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
HOME = Path(os.environ.get("WORLDMODEL_HOME") or
            (Path(sys.executable).parent if FROZEN else Path.cwd())).resolve()
RUNTIME = HOME / ".runtime"
APP = RUNTIME / "app"
RAW = HOME / "data" / "raw" / "SensorCapture"
PROCESSED = HOME / "data" / "processed"
RUN = HOME / "runs" / "main"
LOGS = HOME / "logs"
LOG_FILE = None


# ---------------------------------------------------------------------------------------
# Output helpers

def say(text=""):
    print(text, flush=True)
    if LOG_FILE:
        LOG_FILE.write(text + "\n")
        LOG_FILE.flush()


def header(text):
    say()
    say("=" * 70)
    say(f"  {text}")
    say("=" * 70)


def run(cmd, env_extra=None, cwd=None):
    """Run a command, streaming its output to the console and the log. Returns exit code."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HOME"] = str(RUNTIME / "hf")
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1, cwd=cwd, env=env)
    while True:
        try:
            line = proc.stdout.readline()
        except KeyboardInterrupt:
            # Ctrl+C also reaches the child (training saves and exits); keep reading.
            continue
        if not line and proc.poll() is not None:
            break
        if line:
            say(line.rstrip("\n"))
    try:
        return proc.wait()
    except KeyboardInterrupt:
        return proc.wait()


# ---------------------------------------------------------------------------------------
# First-run setup: Python 3.12 + PyTorch for AMD GPUs, through uv

def python_exe():
    override = os.environ.get("WORLDMODEL_PYTHON")
    if override:
        return Path(override)
    if os.name == "nt":
        return RUNTIME / "venv" / "Scripts" / "python.exe"
    return RUNTIME / "venv" / "bin" / "python"


def uv_exe():
    name = "uv.exe" if os.name == "nt" else "uv"
    bundled = BUNDLE / name
    return bundled if bundled.exists() else Path(shutil.which("uv") or name)


def setup_runtime():
    marker = RUNTIME / "ready.json"
    py = python_exe()
    if os.environ.get("WORLDMODEL_PYTHON") or (marker.exists() and py.exists()):
        return py
    header("First-time setup: installing Python and PyTorch for your AMD GPU")
    say("This downloads a few GB once. It can take 10-30 minutes depending on your connection.")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    uv = uv_exe()
    env = {"UV_PYTHON_INSTALL_DIR": str(RUNTIME / "python"),
           "UV_CACHE_DIR": str(RUNTIME / "cache")}
    steps = [
        ("Installing Python 3.12", [uv, "python", "install", "3.12"]),
        ("Creating the environment", [uv, "venv", "--seed", "--python", "3.12",
                                      RUNTIME / "venv"]),
        ("Installing AMD ROCm libraries", [uv, "pip", "install", "--python", py, *ROCM_SDK]),
        ("Installing PyTorch for ROCm", [uv, "pip", "install", "--python", py, *TORCH]),
        ("Installing image and model tools", [uv, "pip", "install", "--python", py, *EXTRA]),
    ]
    for title, cmd in steps:
        say(f"\n> {title}")
        if run(cmd, env) != 0:
            if cmd[1] == "pip" and py.exists():
                say("uv failed; retrying with pip")
                pkgs = [str(c) for c in cmd[cmd.index(py) + 1:]]
                if run([py, "-m", "pip", "install", "--no-cache-dir", *pkgs], env) == 0:
                    continue
            raise SystemExit(f"Setup failed at: {title}. See the log in {LOGS}.")

    say("\n> Checking the GPU")
    check = ("import torch; ok = torch.cuda.is_available(); "
             "print('GPU:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND')")
    run([py, "-c", check])
    marker.write_text(json.dumps({"launcher": LAUNCHER_VERSION,
                                  "installed": datetime.datetime.now().isoformat()}))
    return py


def gpu_ok(py):
    out = subprocess.run([str(py), "-c", "import torch; print(torch.cuda.is_available())"],
                         capture_output=True, text=True)
    return out.stdout.strip().endswith("True")


def install_scripts():
    """Copy the pipeline scripts bundled in the .exe next to the runtime (refreshed each run)."""
    APP.mkdir(parents=True, exist_ok=True)
    for src in list((BUNDLE / "pipeline").glob("*.py")) + [BUNDLE / "tools" / "check_session.py"]:
        shutil.copy2(src, APP / src.name)


def settings():
    path = HOME / "settings.json"
    s = dict(DEFAULT_SETTINGS)
    if path.exists():
        try:
            s.update(json.loads(path.read_text()))
        except ValueError:
            say("settings.json is not valid JSON; using defaults")
    else:
        path.write_text(json.dumps(s, indent=2))
    return s


# ---------------------------------------------------------------------------------------
# Steps

def import_footage(paths):
    header("Importing footage")
    RAW.mkdir(parents=True, exist_ok=True)
    added = skipped = 0
    for p in paths:
        p = Path(p)
        if not p.exists():
            say(f"Not found: {p}")
            continue
        takes = [p] if (p / "meta.json").exists() else sorted({m.parent for m in p.rglob("meta.json")})
        if not takes:
            say(f"No takes (folders with meta.json) in {p}")
        for t in takes:
            dest = RAW / t.name
            if dest.exists():
                skipped += 1
                continue
            say(f"  + {t.name}")
            shutil.copytree(t, dest)
            added += 1
    say(f"{added} new takes copied, {skipped} already here. Takes folder: {RAW}")
    return added


def step(py, title, script, *args):
    header(title)
    code = run([py, APP / script, *args], cwd=APP)
    if code != 0:
        say(f"\n{title} failed (exit code {code}). The log is in {LOGS}.")
        return False
    return True


def process(py, s):
    if not any(RAW.glob("*/meta.json")):
        say(f"No takes yet. Drop your SensorCapture folder on WorldModel.exe, or copy takes into {RAW}")
        return False
    step(py, "1/3  Checking takes", "check_session.py", RAW)   # informational, never blocks
    if not step(py, "2/3  Aligning video and sensors, computing actions", "preprocess.py",
                RAW, PROCESSED, "--width", s["width"], "--height", s["height"], "--fps", s["fps"]):
        return False
    return step(py, "3/3  Encoding frames into latents (GPU)", "encode_latents.py", PROCESSED)


def train(py, s, minutes=None):
    minutes = minutes or s["train_minutes"]
    ok = step(py, f"Training the world model for {minutes} minutes (Ctrl+C stops and saves)",
              "train.py", PROCESSED, RUN, "--minutes", minutes)
    if ok and (RUN / "dreams").exists():
        say(f"\nDream videos: {RUN / 'dreams'}")
        open_folder(RUN / "dreams")
    return ok


def open_folder(path):
    try:
        if os.name == "nt":
            os.startfile(str(path))  # noqa: S606 - opens Explorer
    except OSError:
        pass


def pick_folder():
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        d = filedialog.askdirectory(title="Choose your SensorCapture folder (or one take)")
        root.destroy()
        return d or None
    except Exception:
        d = input("Path of the SensorCapture folder: ").strip().strip('"')
        return d or None


def menu(py, s):
    while True:
        trained = (RUN / "last.pt").exists()
        n_raw = len(list(RAW.glob("*/meta.json"))) if RAW.exists() else 0
        say()
        say(f"World Model  ·  {HOME}")
        say(f"{n_raw} takes  ·  model {'trained' if trained else 'not trained yet'}  ·  "
            f"training runs {s['train_minutes']} min (settings.json)")
        say("  1  Add footage (pick a folder) and run everything")
        say("  2  Process + train on the footage already here")
        say("  3  Train more")
        say("  4  Walk in the dream (keyboard)")
        say("  5  Make dream videos")
        say("  6  Open the WorldModel folder")
        say("  Q  Quit")
        choice = input("> ").strip().lower()
        if choice == "1":
            d = pick_folder()
            if d:
                import_footage([d])
                if process(py, s):
                    train(py, s)
        elif choice == "2":
            if process(py, s):
                train(py, s)
        elif choice == "3":
            m = input(f"Minutes [{s['train_minutes']}]: ").strip()
            train(py, s, float(m) if m else None)
        elif choice in ("4", "5"):
            if not trained:
                say("Train the model first (option 2).")
                continue
            if choice == "4":
                step(py, "Walking in the dream: the window opens shortly", "play.py",
                     PROCESSED, RUN / "last.pt")
            else:
                if step(py, "Making dream videos", "dream.py", PROCESSED, RUN / "last.pt"):
                    open_folder(RUN / "dreams")
        elif choice == "6":
            open_folder(HOME)
        elif choice in ("q", "quit", "exit"):
            return


def main():
    global LOG_FILE
    LOGS.mkdir(parents=True, exist_ok=True)
    LOG_FILE = open(LOGS / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.log", "w", encoding="utf-8")
    try:
        say(f"World Model launcher v{LAUNCHER_VERSION}  ·  folder: {HOME}")
        s = settings()
        py = setup_runtime()
        install_scripts()
        if not gpu_ok(py):
            say("\nWARNING: PyTorch cannot see your AMD GPU, so training would run on the CPU.")
            say("Update the AMD Adrenalin driver (26.2.2 or newer is required for ROCm 7.2.1),")
            say("restart, and run WorldModel.exe again.")
        dropped = [a for a in sys.argv[1:] if not a.startswith("-")]
        if dropped:
            import_footage(dropped)
            if process(py, s):
                train(py, s)
        else:
            menu(py, s)
    except SystemExit as e:
        if e.code not in (None, 0):
            say(str(e.code))
    except Exception as e:  # show the error instead of closing the window
        say(f"\nUnexpected error: {e!r}")
        raise
    finally:
        say("\nDone. Press Enter to close this window.")
        if LOG_FILE:
            LOG_FILE.close()
        try:
            input()
        except EOFError:
            pass


if __name__ == "__main__":
    main()
