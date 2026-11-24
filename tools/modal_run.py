"""food101_cnn.ipynb'yi Modal'da (GPU, çok çekirdekli CPU) papermill ile baştan sona çalıştırır.

    modal run --detach tools/modal_run.py --mode quick    # birkaç dakikalık uçtan uca deneme
    modal run --detach tools/modal_run.py --mode full     # rapor için tam koşu
    modal volume get bil485-food101 runs/full ./modal_out # çıktıları indir

Kalıcı disk (Volume) düzeni:
    cache/food-101.tar.gz           Food-101 arşivi (ikinci koşu yeniden indirmez)
    runs/<mode>/checkpoints/        checkpoint'ler (iş yarıda kesilirse biten model yeniden eğitilmez)
    runs/<mode>/results/, REPORT.md, food101_cnn_executed.ipynb, run.log
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
GPU = "L4"
CPUS = 8

app = modal.App("bil485-food101")
vol = modal.Volume.from_name("bil485-food101", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "numpy", "pandas", "matplotlib", "scikit-learn", "pillow", "pillow-heif",
                 "gradio", "papermill", "ipykernel", "nbformat")
    .add_local_dir(REPO, "/root/repo", ignore=["data", ".git", "results", "checkpoints", "**/__pycache__",
                                               "*.zip", "modal_out*", "REPORT.md", "*_executed.ipynb"])
)


@app.function(image=image, gpu=GPU, cpu=float(CPUS), memory=32768, timeout=4 * 3600, volumes={"/vol": vol})
def run(mode: str = "full") -> dict:
    t0 = time.time()
    work = Path("/root/work")
    shutil.copytree("/root/repo", work)
    out = Path("/vol/runs") / mode
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    os.symlink(out / "checkpoints", work / "checkpoints")

    data = work / "data"
    data.mkdir()
    cache = Path("/vol/cache/food-101.tar.gz")
    if cache.exists():
        shutil.copy(cache, data / cache.name)
        print(f"Food-101 arşivi önbellekten kopyalandı ({time.time() - t0:.0f} sn)", flush=True)

    env = dict(os.environ, QUICK_RUN="1" if mode == "quick" else "0",
               RUN_PLATFORM=f"Modal ({GPU}, {CPUS} vCPU)")
    executed = out / "food101_cnn_executed.ipynb"
    with open(out / "run.log", "w") as log:
        p = subprocess.Popen([sys.executable, "-m", "papermill", "food101_cnn.ipynb", str(executed), "--log-output",
                              "--request-save-on-cell-execute", "-k", "python3", "--cwd", str(work)],
                             cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            if "%|" not in line:  # indirme ilerleme çubuklarını Modal log'una basma
                print(line, end="", flush=True)
        p.wait()

    for name in ("results", "REPORT.md"):
        src = work / name
        if src.is_dir():
            shutil.copytree(src, out / name, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy(src, out / name)
    if not cache.exists() and (data / cache.name).exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(data / cache.name, cache)
    vol.commit()
    summary = {"mode": mode, "returncode": p.returncode, "minutes": round((time.time() - t0) / 60, 1)}
    print("ÖZET:", summary, flush=True)
    return summary


@app.local_entrypoint()
def main(mode: str = "full"):
    call = run.spawn(mode)
    print("Başlatıldı:", mode, "çağrı:", call.object_id)
