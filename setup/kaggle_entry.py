"""
setup/kaggle_entry.py — Complete Kaggle Notebook Entry Point
=============================================================

Copy each section below into a separate Kaggle notebook cell.
This script handles: environment setup, dataset preparation, training,
and post-training generation — all in one place.

HOW TO USE ON KAGGLE
--------------------
1. Create a new Kaggle notebook (GPU P100 or T4)
2. Add the NEU Surface Defect Database as input dataset:
   https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
3. Upload this entire repo as a second input dataset, OR run:
       !git clone https://github.com/shubham21-ai/fastgan-diffaug-neu.git /kaggle/working/fastgan-diffaug-neu
4. Copy each CELL below into separate notebook cells
5. Run cells sequentially

CRASH RECOVERY
--------------
If the session times out (Kaggle free tier = ~9 hrs), the SIGTERM handler
saves an emergency checkpoint. On restart:
- Re-run all setup cells (install deps, imports)
- Re-run the training cell — it auto-resumes from the latest checkpoint
"""

# =============================================================================
# CELL 1: Install dependencies
# =============================================================================
# Paste this into Cell 1 of your Kaggle notebook

CELL_1 = """
import subprocess, sys

packages = [
    "lpips",
    "clean-fid",
    "PyYAML",
    "tqdm",
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("All dependencies installed.")
"""

# =============================================================================
# CELL 2: Verify GPU and set up paths
# =============================================================================

CELL_2 = """
import torch, os

# ── GPU check ────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "No GPU detected! Enable GPU in Kaggle settings."
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Adjust these if the dataset has a different name in your input
import glob

# Try to auto-locate the NEU-DET dataset among Kaggle inputs
KAGGLE_INPUT = "/kaggle/input"
candidates = glob.glob(f"{KAGGLE_INPUT}/**/NEU_data", recursive=True) + \\
             glob.glob(f"{KAGGLE_INPUT}/**/images", recursive=True) + \\
             glob.glob(f"{KAGGLE_INPUT}/**/*.jpg", recursive=True) + \\
             glob.glob(f"{KAGGLE_INPUT}/**/*.bmp", recursive=True)

if candidates:
    # Walk up to find the dataset root
    import os
    sample = candidates[0]
    DATA_ROOT = os.path.dirname(sample)
    print(f"Auto-detected dataset path: {DATA_ROOT}")
else:
    # Manual fallback — update this line if auto-detection fails
    DATA_ROOT = "/kaggle/input/neu-surface-defect-database"
    print(f"Using manual path: {DATA_ROOT}")

SAVE_DIR = "/kaggle/working"
REPO_DIR = "/kaggle/working/fastgan-diffaug-neu"

print(f"Save dir:  {SAVE_DIR}")
print(f"Repo dir:  {REPO_DIR}")
"""

# =============================================================================
# CELL 3: Clone repo (skip if you uploaded it as a dataset)
# =============================================================================

CELL_3 = """
import subprocess, os

REPO_DIR = "/kaggle/working/fastgan-diffaug-neu"

if not os.path.exists(REPO_DIR):
    # Replace with your actual GitHub URL after pushing the code
    subprocess.run([
        "git", "clone",
        "https://github.com/shubham21-ai/fastgan-diffaug-neu.git",
        REPO_DIR
    ], check=True)
    print(f"Cloned repo to {REPO_DIR}")
else:
    print(f"Repo already exists at {REPO_DIR}")

import sys
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("Repo added to sys.path")
"""

# =============================================================================
# CELL 4: Explore dataset structure
# =============================================================================

CELL_4 = """
import os, glob
from pathlib import Path

# Show what we're working with
DATA_ROOT = "/kaggle/input/neu-surface-defect-database"  # adjust if needed

print("Dataset structure:")
for root, dirs, files in os.walk(DATA_ROOT):
    depth = root.replace(DATA_ROOT, '').count(os.sep)
    if depth > 2:
        break
    indent = '  ' * depth
    print(f"{indent}{os.path.basename(root)}/")
    if depth == 2:
        imgs = [f for f in files if f.endswith(('.bmp', '.jpg', '.png'))]
        print(f"{indent}  └── {len(imgs)} images")

print()
print(f"Total JPG files: {len(glob.glob(DATA_ROOT + '/**/*.jpg', recursive=True))}")
print(f"Total BMP files: {len(glob.glob(DATA_ROOT + '/**/*.bmp', recursive=True))}")
"""

# =============================================================================
# CELL 5: Quick smoke test (verify models work, ~30 seconds)
# =============================================================================

CELL_5 = """
import sys
sys.path.insert(0, "/kaggle/working/fastgan-diffaug-neu")

import torch
from models import Generator, Discriminator
from diffaug import DiffAugment

device = torch.device('cuda')

# Build models
G = Generator(ngf=64, nz=256, im_size=128).to(device)
D = Discriminator(ndf=64, im_size=128).to(device)

# Forward pass test
z    = torch.randn(4, 256, device=device)
fake = G(z)
print(f"Generator output: {fake.shape}")   # (4, 3, 128, 128)

aug  = DiffAugment(fake, 'color,translation,cutout')
pred = D(aug)
print(f"Discriminator pred (fake): {pred.shape}")  # (4, 1)

import random
pred_r, recs = D(aug, part=random.randint(0, 3))
print(f"Discriminator pred (real): {pred_r.shape}")
print(f"Reconstruction shapes: {[r.shape for r in recs]}")

# VRAM check
used  = torch.cuda.memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM used: {used:.2f} / {total:.1f} GB")
print("Smoke test passed!")
"""

# =============================================================================
# CELL 6: Start / Resume Training
# =============================================================================

CELL_6 = """
import sys, os
sys.path.insert(0, "/kaggle/working/fastgan-diffaug-neu")

import yaml

# Load config and override paths for Kaggle
with open("/kaggle/working/fastgan-diffaug-neu/configs/neu_det_128.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Override these two paths for Kaggle ──
cfg['data_path'] = DATA_ROOT     # from Cell 2
cfg['save_dir']  = SAVE_DIR      # /kaggle/working

# Optional tuning
cfg['total_iter']  = 100000      # ~3-4 hours on P100
cfg['batch_size']  = 16          # safe for P100 + AMP at 128×128
cfg['save_every']  = 500         # checkpoint every 500 iters
cfg['vis_every']   = 2000        # image grid every 2000 iters

print("Starting training with config:")
for k, v in cfg.items():
    print(f"  {k}: {v}")

# Import and run training
from train import train
train(cfg)
"""

# =============================================================================
# CELL 7: Generate synthetic images after training
# =============================================================================

CELL_7 = """
import sys, os, glob
sys.path.insert(0, "/kaggle/working/fastgan-diffaug-neu")

from generate import generate
from pathlib import Path

# Find the latest non-emergency checkpoint
ckpts = sorted(glob.glob("/kaggle/working/checkpoints/ckpt_iter_*.pt"))
# Prefer non-emergency checkpoints
normal_ckpts = [c for c in ckpts if '_emergency' not in c]
latest_ckpt  = normal_ckpts[-1] if normal_ckpts else ckpts[-1]

print(f"Using checkpoint: {latest_ckpt}")

# Generate 500 synthetic images
generate(
    ckpt_path  = latest_ckpt,
    out_dir    = "/kaggle/working/synthetic_defects",
    n_images   = 500,
    batch_size = 32,
    nz         = 256,
    ngf        = 64,
    image_size = 128,
)

# Show a few generated images inline
import matplotlib.pyplot as plt
from PIL import Image
import glob

samples = sorted(glob.glob("/kaggle/working/synthetic_defects/*.png"))[:16]
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for ax, path in zip(axes.flat, samples):
    ax.imshow(Image.open(path))
    ax.axis('off')
plt.suptitle('Synthetic NEU Defect Images (FastGAN + DiffAugment)', fontsize=12)
plt.tight_layout()
plt.savefig("/kaggle/working/synthetic_preview.png", dpi=150, bbox_inches='tight')
plt.show()
print("Preview saved to /kaggle/working/synthetic_preview.png")
"""

# =============================================================================
# CELL 8: Compute FID (optional — run after generation)
# =============================================================================

CELL_8 = """
import sys
sys.path.insert(0, "/kaggle/working/fastgan-diffaug-neu")

from evaluate import compute_fid

fid_score = compute_fid(
    real_dir = DATA_ROOT,
    fake_dir = "/kaggle/working/synthetic_defects",
)
print(f"Final FID: {fid_score:.2f}")
"""

# =============================================================================
# Standalone runner (executes all cells sequentially when run as __main__)
# =============================================================================

if __name__ == '__main__':
    """
    This block runs all cells when executed as a plain Python script.
    On Kaggle, paste each CELL_N into its own notebook cell instead.
    """
    import subprocess, sys, os

    print("=" * 60)
    print("FastGAN + DiffAugment — Kaggle Entry Point")
    print("=" * 60)
    print()
    print("To use this on Kaggle:")
    print("  1. Create a Kaggle notebook with GPU enabled")
    print("  2. Add NEU-DET dataset as input")
    print("  3. Paste each CELL_N block into separate notebook cells")
    print("  4. Run cells in order")
    print()
    print("Cell contents have been printed to stdout for reference.")
    print()

    for name, code in [
        ("CELL 1 (Install)", CELL_1),
        ("CELL 2 (Setup paths)", CELL_2),
        ("CELL 3 (Clone repo)", CELL_3),
        ("CELL 4 (Explore data)", CELL_4),
        ("CELL 5 (Smoke test)", CELL_5),
        ("CELL 6 (Train)", CELL_6),
        ("CELL 7 (Generate)", CELL_7),
        ("CELL 8 (FID)", CELL_8),
    ]:
        print(f"{'─'*60}")
        print(f"# {name}")
        print(code)
