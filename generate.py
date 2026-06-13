"""
generate.py — Synthetic Image Generation from Trained Checkpoint
================================================================

Loads the EMA Generator from a checkpoint and generates a batch of
synthetic defect images. Output images are saved as individual PNGs
at 128×128 (ready for downstream detection model training).

Usage
-----
    python generate.py \\
        --ckpt checkpoints/ckpt_iter_0100000.pt \\
        --n_images 500 \\
        --out_dir synthetic_images/ \\
        --batch_size 32

Output structure
----------------
    synthetic_images/
    ├── img_000000.png
    ├── img_000001.png
    ├── ...
    └── img_000499.png

Each image is an un-augmented output from the EMA Generator — no
DiffAugment is applied here. These are the clean synthetic samples
intended for downstream model training.
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from tqdm import tqdm

from models import Generator


def generate(
    ckpt_path:   str | Path,
    out_dir:     str | Path,
    n_images:    int   = 500,
    batch_size:  int   = 32,
    nz:          int   = 256,
    ngf:         int   = 64,
    image_size:  int   = 128,
    device:      str   = 'auto',
    seed:        int   = 0,
) -> None:
    """Generate and save synthetic images from a saved checkpoint.

    Args:
        ckpt_path:  Path to the ``.pt`` checkpoint file.
        out_dir:    Directory to save generated PNGs.
        n_images:   Total number of images to generate.
        batch_size: Generation batch size (tune for VRAM).
        nz:         Latent dimension (must match training config).
        ngf:        Generator base filters (must match training config).
        image_size: Output resolution (must match training config).
        device:     ``'auto'``, ``'cpu'``, or ``'cuda'``.
        seed:       Random seed for reproducible generation.
    """
    # ── Device ──────────────────────────────────────────────────────────────
    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)
    print(f"[Generate] Device: {dev}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=dev, weights_only=False)
    iteration = payload.get('iteration', '?')
    print(f"[Generate] Loaded checkpoint from iteration {iteration}")

    # ── Model ────────────────────────────────────────────────────────────────
    G = Generator(ngf=ngf, nz=nz, im_size=image_size).to(dev)

    # Prefer EMA weights for best image quality
    if 'G_ema' in payload:
        G.load_state_dict(payload['G_ema'])
        print("[Generate] Using EMA generator weights.")
    else:
        G.load_state_dict(payload['G'])
        print("[Generate] EMA not found — using standard generator weights.")

    G.eval()

    # ── Output directory ─────────────────────────────────────────────────────
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Generate] Saving {n_images} images to: {out_dir}")

    # ── Generate in batches ──────────────────────────────────────────────────
    torch.manual_seed(seed)
    n_generated = 0

    with torch.no_grad():
        pbar = tqdm(total=n_images, desc='Generating', unit='img')
        while n_generated < n_images:
            # Last batch may be smaller
            bs  = min(batch_size, n_images - n_generated)
            z   = torch.randn(bs, nz, device=dev)
            imgs = G(z).cpu()                          # (bs, 3, H, W) in [-1,1]
            imgs = (imgs * 0.5 + 0.5).clamp(0.0, 1.0) # → [0, 1]

            for img in imgs:
                fname = out_dir / f"img_{n_generated:06d}.png"
                vutils.save_image(img, fname)
                n_generated += 1

            pbar.update(bs)

        pbar.close()

    print(f"[Generate] Done. {n_generated} images saved to '{out_dir}'.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate synthetic defect images')
    p.add_argument('--ckpt',       required=True,
                   help='Path to checkpoint file (.pt)')
    p.add_argument('--out_dir',    default='synthetic_images',
                   help='Output directory for generated PNGs')
    p.add_argument('--n_images',   type=int, default=500,
                   help='Number of images to generate')
    p.add_argument('--batch_size', type=int, default=32,
                   help='Generation batch size')
    p.add_argument('--nz',         type=int, default=256,
                   help='Latent dimension (must match training)')
    p.add_argument('--ngf',        type=int, default=64,
                   help='Generator base filters (must match training)')
    p.add_argument('--image_size', type=int, default=128,
                   help='Output resolution (must match training)')
    p.add_argument('--device',     default='auto',
                   help="'auto', 'cpu', or 'cuda'")
    p.add_argument('--seed',       type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate(
        ckpt_path  = args.ckpt,
        out_dir    = args.out_dir,
        n_images   = args.n_images,
        batch_size = args.batch_size,
        nz         = args.nz,
        ngf        = args.ngf,
        image_size = args.image_size,
        device     = args.device,
        seed       = args.seed,
    )
