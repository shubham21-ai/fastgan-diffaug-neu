"""
evaluate.py — FID Score Computation
=====================================

Computes the Fréchet Inception Distance (FID) between the real training
images and a folder of generated synthetic images.

FID measures the distance between the feature distributions of real and
synthetic images in an Inception-v3 feature space. Lower FID = more
realistic and diverse synthetic images.

Interpretation for NEU-DET (150 training images):
    FID < 150: Decent (generator has learned basic structure)
    FID < 100: Good  (texture and layout visually plausible)
    FID < 50:  Excellent (high-quality synthesis)

Usage
-----
    # After training — compute FID
    python evaluate.py \\
        --real_dir data/NEU-DET/train \\
        --fake_dir synthetic_images/ \\
        --n_real   150 \\
        --n_fake   500

Requirements
------------
    pip install clean-fid
    (clean-fid is Inception V3 based, handles resizing internally)
"""

import argparse
from pathlib import Path


def compute_fid(
    real_dir: str | Path,
    fake_dir: str | Path,
    n_real:   int = None,
    n_fake:   int = None,
    device:   str = 'auto',
) -> float:
    """Compute FID between real and fake image directories.

    Args:
        real_dir: Directory of real images (can contain subdirectories).
        fake_dir: Directory of generated images (flat or nested).
        n_real:   Maximum number of real images to use. None = all.
        n_fake:   Maximum number of fake images to use. None = all.
        device:   ``'auto'``, ``'cpu'``, or ``'cuda'``.

    Returns:
        FID score (float). Lower is better.
    """
    try:
        from cleanfid import fid as clean_fid
    except ImportError:
        raise ImportError(
            "clean-fid is required for FID evaluation. "
            "Install it with: pip install clean-fid"
        )

    import torch
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)

    if not real_dir.exists():
        raise FileNotFoundError(f"Real image directory not found: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"Fake image directory not found: {fake_dir}")

    # Collect all real images (handles nested class directories)
    _EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    real_files = sorted([
        p for p in real_dir.rglob('*')
        if p.suffix.lower() in _EXTS
    ])
    fake_files = sorted([
        p for p in fake_dir.rglob('*')
        if p.suffix.lower() in _EXTS
    ])

    print(f"[FID] Real images found: {len(real_files)}")
    print(f"[FID] Fake images found: {len(fake_files)}")

    if n_real:
        real_files = real_files[:n_real]
    if n_fake:
        fake_files = fake_files[:n_fake]

    print(f"[FID] Using {len(real_files)} real, {len(fake_files)} fake images")
    print(f"[FID] Computing FID (this may take a few minutes)…")

    score = clean_fid.compute_fid(
        str(real_dir),
        str(fake_dir),
        device  = device,
        verbose = True,
    )

    print(f"\n[FID] Score: {score:.4f}")
    _interpret(score)
    return score


def _interpret(fid: float) -> None:
    """Print a human-readable interpretation of the FID score."""
    if fid < 50:
        quality = "★★★ Excellent"
    elif fid < 100:
        quality = "★★☆ Good"
    elif fid < 150:
        quality = "★☆☆ Decent"
    else:
        quality = "☆☆☆ Needs more training"
    print(f"[FID] Quality assessment: {quality}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compute FID for generated images')
    p.add_argument('--real_dir', required=True,
                   help='Directory of real training images')
    p.add_argument('--fake_dir', required=True,
                   help='Directory of generated synthetic images')
    p.add_argument('--n_real',   type=int, default=None,
                   help='Max real images to use (None = all)')
    p.add_argument('--n_fake',   type=int, default=None,
                   help='Max fake images to use (None = all)')
    p.add_argument('--device',   default='auto')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    compute_fid(
        real_dir = args.real_dir,
        fake_dir = args.fake_dir,
        n_real   = args.n_real,
        n_fake   = args.n_fake,
        device   = args.device,
    )
