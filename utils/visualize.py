"""
utils/visualize.py — Training Visualisation Helpers
====================================================

Provides lightweight utilities for:
- Saving a grid of generated images during training
- Plotting training loss curves (G loss / D loss)
- Logging to TensorBoard (optional, gracefully skipped if not available)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for Kaggle / servers)
import matplotlib.pyplot as plt


def save_image_grid(
    generator:   nn.Module,
    out_path:    str | Path,
    device:      torch.device,
    nz:          int   = 256,
    n_images:    int   = 64,
    nrow:        int   = 8,
    fixed_z:     Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate a grid of sample images and save to disk.

    Uses the EMA generator (should be passed as ``generator``) for
    higher-quality samples.

    Args:
        generator:  The generator model (typically the EMA copy).
        out_path:   File path to save the PNG grid.
        device:     Device to run inference on.
        nz:         Latent vector dimension.
        n_images:   Number of images to include in the grid.
        nrow:       Images per row in the output grid.
        fixed_z:    If provided, use these latent vectors (deterministic
                    output for tracking training progress over time).

    Returns:
        The generated image tensor (n_images, 3, H, W) in [-1, 1].
    """
    generator.eval()
    with torch.no_grad():
        z = fixed_z if fixed_z is not None else \
            torch.randn(n_images, nz, device=device)
        imgs = generator(z).cpu()  # (n_images, 3, H, W) in [-1, 1]

    # De-normalise to [0, 1] for saving
    imgs_01 = (imgs * 0.5 + 0.5).clamp(0.0, 1.0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(imgs_01, out_path, nrow=nrow, padding=2)
    generator.train()
    return imgs


class LossLogger:
    """Tracks and saves G/D loss curves during training.

    Also writes to TensorBoard if available. Plots are saved as PNG so they
    are always viewable even without a TensorBoard server.

    Args:
        log_dir:    Directory to save plots and TensorBoard event files.
        use_tb:     Attempt to use TensorBoard. Silently disabled if the
                    ``tensorboard`` package is not installed.
    """

    def __init__(self, log_dir: str | Path, use_tb: bool = True) -> None:
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._g_losses: list[float] = []
        self._d_losses: list[float] = []
        self._iters:    list[int]   = []

        # TensorBoard writer (optional)
        self._writer = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("[LossLogger] TensorBoard not available — skipping.")

    def log(self, iteration: int, g_loss: float, d_loss: float) -> None:
        """Record losses for one iteration."""
        self._iters.append(iteration)
        self._g_losses.append(g_loss)
        self._d_losses.append(d_loss)

        if self._writer is not None:
            self._writer.add_scalar('Loss/Generator',     g_loss, iteration)
            self._writer.add_scalar('Loss/Discriminator', d_loss, iteration)

    def save_plot(self, smooth_window: int = 100) -> Path:
        """Save a loss-curve plot as PNG.

        Args:
            smooth_window: Rolling average window for smoother curves.

        Returns:
            Path of the saved PNG.
        """
        if not self._iters:
            return

        def _smooth(vals: list[float], w: int) -> list[float]:
            out = []
            for i in range(len(vals)):
                start = max(0, i - w // 2)
                end   = min(len(vals), i + w // 2 + 1)
                out.append(sum(vals[start:end]) / (end - start))
            return out

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self._iters, _smooth(self._g_losses, smooth_window),
                label='G loss', color='#4C9BE8', linewidth=1.5)
        ax.plot(self._iters, _smooth(self._d_losses, smooth_window),
                label='D loss', color='#E87E4C', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('FastGAN + DiffAugment — Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = self.log_dir / 'loss_curve.png'
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path

    def close(self) -> None:
        """Flush and close TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()
