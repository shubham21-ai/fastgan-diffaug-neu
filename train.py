"""
train.py — Main FastGAN + DiffAugment Training Loop
=====================================================

Usage (local / Colab)
---------------------
    python train.py --data_path /path/to/NEU-DET --config configs/neu_det_128.yaml

Usage (Kaggle — override save directory)
-----------------------------------------
    python train.py \\
        --data_path /kaggle/input/neu-surface-defect-database \\
        --save_dir  /kaggle/working/checkpoints \\
        --config    configs/neu_det_128.yaml

Crash Safety (Kaggle Free Tier)
--------------------------------
- Checkpoints are saved every ``save_every`` iterations (default 500).
- A SIGTERM handler saves an emergency checkpoint before the session ends.
- The loop catches any exception and attempts a final save.
- Mixed-precision (AMP) keeps VRAM under 8 GB on a P100/T4.
- Set ``CUDA_VISIBLE_DEVICES=0`` if multi-GPU is available but unwanted.

Training Loop Summary
---------------------
    For each iteration:
      1. Sample real batch from InfiniteDataLoader.
      2. Apply DiffAugment to real images.
      3. Update Discriminator:
           a. Real forward → hinge loss + perceptual reconstruction loss.
           b. Fake forward (detached G output) → hinge loss.
      4. Update Generator:
           a. New fake forward → non-saturating adversarial loss.
      5. Update G_ema (exponential moving average of G weights).
      6. Log losses every 50 iters; save image grid every 1000 iters.
      7. Save checkpoint every 500 iters.
"""

import argparse
import copy
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dataset   import get_dataloader, find_dataset_root
from diffaug   import DiffAugment
from models    import Generator, Discriminator, crop_quadrant, weights_init
from utils.checkpoint import CheckpointManager
from utils.visualize  import save_image_grid, LossLogger


# ── Perceptual loss with graceful MSE fallback ────────────────────────────────

class PerceptualLoss(nn.Module):
    """LPIPS perceptual loss with automatic MSE fallback.

    LPIPS gives much better reconstruction signal than pixel MSE for the
    discriminator's self-supervised decoder heads. However, if the lpips
    package is unavailable (rare on Kaggle, possible offline), we fall back
    to MSE so training does not crash.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lpips_fn = None
        try:
            import lpips
            self._lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
            for p in self._lpips_fn.parameters():
                p.requires_grad_(False)
            print("[PerceptualLoss] Using LPIPS (VGG).")
        except Exception as exc:
            print(f"[PerceptualLoss] LPIPS unavailable ({exc}). "
                  "Falling back to MSE reconstruction loss.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._lpips_fn is not None:
            # Ensure LPIPS module is on same device as inputs
            self._lpips_fn = self._lpips_fn.to(pred.device)
            return self._lpips_fn(pred, target).mean()
        return F.mse_loss(pred, target)


# ── EMA helpers ───────────────────────────────────────────────────────────────

def make_ema_model(G: nn.Module) -> nn.Module:
    """Create an EMA (Exponential Moving Average) copy of the generator."""
    G_ema = copy.deepcopy(G)
    G_ema.eval()
    for p in G_ema.parameters():
        p.requires_grad_(False)
    return G_ema


@torch.no_grad()
def update_ema(G_ema: nn.Module, G: nn.Module, decay: float = 0.999) -> None:
    """Update EMA model: θ_ema ← decay·θ_ema + (1−decay)·θ_G."""
    for p_ema, p in zip(G_ema.parameters(), G.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ── Loss functions ────────────────────────────────────────────────────────────

def hinge_loss_real(pred: torch.Tensor) -> torch.Tensor:
    """Hinge loss for real samples with soft label smoothing.

    Labels drawn from Uniform[0.8, 1.0] rather than hard 1.0 to prevent
    the discriminator from becoming overconfident on the 150-image dataset.
    """
    smooth_label = torch.rand_like(pred) * 0.2 + 0.8  # ∈ [0.8, 1.0]
    return F.relu(smooth_label - pred).mean()


def hinge_loss_fake(pred: torch.Tensor) -> torch.Tensor:
    """Hinge loss for fake samples (soft label at -1)."""
    smooth_label = torch.rand_like(pred) * 0.2 + 0.8  # ∈ [0.8, 1.0]
    return F.relu(smooth_label + pred).mean()


# ── Training function ─────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    """Main training loop.

    Args:
        cfg: Configuration dictionary loaded from YAML.
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")
    if device.type == 'cuda':
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Dataset ────────────────────────────────────────────────────────────
    data_path = cfg['data_path']
    try:
        data_path = find_dataset_root(data_path)
    except FileNotFoundError:
        data_path = Path(data_path)

    loader, dataset = get_dataloader(
        root        = data_path,
        n_per_class = cfg['n_per_class'],
        image_size  = cfg['image_size'],
        batch_size  = cfg['batch_size'],
        num_workers = cfg.get('num_workers', 2),
        seed        = cfg['seed'],
    )

    # ── Models ─────────────────────────────────────────────────────────────
    G     = Generator(ngf=cfg['ngf'], nz=cfg['nz'], im_size=cfg['image_size']).to(device)
    D     = Discriminator(ndf=cfg['ndf'], im_size=cfg['image_size']).to(device)
    G_ema = make_ema_model(G)
    G_ema = G_ema.to(device)

    print(f"[Model] Generator params:     {sum(p.numel() for p in G.parameters()):,}")
    print(f"[Model] Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # ── Optimisers ─────────────────────────────────────────────────────────
    opt_G = torch.optim.Adam(G.parameters(),
                             lr=cfg['lr_g'], betas=(cfg['beta1'], 0.999))
    opt_D = torch.optim.Adam(D.parameters(),
                             lr=cfg['lr_d'], betas=(cfg['beta1'], 0.999))

    # ── Mixed precision (AMP) scalers ──────────────────────────────────────
    # Keeps VRAM ~6 GB at 128×128 / batch=16 on P100
    scaler_G = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    scaler_D = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ── Perceptual loss ────────────────────────────────────────────────────
    percept = PerceptualLoss().to(device)

    # ── Checkpoint manager ─────────────────────────────────────────────────
    save_dir = Path(cfg['save_dir'])
    ckpt_mgr = CheckpointManager(save_dir / 'checkpoints', keep_last_n=3)

    # ── Resume if checkpoint exists ────────────────────────────────────────
    start_iter = ckpt_mgr.load_latest(
        G, D, G_ema, opt_G, opt_D, device, scaler_G, scaler_D
    )

    # ── Logging ────────────────────────────────────────────────────────────
    logger   = LossLogger(save_dir / 'logs')
    img_dir  = save_dir / 'sample_images'
    img_dir.mkdir(parents=True, exist_ok=True)

    # Fixed latent vectors for consistent visualisation across iterations
    fixed_z = torch.randn(64, cfg['nz'], device=device)

    # ── SIGTERM handler (Kaggle session timeout) ───────────────────────────
    def _emergency_save(signo, frame):
        print("\n[SIGTERM] Kaggle session ending! Saving emergency checkpoint…")
        ckpt_mgr.save(
            G, D, G_ema, opt_G, opt_D,
            iteration  = _state['iteration'],
            loss_g     = _state['loss_g'],
            loss_d     = _state['loss_d'],
            scaler_G   = scaler_G,
            scaler_D   = scaler_D,
            is_emergency = True,
        )
        logger.save_plot()
        logger.close()
        sys.exit(0)

    _state = {'iteration': start_iter, 'loss_g': 0.0, 'loss_d': 0.0}
    signal.signal(signal.SIGTERM, _emergency_save)

    # ── Training loop ──────────────────────────────────────────────────────
    aug_policy   = cfg['aug_policy']
    total_iter   = cfg['total_iter']
    save_every   = cfg['save_every']
    log_every    = cfg.get('log_every', 50)
    vis_every    = cfg.get('vis_every', 1000)
    percept_w    = cfg.get('percept_weight', 1.0)
    ema_decay    = cfg.get('ema_decay', 0.999)

    G.train(); D.train()
    t0 = time.time()

    try:
        for iteration in range(start_iter, total_iter):
            _state['iteration'] = iteration

            # ── Sample real batch ────────────────────────────────────────
            real, _ = next(loader)
            real = real.to(device, non_blocking=True)

            # Choose a random quadrant for the reconstruction partial task
            part = random.randint(0, 3)

            # ── Discriminator update ─────────────────────────────────────
            opt_D.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                # --- Real images ---
                real_aug = DiffAugment(real, aug_policy)
                pred_real, (rec_full, rec_small, rec_part) = D(real_aug, part=part)

                d_real_loss = hinge_loss_real(pred_real)

                # Resize real to match decoder output sizes
                real_128  = F.interpolate(real, rec_full.shape[2:],  mode='bilinear', align_corners=False)
                real_32   = F.interpolate(real, rec_small.shape[2:], mode='bilinear', align_corners=False)
                real_crop = F.interpolate(
                    crop_quadrant(real, part), rec_part.shape[2:],
                    mode='bilinear', align_corners=False
                )

                recon_loss = (
                    percept(rec_full,  real_128.detach()) +
                    percept(rec_small, real_32.detach())  +
                    percept(rec_part,  real_crop.detach())
                )

                # --- Fake images (detach — no G gradient here) ---
                z    = torch.randn(real.size(0), cfg['nz'], device=device)
                fake = G(z).detach()
                fake_aug  = DiffAugment(fake, aug_policy)
                pred_fake = D(fake_aug)

                d_fake_loss = hinge_loss_fake(pred_fake)
                d_loss      = d_real_loss + d_fake_loss + percept_w * recon_loss

            scaler_D.scale(d_loss).backward()
            scaler_D.unscale_(opt_D)
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            scaler_D.step(opt_D)
            scaler_D.update()

            # ── Generator update ─────────────────────────────────────────
            opt_G.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                z    = torch.randn(real.size(0), cfg['nz'], device=device)
                fake = G(z)
                fake_aug  = DiffAugment(fake, aug_policy)
                pred_fake = D(fake_aug)
                g_loss    = -pred_fake.mean()   # Non-saturating adversarial loss

            scaler_G.scale(g_loss).backward()
            scaler_G.unscale_(opt_G)
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            scaler_G.step(opt_G)
            scaler_G.update()

            # ── EMA update ───────────────────────────────────────────────
            update_ema(G_ema, G, decay=ema_decay)

            # ── Collect scalar losses ────────────────────────────────────
            g_loss_val = g_loss.item()
            d_loss_val = d_loss.item()
            _state['loss_g'] = g_loss_val
            _state['loss_d'] = d_loss_val

            # ── Logging ──────────────────────────────────────────────────
            if iteration % log_every == 0:
                elapsed = time.time() - t0
                its_per_sec = (iteration - start_iter + 1) / max(elapsed, 1)
                eta_sec = (total_iter - iteration) / max(its_per_sec, 1e-6)
                print(
                    f"[{iteration:7d}/{total_iter}] "
                    f"G={g_loss_val:+.4f}  D={d_loss_val:+.4f}  "
                    f"D_real={d_real_loss.item():.4f}  D_fake={d_fake_loss.item():.4f}  "
                    f"Recon={recon_loss.item():.4f}  "
                    f"{its_per_sec:.1f} it/s  ETA {eta_sec/60:.0f} min"
                )
                logger.log(iteration, g_loss_val, d_loss_val)

            # ── Visualise ────────────────────────────────────────────────
            if iteration % vis_every == 0:
                out = img_dir / f'samples_{iteration:07d}.png'
                save_image_grid(G_ema, out, device, cfg['nz'], fixed_z=fixed_z)
                print(f"[Visual] Saved: {out.name}")
                logger.save_plot()

            # ── Checkpoint ───────────────────────────────────────────────
            if (iteration + 1) % save_every == 0:
                ckpt_mgr.save(
                    G, D, G_ema, opt_G, opt_D,
                    iteration = iteration + 1,
                    loss_g    = g_loss_val,
                    loss_d    = d_loss_val,
                    scaler_G  = scaler_G,
                    scaler_D  = scaler_D,
                )

            # ── Periodic VRAM flush ──────────────────────────────────────
            if iteration % 2000 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n[Train] Interrupted by user — saving checkpoint…")
    except Exception as exc:
        print(f"\n[Train] Exception: {exc}")
        traceback.print_exc()
        print("[Train] Attempting emergency checkpoint save…")
    finally:
        ckpt_mgr.save(
            G, D, G_ema, opt_G, opt_D,
            iteration    = _state['iteration'],
            loss_g       = _state['loss_g'],
            loss_d       = _state['loss_d'],
            scaler_G     = scaler_G,
            scaler_D     = scaler_D,
            is_emergency = True,
        )
        logger.save_plot()
        logger.close()
        print("[Train] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FastGAN + DiffAugment training for NEU-DET defect synthesis"
    )
    p.add_argument('--config',      default='configs/neu_det_128.yaml',
                   help='Path to YAML config file')
    p.add_argument('--data_path',   type=str,
                   help='Override config: path to NEU-DET dataset root')
    p.add_argument('--save_dir',    type=str,
                   help='Override config: directory for checkpoints and images')
    p.add_argument('--total_iter',  type=int,
                   help='Override config: total training iterations')
    p.add_argument('--batch_size',  type=int,
                   help='Override config: batch size')
    p.add_argument('--aug_policy',  type=str,
                   help="Override config: DiffAugment policy, e.g. 'color,translation'")
    p.add_argument('--n_per_class', type=int,
                   help='Override config: images per class (default 25)')
    p.add_argument('--resume',      action='store_true',
                   help='Force resume even if no checkpoint found (no-op if none)')
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """Load YAML config and apply CLI overrides."""
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides (only if explicitly provided)
    overrides = {
        'data_path':   args.data_path,
        'save_dir':    args.save_dir,
        'total_iter':  args.total_iter,
        'batch_size':  args.batch_size,
        'aug_policy':  args.aug_policy,
        'n_per_class': args.n_per_class,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val

    # Defaults for optional keys
    cfg.setdefault('seed',          42)
    cfg.setdefault('num_workers',   2)
    cfg.setdefault('log_every',     50)
    cfg.setdefault('vis_every',     1000)
    cfg.setdefault('percept_weight', 1.0)
    cfg.setdefault('ema_decay',     0.999)

    print("[Config]")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    return cfg


if __name__ == '__main__':
    args = parse_args()
    cfg  = load_config(args)
    train(cfg)
