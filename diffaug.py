"""
diffaug.py — Differentiable Augmentation for Data-Efficient GAN Training
=========================================================================

Re-implemented from first principles in pure PyTorch.

Design Rationale
----------------
When training a GAN with very few images (25/class), the discriminator rapidly
memorises the training set — it learns pixel-level shortcuts rather than
semantic structure. This is called "discriminator overfitting."

DiffAugment solves this by applying the *same* differentiable augmentation to
**both** real and fake images before the discriminator. Because every operation
here is a PyTorch tensor op (no PIL/OpenCV), gradients flow cleanly back
through the augmentation to the generator. The generator therefore learns to
produce un-augmented, realistic images whose augmented versions fool the
discriminator.

Reference: Zhao et al., "Differentiable Augmentation for Data-Efficient GAN
Training", NeurIPS 2020 — https://arxiv.org/abs/2006.10738

Usage
-----
    from diffaug import DiffAugment

    # Apply during both D and G update steps
    real_aug = DiffAugment(real, policy='color,translation,cutout')
    fake_aug = DiffAugment(fake.detach(), policy='color,translation,cutout')
"""

import torch
import torch.nn.functional as F


# ── Public API ────────────────────────────────────────────────────────────────

def DiffAugment(x: torch.Tensor, policy: str = 'color,translation,cutout') -> torch.Tensor:
    """Apply differentiable augmentations to a batch of images.

    Args:
        x:      Image tensor (B, C, H, W), expected in the range [-1, 1].
        policy: Comma-separated list of augmentation groups to apply.
                Available: ``'color'``, ``'translation'``, ``'cutout'``.
                Order matters — augmentations are applied left-to-right.

    Returns:
        Augmented tensor with the same shape and device as ``x``.
        Gradients are propagated through all augmentation ops.
    """
    if not policy:
        return x

    for group in policy.split(','):
        group = group.strip()
        if group not in AUGMENT_FNS:
            raise ValueError(
                f"Unknown DiffAugment policy '{group}'. "
                f"Valid options: {list(AUGMENT_FNS.keys())}"
            )
        for fn in AUGMENT_FNS[group]:
            x = fn(x)

    return x.contiguous()


# ── Color augmentations ───────────────────────────────────────────────────────

def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    """Add a per-image random brightness offset in [-0.5, +0.5].

    The offset is sampled independently for each image in the batch, so the
    discriminator cannot rely on absolute brightness as a real/fake cue.
    """
    # (B, 1, 1, 1) broadcasts over (C, H, W)
    offset = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5
    return x + offset


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    """Scale per-image saturation by a random factor in [0, 2].

    Computed by pulling each pixel toward the channel mean. A scale > 1
    increases saturation; < 1 desaturates toward grayscale; 0 = full grey.
    """
    x_mean = x.mean(dim=1, keepdim=True)  # luminance proxy — (B, 1, H, W)
    scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2.0
    return (x - x_mean) * scale + x_mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    """Scale per-image contrast by a random factor in [0.5, 1.5].

    Computed by pulling each pixel toward the spatial mean. A scale > 1 sharpens
    the dynamic range; < 1 compresses it.
    """
    # Global spatial mean per image-channel: (B, C, 1, 1)
    x_mean = x.mean(dim=[2, 3], keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return (x - x_mean) * scale + x_mean


# ── Translation augmentations ─────────────────────────────────────────────────

def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Apply a random integer-pixel translation per image.

    Images are shifted by a random (dh, dw) within ±(ratio × H/W) pixels.
    The image is first zero-padded by 1 pixel on all sides and then re-cropped
    to the original size using integer-index lookups. This keeps the operation
    fully differentiable — the indexing act as a gather op which PyTorch's
    autograd handles correctly.

    Args:
        x:     Image tensor (B, C, H, W).
        ratio: Maximum fractional shift (e.g. 0.125 → ±12.5% of image size).
    """
    B, C, H, W = x.shape
    shift_h = int(H * ratio + 0.5)
    shift_w = int(W * ratio + 0.5)

    # Random integer shift per sample in the batch
    dh = torch.randint(-shift_h, shift_h + 1, (B, 1, 1), device=x.device)
    dw = torch.randint(-shift_w, shift_w + 1, (B, 1, 1), device=x.device)

    # Pad by 1 on every edge (fills with 0.0 / "gray" since images in [-1,1])
    x_pad = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)  # (B, C, H+2, W+2)

    # Build coordinate grids — all on the same device as x
    gb = torch.arange(B, device=x.device).view(B, 1, 1)   # batch index
    gi = torch.arange(H, device=x.device).view(1, H, 1)   # height index
    gj = torch.arange(W, device=x.device).view(1, 1, W)   # width index

    # +1 accounts for the 1-pixel padding offset; clamp keeps indices in bounds
    idx_i = (gi + dh + 1).clamp(0, H + 1)  # (B, H, W) after broadcast
    idx_j = (gj + dw + 1).clamp(0, W + 1)

    # Advanced indexing: permute to (B, H+2, W+2, C) → index → permute back
    x_out = x_pad.permute(0, 2, 3, 1)[gb, idx_i, idx_j, :]
    return x_out.permute(0, 3, 1, 2).contiguous()


# ── Cutout augmentations ──────────────────────────────────────────────────────

def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Mask a random rectangular region per image with its own channel mean.

    For defect images, this is especially useful: it forces the discriminator
    to reason about texture and structure globally rather than fixating on a
    single salient region (like the exact shape of a crack).

    The masked region is filled with the per-image channel mean so that the
    augmentation does not artificially introduce a hard zero-boundary that the
    discriminator could use as a cue.

    Args:
        x:     Image tensor (B, C, H, W).
        ratio: Side-length ratio of the cutout box (e.g. 0.5 → 50% of H/W).
    """
    B, C, H, W = x.shape
    cut_h = int(H * ratio + 0.5)
    cut_w = int(W * ratio + 0.5)

    # Random box top-left corner (per image)
    off_h = torch.randint(0, H + (1 - cut_h % 2), (B, 1, 1), device=x.device)
    off_w = torch.randint(0, W + (1 - cut_w % 2), (B, 1, 1), device=x.device)

    # Build coordinate grids
    gb = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1, 1)
    gi = torch.arange(H, dtype=torch.long, device=x.device).view(1, H, 1)
    gj = torch.arange(W, dtype=torch.long, device=x.device).view(1, 1, W)

    # Boolean mask: True inside the cutout box for each image
    in_box = (
        (gi >= off_h - cut_h // 2) & (gi < off_h + (cut_h + 1) // 2) &
        (gj >= off_w - cut_w // 2) & (gj < off_w + (cut_w + 1) // 2)
    )  # (B, H, W)

    # Per-image channel mean as fill value
    fill = x.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
    mask = in_box.unsqueeze(1).to(x.dtype)   # (B, 1, H, W) — broadcasts over C

    return x * (1.0 - mask) + fill * mask


# ── Augmentation registry ─────────────────────────────────────────────────────

AUGMENT_FNS: dict = {
    # Applied sequentially within each group
    'color':       [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout':      [rand_cutout],
}
