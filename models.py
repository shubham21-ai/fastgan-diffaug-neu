"""
models.py — Generator and Discriminator Architectures
======================================================

Architecture Overview
---------------------

**Generator** (Skip-Layer Excitation Network)
    Inspired by FastGAN (Liu et al., ICLR 2021). Key innovations:

    1. GLU (Gated Linear Unit) activations — each UpBlock doubles spatial
       resolution but halves channel count via channel-splitting gating.
       This gives the generator a richer representational vocabulary than
       plain ReLU, at no extra parameter cost.

    2. NoiseInjection — per-pixel learnable-weighted Gaussian noise added
       after each UpBlock. For texture-heavy defects (cracks, porosity), this
       produces naturalistic stochastic surface variation.

    3. SEBlock (Skip-Layer Excitation) — low-resolution feature maps from
       early generator layers modulate (excite) high-resolution feature maps
       later in the network. This creates long-range gradient highways that
       prevent mode collapse on tiny datasets by keeping the full network
       active during backprop.

    Resolution ladder (128×128 output):  4 → 8 → 16 → 32 → 64 → 128
    SLE connections: feat₄ → feat₃₂,   feat₈ → feat₆₄

**Discriminator** (Dual-Head Self-Supervised Reconstructor)
    Inspired by FastGAN's discriminator design:

    1. Main head — spectral-normalised convolutions downsample the input
       to a single real/fake prediction score.

    2. Decoder heads (self-supervised) — decode intermediate feature maps
       back to pixel space (full image + small + quadrant patch). The
       reconstruction losses prevent the discriminator from memorising the
       small training set by forcing it to learn rich structural features.
       Applied only on real images — not fake.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ── Shared utilities ──────────────────────────────────────────────────────────

def weights_init(m: nn.Module) -> None:
    """Initialise Conv and BatchNorm layers with standard GAN values."""
    cls = m.__class__.__name__
    if 'Conv' in cls:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except AttributeError:
            pass
    elif 'BatchNorm' in cls:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def crop_quadrant(x: torch.Tensor, part: int) -> torch.Tensor:
    """Extract one of four equal quadrants from a batch of images.

    Args:
        x:    (B, C, H, W) — spatial dims must be even.
        part: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right.

    Returns:
        (B, C, H//2, W//2) tensor.
    """
    H, W = x.shape[2] // 2, x.shape[3] // 2
    regions = {
        0: x[:, :, :H, :W],
        1: x[:, :, :H, W:],
        2: x[:, :, H:, :W],
        3: x[:, :, H:, W:],
    }
    return regions[part]


# ── Generator building blocks ─────────────────────────────────────────────────

class PixelNorm(nn.Module):
    """Normalise latent vector z across the channel dimension.

    Keeps the latent space on a unit hypersphere, preventing the generator
    from exploiting magnitude differences in z to control style, which leads
    to more stable early training.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class GLU(nn.Module):
    """Gated Linear Unit — splits channels in half and applies sigmoid gating.

    Channel halving: for a tensor with 2C channels, outputs C channels.
    The first C channels form the "value" and the second C are the "gate".
    This is the core non-linearity in each UpBlock.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nc = x.size(1)
        assert nc % 2 == 0, f"GLU requires even channel count, got {nc}"
        return x[:, :nc // 2] * torch.sigmoid(x[:, nc // 2:])


class NoiseInjection(nn.Module):
    """Add spatially-independent Gaussian noise scaled by a learnable weight.

    Starting at weight=0, the generator initially ignores noise. As training
    progresses the model learns to use controlled amounts of stochasticity,
    which is essential for producing varied surface texture in defect images.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            B, _, H, W = x.shape
            noise = torch.randn(B, 1, H, W, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class InitLayer(nn.Module):
    """Map latent vector z ∈ ℝ^nz to a (ch, 4, 4) feature volume.

    Uses a transposed convolution with stride 1 and no padding to expand a
    (B, nz, 1, 1) tensor to (B, ch, 4, 4), followed by BatchNorm + GLU.
    """

    def __init__(self, nz: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(nz, out_ch * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            GLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.block(z.view(z.size(0), -1, 1, 1))  # (B, out_ch, 4, 4)


class UpBlock(nn.Module):
    """Double spatial resolution and halve channel count.

    Pipeline: nearest upsample → 3×3 conv → BN → GLU → noise injection.

    Using nearest upsample (vs. transposed conv) avoids checkerboard
    artefacts common in defect texture synthesis. BN + GLU provides smooth
    gradient flow during the early training iterations when only 25 images
    are seen per epoch.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch * 2, 3, 1, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.glu  = GLU()
        self.noise = NoiseInjection()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.glu(x)
        x = self.noise(x)
        return x


class SEBlock(nn.Module):
    """Skip-Layer Excitation — long-range feature modulation.

    Takes a low-resolution feature map ``feat_small`` and produces a
    per-channel attention map that scales the high-resolution ``feat_big``.

    Mechanism:
        feat_small  (B, ch_in, small_h, small_w)
            → AdaptiveAvgPool(4)
            → Conv 4×4 (ch_in → ch_out)   [squeeze to single value per ch]
            → SiLU
            → Conv 1×1 (ch_out → ch_out)  [channel mixing]
            → Sigmoid
            → Scale feat_big (B, ch_out, big_h, big_w)

    Why this prevents mode collapse: The gradient of feat_big flows back
    through the scale gate all the way to feat_small. Even when training
    with 25 images, the entire generator receives gradients, preventing
    "dead channels" that occur in plain forward networks.
    """

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_small: torch.Tensor, feat_big: torch.Tensor) -> torch.Tensor:
        return feat_big * self.se(feat_small)


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """Full multi-scale generator with SLE skip connections.

    Resolution progression:
        z (nz,) → InitLayer → (ch₄, 4, 4)
                             → UpBlock → (ch₈, 8, 8)
                             → UpBlock → (ch₁₆, 16, 16)
                             → UpBlock → (ch₃₂, 32, 32) ←── SEBlock(ch₄ → ch₃₂)
                             → UpBlock → (ch₆₄, 64, 64) ←── SEBlock(ch₈ → ch₆₄)
                             → UpBlock → (ch₁₂₈, 128, 128)
                             → Conv3×3 + Tanh → (3, 128, 128)

    Args:
        ngf:     Base number of generator filters. Default 64.
        nz:      Latent dimension. Default 256.
        im_size: Output image resolution (square). Default 128.
    """

    def __init__(self, ngf: int = 64, nz: int = 256, im_size: int = 128) -> None:
        super().__init__()
        self.nz = nz

        # Channel schedule (larger = more capacity at small resolutions)
        ch = {
            4:   ngf * 8,     # 512  — richest, captures global structure
            8:   ngf * 4,     # 256
            16:  ngf * 2,     # 128
            32:  ngf,         # 64
            64:  ngf // 2,    # 32
            128: ngf // 4,    # 16   — thinnest, just needs to produce RGB
        }
        self._ch = ch

        # Stem: z → 4×4 features
        self.init = InitLayer(nz, ch[4])

        # Upsampling blocks
        self.up_4_8    = UpBlock(ch[4],   ch[8])
        self.up_8_16   = UpBlock(ch[8],   ch[16])
        self.up_16_32  = UpBlock(ch[16],  ch[32])
        self.up_32_64  = UpBlock(ch[32],  ch[64])
        self.up_64_128 = UpBlock(ch[64],  ch[128])

        # SLE skip connections
        # feat₄ (512 ch) excites feat₃₂ (64 ch)
        self.se_4_32 = SEBlock(ch[4], ch[32])
        # feat₈ (256 ch) excites feat₆₄ (32 ch)
        self.se_8_64 = SEBlock(ch[8], ch[64])

        # Output head: convert final feature map to RGB image
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch[128], 3, 3, 1, 1),
            nn.Tanh(),
        )

        # Apply weight initialisation
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from latent vectors.

        Args:
            z: Latent tensor (B, nz).

        Returns:
            Generated images (B, 3, im_size, im_size) in [-1, 1].
        """
        f4  = self.init(z)               # (B, 512, 4, 4)

        f8  = self.up_4_8(f4)            # (B, 256, 8, 8)
        f16 = self.up_8_16(f8)           # (B, 128, 16, 16)
        f32 = self.up_16_32(f16)         # (B, 64, 32, 32)
        f32 = self.se_4_32(f4, f32)      # SLE: feat₄ excites feat₃₂

        f64 = self.up_32_64(f32)         # (B, 32, 64, 64)
        f64 = self.se_8_64(f8, f64)      # SLE: feat₈ excites feat₆₄

        f128 = self.up_64_128(f64)       # (B, 16, 128, 128)

        return self.to_rgb(f128)          # (B, 3, 128, 128)


# ── Discriminator building blocks ─────────────────────────────────────────────

class DownBlock(nn.Module):
    """Halve spatial resolution via stride-2 spectral-normalised convolution.

    Spectral normalisation constrains the Lipschitz constant of the
    discriminator, which stabilises training without needing BatchNorm
    (BatchNorm in D can cause gradient instabilities with small batches).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleDecoder(nn.Module):
    """Lightweight decoder for discriminator self-supervised reconstruction.

    Upsamples from a small feature map back to pixel space using bilinear
    interpolation + convolutions. Bilinear upsampling (vs. transposed conv)
    avoids checkerboard artefacts in the decoded images.

    Args:
        ch_in:        Number of input channels.
        ch_out:       Number of output channels (typically 3 for RGB).
        num_upsample: Number of 2× upsampling steps to perform.
    """

    def __init__(self, ch_in: int, ch_out: int = 3, num_upsample: int = 4) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        ch = ch_in
        for _ in range(num_upsample):
            ch_next = max(ch // 2, 32)
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(ch, ch_next, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next

        # Final conv to target channels + Tanh to match image range [-1, 1]
        layers += [
            nn.Conv2d(ch, ch_out, 3, 1, 1),
            nn.Tanh(),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x)


# ── Discriminator ─────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """Dual-head discriminator with self-supervised reconstruction.

    Main path:
        (3, 128) → (ndf, 64) → (ndf×2, 32) → (ndf×4, 16)
                 → (ndf×8, 8) → (ndf×8, 4) → pred (B, 1)

    Reconstruction decoders (real images only):
        rec_full  — decoded from f₄  (4×4)   to (3, 128, 128)
        rec_small — decoded from f₃₂ (32×32)  to (3, 32, 32)   [conv only]
        rec_part  — decoded from f₁₆[quadrant] (8×8) to (3, 32, 32)

    Forward returns:
        part=None  → pred                                (fake mode)
        part=0..3  → pred, (rec_full, rec_small, rec_part) (real mode)

    Args:
        ndf:     Base number of discriminator filters. Default 64.
        im_size: Expected input resolution (square). Default 128.
    """

    def __init__(self, ndf: int = 64, im_size: int = 128) -> None:
        super().__init__()
        self.ndf = ndf

        # ── Downsampling path ──
        # Resolution: 128 → 64 → 32 → 16 → 8 → 4
        self.down1 = DownBlock(3,        ndf)         # (ndf,   64, 64)
        self.down2 = DownBlock(ndf,      ndf * 2)     # (ndf×2, 32, 32) ← rec_small src
        self.down3 = DownBlock(ndf * 2,  ndf * 4)     # (ndf×4, 16, 16) ← rec_part src
        self.down4 = DownBlock(ndf * 4,  ndf * 8)     # (ndf×8,  8,  8)
        self.down5 = DownBlock(ndf * 8,  ndf * 8)     # (ndf×8,  4,  4) ← rec_full src

        # ── Real/fake prediction head ──
        # 4×4 features → single score per image
        self.pred_head = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0)),   # (B, 1, 1, 1)
            nn.Flatten(),                                       # (B, 1)
        )

        # ── Reconstruction decoders ──
        # Full-image reconstruction: from (ndf×8, 4, 4) → (3, 128, 128)
        # 4 → 8 → 16 → 32 → 64 → 128 = 5 upsampling steps
        n_full = int(math.log2(im_size // 4))
        self.dec_full = SimpleDecoder(ndf * 8, ch_out=3, num_upsample=n_full)

        # Small reconstruction: from (ndf×2, 32, 32) → (3, 32, 32), no upsampling
        self.dec_small = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 2, ndf, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, 3, 3, 1, 1),
            nn.Tanh(),
        )

        # Part reconstruction: from (ndf×4, 8, 8) → (3, 32, 32) — 2 upsampling steps
        # (f16 quadrant is (ndf×4, 8, 8) after halving spatial dims)
        self.dec_part = SimpleDecoder(ndf * 4, ch_out=3, num_upsample=2)

        self.apply(weights_init)

    def forward(
        self,
        x:    torch.Tensor,
        part: int = None,
    ):
        """Forward pass.

        Args:
            x:    Input images (B, 3, H, W), values in [-1, 1].
            part: If not None (0–3), compute self-supervised reconstructions
                  for the given quadrant. Used only for real images.

        Returns:
            ``part is None``  → pred (B, 1)
            ``part is int``   → pred, (rec_full, rec_small, rec_part)
        """
        f64  = self.down1(x)    # (B, ndf,   64, 64)
        f32  = self.down2(f64)  # (B, ndf×2, 32, 32)
        f16  = self.down3(f32)  # (B, ndf×4, 16, 16)
        f8   = self.down4(f16)  # (B, ndf×8,  8,  8)
        f4   = self.down5(f8)   # (B, ndf×8,  4,  4)

        pred = self.pred_head(f4)  # (B, 1)

        if part is None:
            return pred  # fake images: only need the score

        # ── Self-supervised reconstruction (real images only) ──
        rec_full  = self.dec_full(f4)    # (B, 3, 128, 128)
        rec_small = self.dec_small(f32)  # (B, 3, 32, 32)

        # Crop feature map f16 to the selected quadrant, then decode
        f16_crop = crop_quadrant(f16, part)   # (B, ndf×4, 8, 8)
        rec_part = self.dec_part(f16_crop)    # (B, 3, 32, 32)

        return pred, (rec_full, rec_small, rec_part)
