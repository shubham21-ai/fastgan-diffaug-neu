# FastGAN + DiffAugment for Extreme Data-Scarce Defect Synthesis

This repository implements a data-efficient generative framework using **FastGAN** (Skip-Layer Excitation) and **DiffAugment** (Differentiable Augmentation) to synthesize high-quality, physically realistic images of metal casting defects from extremely limited datasets (e.g., 25 images per class).

This framework is specifically tailored for the **Kaggle free-tier environment** (P100/T4 GPUs) and includes robust crash-recovery mechanisms and mixed-precision (AMP) training.

## Features
- **FastGAN Architecture:** 
  - Skip-Layer Excitation (SLE) in the Generator for stable gradient flow.
  - Dual-head self-supervised reconstruction in the Discriminator to prevent overfitting.
- **DiffAugment:** 
  - Re-implemented in pure PyTorch (Color, Translation, Cutout).
  - Applied symmetrically to both real and fake images *before* the Discriminator.
- **Kaggle-Ready:**
  - Automated Checkpointing every 500 iterations.
  - Emergency save handler on SIGTERM (handles Kaggle 9-hour session timeouts).
  - Out-of-the-box `kaggle_entry.py` notebook script.
- **Evaluation:** Built-in `clean-fid` evaluation script for FID score calculation.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/fast_gan_diffAug.git
cd fast_gan_diffAug
pip install -r requirements.txt
```

## Quick Start (Kaggle)

If you are running on Kaggle, simply open `setup/kaggle_entry.py` and copy each highlighted block into a separate Kaggle Notebook cell. Ensure you have added the NEU Surface Defect Database as an input dataset.

## Local / Server Usage

### 1. Training
Configure your dataset path in `configs/neu_det_128.yaml`, then run:
```bash
python train.py --config configs/neu_det_128.yaml
```

To override paths without editing the config:
```bash
python train.py \\
    --data_path /path/to/NEU-DET \\
    --save_dir ./working_dir \\
    --config configs/neu_det_128.yaml
```

### 2. Generating Synthetic Images
Once trained, use the latest checkpoint to generate clean, un-augmented synthetic images for downstream tasks:
```bash
python generate.py \\
    --ckpt working_dir/checkpoints/ckpt_iter_0100000.pt \\
    --out_dir synthetic_images/ \\
    --n_images 500
```

### 3. Evaluation
Compute the Fréchet Inception Distance (FID) to evaluate generation quality:
```bash
python evaluate.py \\
    --real_dir /path/to/NEU-DET/train \\
    --fake_dir synthetic_images/
```

## Directory Structure
```
├── configs/             # YAML configurations
├── dataset.py           # NEU-DET data loader & 25/class sampler
├── diffaug.py           # Differentiable augmentation logic
├── evaluate.py          # FID score computation
├── generate.py          # Script to generate synthetic datasets
├── models.py            # FastGAN Generator (SLE) and Discriminator
├── setup/               # Kaggle entry point scripts
├── tests/               # Pytest unit tests for augmentations and models
├── train.py             # Main training loop with AMP and crash-safety
└── utils/               # Checkpointing and visualization tools
```

## References
- **FastGAN:** [Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis](https://arxiv.org/abs/2101.04775) (ICLR 2021)
- **DiffAugment:** [Differentiable Augmentation for Data-Efficient GAN Training](https://arxiv.org/abs/2006.10738) (NeurIPS 2020)
