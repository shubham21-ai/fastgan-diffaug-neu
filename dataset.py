"""
dataset.py — NEU Surface Defect Dataset Loader
===============================================

This module handles loading the NEU-DET dataset in a way that faithfully
simulates the extreme data-scarcity scenario (25 images/class, 150 total).

NEU-DET Dataset Structure
--------------------------
The NEU Surface Defect Dataset has two layouts depending on the Kaggle source:

Layout A (classification):
    neu-det/
    └── images/
        ├── Cr/          ← Crazing
        ├── In/          ← Inclusion
        ├── Pa/          ← Patches
        ├── PS/          ← Pitted surface
        ├── RS/          ← Rolled-in scale
        └── Sc/          ← Scratches

Layout B (detection, some Kaggle uploads):
    NEU-DET/
    └── train/
        ├── images/
        └── annotations/

This loader supports both layouts by scanning for image files recursively
and grouping them by their parent directory name.

Design Decisions
----------------
- We use 25 images per class (150 total) to simulate the worst-case scenario
  that FastGAN + DiffAugment needs to handle.
- Pre-DiffAugment transforms include horizontal flip, vertical flip, and
  slight rotation — these are applied by the DataLoader before DiffAugment
  sees the image. They expand effective dataset diversity at the input level.
- Images are loaded as RGB (even if originally grayscale) because the
  Generator produces 3-channel output for maximum downstream compatibility.
- Reproducible subset selection: pass `seed` to get the same 25/class split
  across runs, enabling fair comparison between training configurations.
"""

import os
import random
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


# Image extensions to look for when scanning dataset directories
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


class NEUDefectDataset(Dataset):
    """PyTorch Dataset for the NEU Surface Defect Dataset.

    Scans a root directory for subdirectories, treating each subdirectory as
    one defect class. Supports selecting a fixed number of images per class
    (``n_per_class``) to simulate limited-data training.

    Args:
        root:        Path to the dataset root (contains one folder per class).
        n_per_class: Number of images to select from each class. If ``None``
                     or larger than the available count, all images are used.
        image_size:  Images are resized to ``(image_size, image_size)``.
        augment:     If True, apply random horizontal/vertical flips and
                     slight rotation. Set False for evaluation/generation.
        seed:        Random seed for reproducible subset selection.
    """

    def __init__(
        self,
        root: str | Path,
        n_per_class:  Optional[int] = 25,
        image_size:   int           = 128,
        augment:      bool          = True,
        seed:         int           = 42,
    ) -> None:
        super().__init__()
        self.root        = Path(root)
        self.image_size  = image_size
        self.augment     = augment

        # ── Build class → file list mapping ──────────────────────────────────
        self.classes, self.class_to_idx = self._discover_classes()
        if not self.classes:
            raise RuntimeError(
                f"No class subdirectories found under '{self.root}'. "
                "Check that the path points to the dataset root."
            )

        all_samples: list[tuple[Path, int]] = []
        rng = random.Random(seed)

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = self.root / cls_name
            imgs = sorted([
                p for p in cls_dir.iterdir()
                if p.suffix.lower() in _IMAGE_EXTS
            ])

            if not imgs:
                print(f"  [WARNING] No images found in class '{cls_name}'")
                continue

            # Select subset
            if n_per_class is not None and n_per_class < len(imgs):
                imgs = rng.sample(imgs, n_per_class)
                imgs.sort()  # keep order deterministic post-sample

            for img_path in imgs:
                all_samples.append((img_path, cls_idx))

        self.samples = all_samples

        # ── Transforms ───────────────────────────────────────────────────────
        aug_ops: list = []
        if augment:
            aug_ops = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),            # [0, 255] → [0.0, 1.0]
            *aug_ops,
            transforms.Normalize(             # [0, 1] → [-1, 1]
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    def _discover_classes(self) -> tuple[list[str], dict[str, int]]:
        """Return sorted class names and their integer indices."""
        classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        return classes, {c: i for i, c in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), label

    def __repr__(self) -> str:
        return (
            f"NEUDefectDataset(root='{self.root}', "
            f"classes={self.classes}, "
            f"n_samples={len(self.samples)})"
        )


class InfiniteDataLoader:
    """Wraps a DataLoader to iterate indefinitely (no epoch boundary).

    GAN training is measured in iterations, not epochs. This wrapper ensures
    we always have a fresh batch available without manually tracking epoch ends.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self._iter      = iter(dataloader)

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            return next(self._iter)

    def __iter__(self):
        return self


def get_dataloader(
    root:         str | Path,
    n_per_class:  Optional[int] = 25,
    image_size:   int           = 128,
    batch_size:   int           = 16,
    num_workers:  int           = 2,
    seed:         int           = 42,
) -> tuple[InfiniteDataLoader, NEUDefectDataset]:
    """Build the training dataloader for the NEU-DET dataset.

    Args:
        root:        Path to the dataset root.
        n_per_class: Images per class (25 for limited-data regime).
        image_size:  Target image resolution.
        batch_size:  Batch size for training.
        num_workers: DataLoader worker processes. Use 2 for Kaggle P100.
        seed:        Reproducibility seed.

    Returns:
        (InfiniteDataLoader, NEUDefectDataset) — the loader and the dataset
        object (useful for inspecting class names and sample counts).
    """
    dataset = NEUDefectDataset(
        root        = root,
        n_per_class = n_per_class,
        image_size  = image_size,
        augment     = True,
        seed        = seed,
    )

    print(f"[Dataset] {dataset}")
    print(f"[Dataset] Classes: {dataset.classes}")
    print(f"[Dataset] Total training samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,   # drop partial batches for stable BN stats
    )

    return InfiniteDataLoader(loader), dataset


def find_dataset_root(base: str | Path) -> Path:
    """Attempt to auto-detect the NEU-DET root directory.

    Searches ``base`` for a directory containing image-filled subdirectories.
    Handles common Kaggle extraction layouts (e.g. ``neu-det/images/``).

    Args:
        base: Starting directory to search.

    Returns:
        Detected dataset root as a ``Path``.

    Raises:
        FileNotFoundError: If no valid root is found within 3 levels.
    """
    base = Path(base)
    candidates = [base]

    # BFS up to 3 levels deep
    for depth in range(3):
        next_candidates = []
        for cand in candidates:
            if not cand.is_dir():
                continue
            subdirs = [d for d in cand.iterdir() if d.is_dir()]
            # A valid root has subdirectories that contain images
            valid_sub = [
                d for d in subdirs
                if any(f.suffix.lower() in _IMAGE_EXTS for f in d.iterdir()
                       if f.is_file())
            ]
            if len(valid_sub) >= 2:
                print(f"[Dataset] Auto-detected root: {cand}")
                return cand
            next_candidates.extend(subdirs)
        candidates = next_candidates

    raise FileNotFoundError(
        f"Could not auto-detect NEU-DET root under '{base}'. "
        "Please set --data_path explicitly."
    )
