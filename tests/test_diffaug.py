import torch
import pytest

from diffaug import (
    rand_brightness,
    rand_saturation,
    rand_contrast,
    rand_translation,
    rand_cutout,
    DiffAugment
)


@pytest.fixture
def dummy_batch():
    # Batch of 4 images, 3 channels, 32x32 size
    # Values mostly in [-1, 1] to simulate GAN inputs
    return torch.randn(4, 3, 32, 32).clamp(-1.0, 1.0)


def test_color_augmentations_shape_and_dtype(dummy_batch):
    for fn in [rand_brightness, rand_saturation, rand_contrast]:
        out = fn(dummy_batch)
        assert out.shape == dummy_batch.shape, f"{fn.__name__} changed shape"
        assert out.dtype == dummy_batch.dtype, f"{fn.__name__} changed dtype"
        assert out.device == dummy_batch.device, f"{fn.__name__} changed device"


def test_translation_augmentation(dummy_batch):
    out = rand_translation(dummy_batch, ratio=0.125)
    assert out.shape == dummy_batch.shape, "rand_translation changed shape"
    assert out.dtype == dummy_batch.dtype, "rand_translation changed dtype"


def test_cutout_augmentation(dummy_batch):
    out = rand_cutout(dummy_batch, ratio=0.5)
    assert out.shape == dummy_batch.shape, "rand_cutout changed shape"
    assert out.dtype == dummy_batch.dtype, "rand_cutout changed dtype"


def test_diffaugment_pipeline(dummy_batch):
    # Test with full default policy
    out = DiffAugment(dummy_batch, policy='color,translation,cutout')
    assert out.shape == dummy_batch.shape, "DiffAugment changed shape"
    assert out.dtype == dummy_batch.dtype, "DiffAugment changed dtype"
    
    # Test gradient flow
    dummy_batch.requires_grad_(True)
    out = DiffAugment(dummy_batch, policy='color,translation,cutout')
    loss = out.mean()
    loss.backward()
    
    assert dummy_batch.grad is not None, "Gradients did not flow through DiffAugment"
    assert dummy_batch.grad.shape == dummy_batch.shape, "Gradient shape mismatch"


def test_diffaugment_empty_policy(dummy_batch):
    out = DiffAugment(dummy_batch, policy='')
    assert torch.allclose(out, dummy_batch), "Empty policy modified input"
