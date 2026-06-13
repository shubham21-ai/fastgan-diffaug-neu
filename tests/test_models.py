import torch
import pytest

from models import Generator, Discriminator


@pytest.fixture
def device():
    # Run tests on CPU unless GPU is explicitly requested
    return torch.device('cpu')


def test_generator_output_shape(device):
    # Test standard 128x128 config
    nz = 256
    ngf = 64
    batch_size = 4
    
    G = Generator(ngf=ngf, nz=nz, im_size=128).to(device)
    z = torch.randn(batch_size, nz).to(device)
    
    out = G(z)
    
    assert out.shape == (batch_size, 3, 128, 128), "Generator output shape is incorrect"
    assert out.dtype == torch.float32, "Generator output dtype is incorrect"
    
    # Check if output is roughly in [-1, 1] range (Tanh activation)
    assert out.max() <= 1.0 + 1e-5, "Generator output > 1.0"
    assert out.min() >= -1.0 - 1e-5, "Generator output < -1.0"


def test_discriminator_fake_forward(device):
    batch_size = 4
    ndf = 64
    
    D = Discriminator(ndf=ndf, im_size=128).to(device)
    fake_imgs = torch.randn(batch_size, 3, 128, 128).to(device)
    
    # Forward pass without part (for fake images)
    pred = D(fake_imgs)
    
    assert pred.shape == (batch_size, 1), "Discriminator fake pred shape is incorrect"


def test_discriminator_real_forward_reconstruction(device):
    batch_size = 4
    ndf = 64
    
    D = Discriminator(ndf=ndf, im_size=128).to(device)
    real_imgs = torch.randn(batch_size, 3, 128, 128).to(device)
    
    # Forward pass with part=0 (for real images)
    pred, (rec_full, rec_small, rec_part) = D(real_imgs, part=0)
    
    assert pred.shape == (batch_size, 1), "Discriminator real pred shape is incorrect"
    assert rec_full.shape == (batch_size, 3, 128, 128), "rec_full shape is incorrect"
    assert rec_small.shape == (batch_size, 3, 32, 32), "rec_small shape is incorrect"
    assert rec_part.shape == (batch_size, 3, 32, 32), "rec_part shape is incorrect"
