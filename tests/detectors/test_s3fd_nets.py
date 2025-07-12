"""Tests for S3FD neural network classes."""

import pytest
import torch

from wes_tube.detectors.s3fd.nets import L2Norm, S3FDNet


def test_l2norm():
    """Test the L2Norm module."""
    # Create test data
    batch_size = 2
    channels = 4
    height = 8
    width = 8
    scale = 10.0

    # Create a test input tensor
    x = torch.rand(batch_size, channels, height, width)

    # Create L2Norm module
    l2norm = L2Norm(n_channels=channels, scale=scale)

    # Check that weights were initialized correctly
    assert torch.allclose(l2norm.weight, torch.tensor(scale))

    # Apply L2Norm
    output = l2norm(x)

    # Check output shape
    assert output.shape == (batch_size, channels, height, width)

    # Check that normalization was applied correctly
    # Calculate the L2 norm manually for comparison
    norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + l2norm.eps
    expected = torch.div(x, norm) * scale
    assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", ["cpu"])
def test_s3fdnet_init(device):
    """Test S3FDNet initialization."""
    # Create S3FDNet
    model = S3FDNet(device=device)

    # Check model structure
    assert len(model.vgg) == 35
    assert len(model.extras) == 4
    assert len(model.loc) == 6
    assert len(model.conf) == 6

    # Check that all components are on the correct device
    assert next(model.parameters()).device.type == device


def test_s3fd_net_forward():
    """Test S3FDNet forward pass with a small input."""

    # Create a minimal test input (using small size to speed up test)
    batch_size = 1
    channels = 3
    height = 32
    width = 32
    x = torch.rand(batch_size, channels, height, width, device="cpu")

    # Create model
    model = S3FDNet(device="cpu")

    # Set to eval mode to avoid batch norm issues
    model.eval()

    # Run forward pass
    with torch.no_grad():
        output = model(x)

    # Check the output format
    assert isinstance(output, torch.Tensor)

    # Output should be a batch of detections
    # Shape should be [batch_size, num_classes, top_k, 5]
    # where 5 means [score, x1, y1, x2, y2]
    assert tuple(map(int, output.shape)) == (batch_size, 2, 750, 5)
