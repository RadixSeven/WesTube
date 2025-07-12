"""Fixtures for detector tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple RGB test image
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_bounding_boxes():
    """Create mock bounding boxes for testing."""
    return np.array(
        [
            [10, 10, 30, 30, 0.9],  # box 1
            [15, 15, 35, 35, 0.8],  # box 2 - overlaps with box 1
            [50, 50, 70, 70, 0.7],  # box 3 - no overlap
        ]
    )
