"""Tests for S3FD box utilities."""

from unittest import mock

import numpy as np
import torch

from wes_tube.detectors.s3fd.box_utils import Detect, PriorBox, decode, nms, nms_


def test_nms_():
    """Test the nms_ function with simple bounding boxes."""
    # Create some test detection boxes
    dets = np.array(
        [
            [10, 10, 30, 30, 0.9],  # box 1
            [15, 15, 35, 35, 0.8],  # box 2 - overlaps with box 1
            [50, 50, 70, 70, 0.7],  # box 3 - no overlap
        ],
        dtype=np.float32,
    )

    # Test with threshold that allows some overlap
    keep = nms_(dets, 0.5)
    assert list(map(int, keep)) == [0, 1, 2]

    # Test with threshold that removes all overlaps
    keep = nms_(dets, 0.1)
    assert list(map(int, keep)) == [0, 2]


def test_decode():
    """Test the decode function for converting predictions to bounding boxes."""
    # Create sample inputs
    loc = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    priors = torch.tensor([[0.5, 0.5, 0.2, 0.2]])  # center_x, center_y, width, height
    variances = [0.1, 0.2]

    # Run the decode function
    boxes = decode(loc, priors, variances)

    # Check output shape
    assert boxes.shape == (1, 4)

    # Check decoding logic - we don't test exact values but basic transformation
    # The resulting boxes should be in (x1, y1, x2, y2) format
    assert boxes[0, 0] < boxes[0, 2]  # x1 < x2
    assert boxes[0, 1] < boxes[0, 3]  # y1 < y2


def test_nms():
    """Test the nms function (torch version)."""
    # Create sample boxes and scores
    boxes = torch.tensor(
        [[10, 10, 30, 30], [15, 15, 35, 35], [50, 50, 70, 70]], dtype=torch.float
    )
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float)

    # Run NMS
    keep, count = nms(boxes, scores, overlap=0.5, top_k=200)

    # Check results
    assert count == 3
    assert list(map(int, keep)) == [0, 1, 2]


def test_priorbox():
    """Test the PriorBox class that generates anchor boxes."""
    # Create a simple PriorBox with minimal params
    input_size = (300, 300)
    feature_maps = [(38, 38), (19, 19)]
    min_sizes = [30, 60]
    steps = [8, 16]

    prior_box = PriorBox(
        input_size=input_size,
        feature_maps=feature_maps,
        min_sizes=min_sizes,
        steps=steps,
    )

    # Generate prior boxes
    priors = prior_box.forward()

    # Check output is a tensor with correct shape
    assert isinstance(priors, torch.Tensor)
    assert priors.shape[1] == 4  # Each prior has (cx, cy, w, h)

    # Number of priors should match feature map dimensions
    expected_num_priors = sum(fm[0] * fm[1] for fm in feature_maps)
    assert priors.shape[0] == expected_num_priors


def test_detect():
    """Test the Detect class that runs NMS on predictions."""
    # Create instance with default params
    detector = Detect()

    # Create mock inputs
    batch_size = 1
    num_priors = 4
    num_classes = 2

    loc_data = torch.rand(batch_size, num_priors * 4).view(batch_size, num_priors, 4)
    conf_data = torch.rand(batch_size, num_priors * num_classes).view(
        batch_size, num_priors, num_classes
    )
    prior_data = torch.tensor(
        [
            [0.1, 0.1, 0.2, 0.2],
            [0.3, 0.3, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
            [0.7, 0.7, 0.2, 0.2],
        ]
    )

    # Make one class confidence high to ensure detection
    conf_data[0, 0, 1] = 0.99

    # Run detection
    output = detector.forward(loc_data, conf_data, prior_data)

    # Check output shape - [batch_size, num_classes, top_k, 5]
    # where 5 is [score, x1, y1, x2, y2]
    assert output.shape == (batch_size, num_classes, detector.top_k, 5)


def test_decode_for_location_predictions():
    """Test the decode function for location predictions."""
    # Create test data
    loc = torch.tensor([[0.1, 0.2, 0.1, 0.2]], dtype=torch.float32)
    priors = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
    variances = [0.1, 0.2]

    # Decode locations
    boxes = decode(loc, priors, variances)

    # Expected: priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:] for cx, cy
    # and priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]) for w, h
    # Then convert from center-size to corner format
    expected_cx = 0.5 + 0.1 * 0.1 * 0.2
    expected_cy = 0.5 + 0.2 * 0.1 * 0.2
    expected_w = 0.2 * torch.exp(torch.tensor([0.1 * 0.2]))
    expected_h = 0.2 * torch.exp(torch.tensor([0.2 * 0.2]))

    expected_x1 = expected_cx - expected_w / 2
    expected_y1 = expected_cy - expected_h / 2
    expected_x2 = expected_cx + expected_w / 2
    expected_y2 = expected_cy + expected_h / 2

    assert torch.allclose(boxes[0][0], expected_x1, atol=1e-5)
    assert torch.allclose(boxes[0][1], expected_y1, atol=1e-5)
    assert torch.allclose(boxes[0][2], expected_x2, atol=1e-5)
    assert torch.allclose(boxes[0][3], expected_y2, atol=1e-5)


def test_nms_torch():
    """Test the torch implementation of nms."""
    # Create some test boxes and scores
    boxes = torch.tensor(
        [
            [10, 10, 30, 30],  # box 1
            [15, 15, 35, 35],  # box 2 - overlaps with box 1
            [50, 50, 70, 70],  # box 3 - no overlap
        ],
        dtype=torch.float32,
    )

    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)

    # Test with threshold that allows some overlap
    keep, count = nms(boxes, scores, overlap=0.5, top_k=200)
    assert count == 3
    assert list(map(int, keep)) == [0, 1, 2]


def test_detect_forward():
    """Test the Detect class forward method."""
    batch_size = 1
    num_priors = 2
    num_classes = 2

    # Create test data
    loc_data = torch.rand(batch_size, num_priors * 4).view(batch_size, num_priors, 4)
    conf_data = torch.rand(batch_size, num_priors * num_classes).view(
        batch_size, num_priors, num_classes
    )
    prior_data = torch.tensor(
        [
            [0.5, 0.5, 0.2, 0.2],  # prior 1
            [0.7, 0.7, 0.2, 0.2],  # prior 2
        ]
    )

    # Create detector with test parameters
    detector = Detect(num_classes=num_classes, conf_thresh=0.01, top_k=10)

    # Mock nms function to avoid actual computation
    with mock.patch(
        "wes_tube.detectors.s3fd.box_utils.nms",
        return_value=(torch.zeros(1, dtype=torch.long), 0),
    ):
        output = detector.forward(loc_data, conf_data, prior_data)

        # Check output shape
        assert output.shape == (batch_size, num_classes, detector.top_k, 5)


def test_prior_box():
    """Test the PriorBox class."""
    # Use smaller feature maps for testing
    input_size = (32, 32)
    feature_maps = [(4, 4), (2, 2)]
    min_sizes = [16, 32]
    steps = [8, 16]

    # Create PriorBox and generate priors
    prior_box = PriorBox(
        input_size=input_size,
        feature_maps=feature_maps,
        min_sizes=min_sizes,
        steps=steps,
    )
    priors = prior_box.forward()

    # Check output shape: should be (num_priors, 4)
    # num_priors = sum(fm[0] * fm[1] for fm in feature_maps)
    expected_num_priors = 4 * 4 + 2 * 2
    assert priors.shape == (expected_num_priors, 4)

    # Test with clipping
    prior_box_clip = PriorBox(
        input_size=input_size,
        feature_maps=feature_maps,
        min_sizes=min_sizes,
        steps=steps,
        clip=True,
    )
    priors_clip = prior_box_clip.forward()

    # All values should be between 0 and 1
    assert torch.all(priors_clip >= 0)
    assert torch.all(priors_clip <= 1)
