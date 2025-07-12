"""Tests for the S3FD face detector."""

from unittest import mock

import cv2
import numpy as np
import torch

from wes_tube.detectors.s3fd import S3FD


def test_init():
    """Test S3FD initialization with mocked dependencies."""
    mock_net = mock.MagicMock()
    mock_net.to.return_value = mock_net
    with (
        mock.patch("torch.load", return_value={}),
        mock.patch(
            "wes_tube.detectors.s3fd.S3FDNet", return_value=mock_net
        ) as mock_net_constr,
    ):
        S3FD(device="cpu")

        # Check that the network was initialized with the correct device
        mock_net_constr.assert_called_once_with(device="cpu")
        mock_net.to.assert_called_once_with("cpu")

        # Check that the model was put in eval mode
        mock_net.eval.assert_called_once()


def test_detect_faces():
    """Test the detect_faces method with mocked dependencies."""
    # Create a mock image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create mock detections tensor
    # Format: [batch, num_classes, top_k, 5] where 5 is [score, x1, y1, x2, y2]
    mock_detections = torch.zeros((1, 2, 10, 5))

    # Add two face detections with scores above threshold
    mock_detections[0, 1, 0, 0] = 0.9  # score for first detection
    mock_detections[0, 1, 0, 1:] = torch.tensor([10, 20, 30, 40])  # box coordinates

    mock_detections[0, 1, 1, 0] = 0.85  # score for second detection
    mock_detections[0, 1, 1, 1:] = torch.tensor([50, 60, 70, 80])  # box coordinates

    # Create a class to mock the network's __call__ method
    class MockOutput:
        def __init__(self):
            self.data = mock_detections

    # Mock dependencies
    with (
        mock.patch("torch.load", return_value={}),
        mock.patch("wes_tube.detectors.s3fd.S3FDNet"),
        mock.patch.object(S3FD, "net", create=True),
        mock.patch("torch.from_numpy", return_value=torch.zeros((1, 3, 100, 100))),
        mock.patch("torch.Tensor", return_value=torch.tensor([1, 1, 1, 1])),
        mock.patch("wes_tube.detectors.s3fd.nms_", return_value=np.array([0, 1])),
    ):
        # Create S3FD instance
        detector = S3FD(device="cpu")

        # Mock the network's __call__ method to return our mock output
        detector.net.return_value = MockOutput()

        # Call detect_faces
        bboxes = detector.detect_faces(image, conf_th=0.8, scales=[1])

        # Check that the output has the expected shape and values
        assert isinstance(bboxes, np.ndarray)
        assert bboxes.shape == (2, 5)  # 2 detections, each with 5 values

        # Check the first detection
        np.testing.assert_allclose(bboxes[0, :4], [10, 20, 30, 40])
        np.testing.assert_allclose(bboxes[0, 4], 0.9, rtol=1e-5)

        # Check the second detection
        np.testing.assert_allclose(bboxes[1, :4], [50, 60, 70, 80])
        np.testing.assert_allclose(bboxes[1, 4], 0.85, rtol=1e-5)


def test_detect_faces_with_multiple_scales():
    """Test the detect_faces method with multiple scales."""
    # Create a mock image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create mock detections tensor
    mock_detections = torch.zeros((1, 2, 10, 5))

    # Add one face detection for each scale
    mock_detections[0, 1, 0, 0] = 0.9  # score for first detection
    mock_detections[0, 1, 0, 1:] = torch.tensor([10, 20, 30, 40])  # box coordinates

    # Create a class to mock the network's __call__ method
    class MockOutput:
        def __init__(self):
            self.data = mock_detections

    # Mock dependencies
    with (
        mock.patch("torch.load", return_value={}),
        mock.patch("wes_tube.detectors.s3fd.S3FDNet"),
        mock.patch.object(S3FD, "net", create=True),
        mock.patch("torch.from_numpy", return_value=torch.zeros((1, 3, 100, 100))),
        mock.patch("torch.Tensor", return_value=torch.tensor([1, 1, 1, 1])),
        mock.patch("cv2.resize", return_value=np.zeros((100, 100, 3))),
        mock.patch("wes_tube.detectors.s3fd.nms_", return_value=np.array([0, 1])),
    ):
        # Create S3FD instance
        detector = S3FD(device="cpu")

        # Mock the network's __call__ method to return our mock output
        detector.net.return_value = MockOutput()

        # Call detect_faces with multiple scales
        bboxes = detector.detect_faces(image, conf_th=0.8, scales=[0.5, 1.0, 2.0])

        # Check that cv2.resize was called for each scale
        assert cv2.resize.call_count == 3

        # Check that the model was called for each scale
        assert detector.net.call_count == 3

        # Check that the output has the expected shape
        assert isinstance(bboxes, np.ndarray)
        assert bboxes.shape[1] == 5  # Each detection has 5 values


def test_detect_faces_no_detections():
    """Test the detect_faces method when no faces are detected."""
    # Create a mock image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create mock detections tensor with all scores below threshold
    mock_detections = torch.zeros((1, 2, 10, 5))

    # Create a class to mock the network's __call__ method
    class MockOutput:
        def __init__(self):
            self.data = mock_detections

    # Mock dependencies
    with (
        mock.patch("torch.load", return_value={}),
        mock.patch("wes_tube.detectors.s3fd.S3FDNet"),
        mock.patch.object(S3FD, "net", create=True),
        mock.patch("torch.from_numpy", return_value=torch.zeros((1, 3, 100, 100))),
        mock.patch("torch.Tensor", return_value=torch.tensor([1, 1, 1, 1])),
        mock.patch(
            "wes_tube.detectors.s3fd.nms_", return_value=np.array([], dtype=np.int64)
        ),
    ):
        # Create S3FD instance
        detector = S3FD(device="cpu")

        # Mock the network's __call__ method to return our mock output
        detector.net.return_value = MockOutput()

        # Call detect_faces
        bboxes = detector.detect_faces(image, conf_th=0.8, scales=[1])

        # Check that the output is an empty array with the correct shape
        assert isinstance(bboxes, np.ndarray)
        assert bboxes.shape == (0, 5)
