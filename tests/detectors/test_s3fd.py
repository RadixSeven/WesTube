"""Tests for the S3FD face detector."""

from unittest import mock

import pytest

from wes_tube.detectors.s3fd import S3FD


@pytest.fixture
def mock_s3fd():
    """Create a mocked S3FD object without loading the actual model."""
    with (
        mock.patch("torch.load", return_value={}),
        mock.patch("wes_tube.detectors.s3fd.S3FDNet") as mock_net,
        mock.patch("wes_tube.detectors.s3fd.PATH_WEIGHT", "mock_path.pth"),
    ):
        mock_net.return_value = mock.MagicMock()
        return S3FD(device="cpu")


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
