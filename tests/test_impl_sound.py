from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from wes_tube.impl.sound import re_sampled_video_frames_iterator


def test_re_sampled_video_frames_iterator_bad_fps():
    p = Path("ignored")
    with pytest.raises(ValueError, match="greater.*0"):
        list(re_sampled_video_frames_iterator(p, 0))


def test_re_sampled_video_frames_iterator_file_not_found():
    p = Path("nonexistent.mp4")
    frames = list(re_sampled_video_frames_iterator(p))
    assert len(frames) == 0


# noinspection PyUnresolvedReferences
def test_re_sampled_video_frames_iterator_halving(tmp_path):
    # Create a test video file - lossless
    # Write 30 frames (1 second at 30 fps)
    video_path = tmp_path / "test.avi"
    orig_frames = [idx * np.ones((2, 2, 3), dtype=np.uint8) for idx in range(30)]

    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (2, 2)
    )
    try:
        for frame in orig_frames:
            writer.write(frame)
    finally:
        writer.release()

    # Test resampling to 15 fps
    # noinspection PyUnreachableCode
    frames = list(re_sampled_video_frames_iterator(video_path, 15.0))
    assert len(frames) == 15  # Should get half the frames
    assert (frames[0] == orig_frames[0]).all()
    assert (frames[14] == orig_frames[28]).all()


# noinspection PyUnresolvedReferences
def test_re_sampled_video_frames_iterator_doubling(tmp_path):
    # Create a test video file - lossless
    # Write 30 frames (1 second at 30 fps)
    orig_frames = [idx * np.ones((2, 2, 3), dtype=np.uint8) for idx in range(30)]
    video_path = tmp_path / "test.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (2, 2)
    )
    try:
        for frame in orig_frames:
            writer.write(frame)
    finally:
        writer.release()

    # Test resampling to 60 fps
    # noinspection PyUnreachableCode
    frames = list(re_sampled_video_frames_iterator(video_path, 60.0))
    assert len(frames) == 60  # Should get every frame doubled
    assert (frames[0] == orig_frames[0]).all()
    assert (frames[1] == orig_frames[0]).all()
    assert (frames[58] == orig_frames[29]).all()
    assert (frames[59] == orig_frames[29]).all()


def test_re_sampled_video_frames_iterator_unopened(caplog):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    test_path = Path("test.mp4")

    with patch("cv2.VideoCapture", return_value=mock_cap):
        frames = list(re_sampled_video_frames_iterator(test_path))
        assert len(frames) == 0
        assert str(test_path) in caplog.text
        assert "open" in caplog.text.lower()
        assert "failed" in caplog.text.lower()
