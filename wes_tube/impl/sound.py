"""Sound processing functionality for WesTube."""

import logging
from collections import defaultdict
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import scenedetect

from wes_tube import assets

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from types import ModuleType
    from typing import cast

    from scenedetect.video_splitter import TimecodePair

    # mypy / the type checker does not
    # understand that assets is an instance
    # of ModuleType.
    assets = cast(ModuleType, assets)

BoundingBox = tuple[int, int, int, int]


@dataclass
class DetectedFace:
    """A detected face in a video frame."""

    frame_num: int
    b_box: BoundingBox
    confidence: float


# The face in each entry in the trajectory is assumed to be the same face
Trajectory = list[DetectedFace]


def detect_offset(video_file_path: Path) -> int:
    """Detect the audio offset in a video file.

    This function uses PyTorch to analyze the video and audio streams
    to determine if there is an offset between them.

    Args:
        video_file_path: Path to the video file to analyze.

    Returns:
        Exit code (0 for success, non-zero for failure).

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        RuntimeError: If the video file can't be processed.
    """
    # TODO: Implement sound offset detection using PyTorch

    # # Calculate the face trajectories in the first pass through the video file
    #
    # Trajectory = list[FaceDetect]
    # shots = scene_detect(input_video_file)
    # face_trajectories: dict[ShotBoundaries, list[Trajectory]] = defaultdict(list)
    # for each frame in input_video:
    #   frame_shots = [shot for shot in shots if is_in_shot(frame, shot)]
    #   face_detects = detect_faces(frame)
    #   for each shot in frame_shots:
    #      open_trajectories = [t for t in face_trajectories[shot] if
    #          last frame in t is at most num_failed_det frames ago]
    #      assign each bounding box to the first open trajectory without an assignment
    #            yet this frame whose last assigned frame overlaps (intersection/union) by
    #            more than iou_threshold
    #   ending_shots = [shot for shot in shots if is_last_frame(frame, shot)]
    #   # Eliminate too short trajectories and interpolate the rest
    #   # Eliminate any trajectories that are too small
    #   for each shot in ending_shots:
    #       f = [interpolate_missing_bounding_boxes(t) for t in face_trajectories[shot] if
    #           length(t) > too_small_trajectory_len (a.k.a. opt.min_track)]
    #       face_trajectories[shot] = [t for t in f if mean bounding box width
    #            or mean bounding box height is larger than min_face_size]
    #
    # # Second pass through the video file
    # # Run the SyncNet model on frames cropped to each face trajectory
    # # Maintain a window of batch_size frames and the associated audio.
    # # For each trajectory, crop each frame in the window to the trajectory.
    # # then run the SyncNet model on those frames and record the offset calculated
    # # from those frames.
    # # Offset is calculated by creating an audio feature vector and a video
    # # feature vector and calculating the frame offset (-v_shift..+v_shift) at
    # # which the distance between those vectors is minimized. I do not yet
    # # understand how the batch size relates to the feature vector size. I'm
    # # also confused why when I ran the original run_syncnet.py, it created
    # # 3 face-cropped video files, but only printed one offset. My reading
    # # of the code makes me expect it to print one offset per face-cropped file.
    #
    # # To make the coding more like the research code, I might implement this
    # # as N passes: one per face in the shot with the maximum number of faces.
    # # This will yield a more direct port in which I don't need to be concerned
    # # about the details of the batches because I can copy that part of the
    # # code.
    # TODO: Write pseudocode for the second and later passes
    with resources.open_binary(assets, "syncnet_v2.model") as f:
        syncnet_model = f.read()
        logger.info(f"SyncNet Model is {len(syncnet_model)} bytes long")
    logger.info(f"Detecting sound offset in {video_file_path}")
    trajectories = face_trajectories(video_file_path)  # noqa: F841
    # TODO: add the second phase (passes 2..N) of the offset detection and remove F841
    return 0


def re_sampled_video_frames_iterator(
    video_path: Path, target_fps: float = 25.0
) -> Generator[np.ndarray]:
    """
    Return an iterator through video frames that resamples to target FPS.

    Resampling is done through skipping or duplicating frames

    Original code by Claude 4 Sonnet but modified to fix precision and edge
    case issues and to improve readability.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second for resampling. Must be greater than 0.

    Yields:
        numpy.ndarray: Frame in OpenCV format (BGR, uint8)
    """
    if target_fps <= 0:
        raise ValueError("Target FPS must be greater than 0")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Failed to open video file {video_path}")
        return
    logger.info(f"Resampling video {video_path} to {target_fps} FPS")

    try:
        # How many original frames are in each output frame
        original_fps = Fraction(cap.get(cv2.CAP_PROP_FPS)).limit_denominator()
        original_per_output = original_fps / Fraction(target_fps).limit_denominator()

        # The original frame number (fractional) corresponding to the next frame
        # to output
        next_output_frame = Fraction(0)
        # The number in the original video of the image held in ``frame``
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Yield the next frame
            while current_frame >= int(next_output_frame):
                yield frame
                next_output_frame += original_per_output

            current_frame += 1

    finally:
        cap.release()


# Pair of the first and last frame numbers in a shot
# Converted from a TimecodePair
ShotBoundaries = tuple[int, int]


def face_trajectories(video_file_path: Path) -> Sequence[Trajectory]:
    """Return the face trajectories in a video file."""
    timecode_shots: Sequence[TimecodePair] = scenedetect.detect(
        str(video_file_path), scenedetect.ContentDetector()
    )
    shots: list[ShotBoundaries] = [  # noqa: F841
        (s[0].frame_num, s[1].frame_num) for s in timecode_shots
    ]
    face_trj: dict[ShotBoundaries, list[Trajectory]] = defaultdict(list)  # noqa: F841
    # TODO: finish filling in face_trj, use shots and remove the noqa: F841


def correct_offset(input_file: Path, output_file: Path) -> int:
    """Correct the audio offset in a video file.

    This function uses PyAV to adjust the audio stream to match the video stream.
    It depends on offset data stored in the system state.

    It is mainly present for special uses. Most times, users will correct
    the offset at the same time as they apply other changes.

    Args:
        input_file: Path to the video file that needs correction.
        output_file: Path to write the corrected video.

    Returns:
        Exit code (0 for success, non-zero for failure).

    Raises:
        FileNotFoundError: If either video file doesn't exist.
        RuntimeError: If the video files can't be processed.
    """
    # TODO: Implement sound offset correction using PyAV
    print(  # noqa: T201
        f"Correcting sound offset in {input_file} writing to {output_file}"
    )
    return 0
