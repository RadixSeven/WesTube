"""Sound processing functionality for WesTube."""

from pathlib import Path


def detect_offset(filename: Path) -> int:
    """Detect the audio offset in a video file.

    This function uses PyTorch to analyze the video and audio streams
    to determine if there is an offset between them.

    Args:
        filename: Path to the video file to analyze.

    Returns:
        Exit code (0 for success, non-zero for failure).

    Raises:
        FileNotFoundError: If the video file doesn't exist.
        RuntimeError: If the video file can't be processed.
    """
    # TODO: Implement sound offset detection using PyTorch
    print(f"Detecting sound offset in {filename}")  # noqa: T201
    return 0


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
        f"Correcting sound offset in {output_file} using {input_file} as reference"
    )
    return 0
