"""Main entry point for WesTube."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class SoundOffsetArgs:
    """Arguments for the sound-offset subcommand."""

    filename: Path


@dataclass
class CorrectSoundArgs:
    """Arguments for the correct-sound subcommand."""

    input_file: Path
    output_file: Path


@dataclass
class CliArgs:
    """Command line arguments."""

    command: str
    sound_offset_args: SoundOffsetArgs | None = None
    correct_sound_args: CorrectSoundArgs | None = None


def parse_args(args: Sequence[str] | None = None) -> CliArgs:
    """Parse command line arguments.

    Args:
        args: Command line arguments to parse. If None, uses sys.argv[1:].

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="westube", description="WesTube - YouTube Channel Tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # GUI subcommand
    subparsers.add_parser("gui", help="Start the GUI")

    # Sound offset detection subcommand
    sound_offset_parser = subparsers.add_parser(
        "sound-offset", help="Detect sound offset in a video file"
    )
    sound_offset_parser.add_argument(
        "filename", help="Video file to check for sound offset", type=Path
    )

    # Sound correction subcommand
    correct_sound_parser = subparsers.add_parser(
        "correct-sound", help="Correct sound offset in a video file"
    )
    correct_sound_parser.add_argument(
        "input_file", help="Video file to correct", type=Path
    )
    correct_sound_parser.add_argument(
        "output_file", help="Corrected video file", type=Path
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "sound-offset":
        return CliArgs(
            command="sound-offset",
            sound_offset_args=SoundOffsetArgs(filename=parsed_args.filename),
        )
    # correct-sound
    return CliArgs(
        command="correct-sound",
        correct_sound_args=CorrectSoundArgs(
            input_file=parsed_args.input_file,
            output_file=parsed_args.output_file,
        ),
    )


def main(args: Sequence[str] | None = None) -> int:
    """Start the application.

    Args:
        args: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code.
    """
    cli_args = parse_args(args)
    logging.basicConfig(
        format="%(asctime)s || %(levelname)s || %(name)s || %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=logging.INFO,
    )

    if cli_args.command == "sound-offset":
        from wes_tube.impl.sound import detect_offset

        return detect_offset(cli_args.sound_offset_args.filename)
    # correct-sound
    from wes_tube.impl.sound import correct_offset

    return correct_offset(
        cli_args.correct_sound_args.input_file,
        cli_args.correct_sound_args.output_file,
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
