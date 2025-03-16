"""Main entry point for WesTube."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class GuiArgs:
    """Arguments for the GUI subcommand."""


@dataclass
class SoundOffsetArgs:
    """Arguments for the sound-offset subcommand."""

    filename: str


@dataclass
class CorrectSoundArgs:
    """Arguments for the correct-sound subcommand."""

    source_file: str
    target_file: str


@dataclass
class CliArgs:
    """Command line arguments."""

    command: str
    gui_args: GuiArgs | None = None
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
        description="WesTube - YouTube Channel Tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # GUI subcommand
    subparsers.add_parser("gui", help="Start the GUI")

    # Sound offset detection subcommand
    sound_offset_parser = subparsers.add_parser(
        "sound-offset", help="Detect sound offset in a video file"
    )
    sound_offset_parser.add_argument(
        "filename", help="Video file to check for sound offset"
    )

    # Sound correction subcommand
    correct_sound_parser = subparsers.add_parser(
        "correct-sound", help="Correct sound offset in a video file"
    )
    correct_sound_parser.add_argument(
        "source_file", help="Source video file with correct timing"
    )
    correct_sound_parser.add_argument(
        "target_file", help="Video file to correct"
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "gui":
        return CliArgs(command="gui", gui_args=GuiArgs())
    if parsed_args.command == "sound-offset":
        return CliArgs(
            command="sound-offset",
            sound_offset_args=SoundOffsetArgs(filename=parsed_args.filename),
        )
    # correct-sound
    return CliArgs(
        command="correct-sound",
        correct_sound_args=CorrectSoundArgs(
            source_file=parsed_args.source_file,
            target_file=parsed_args.target_file,
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

    if cli_args.command == "gui":
        from wes_tube.gui import main as gui_main

        return gui_main()
    if cli_args.command == "sound-offset":
        from wes_tube.impl.sound import detect_offset

        return detect_offset(cli_args.sound_offset_args.filename)
    # correct-sound
    from wes_tube.impl.sound import correct_offset

    return correct_offset(
        cli_args.correct_sound_args.source_file,
        cli_args.correct_sound_args.target_file,
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
