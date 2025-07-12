"""Tests for the CLI interface."""

import pytest

from wes_tube.cli.main import parse_args


def test_parse_args_sound_offset():
    """Test parsing the sound-offset subcommand."""
    args = parse_args(["sound-offset", "video.mp4"])
    assert args.command == "sound-offset"
    assert args.sound_offset_args is not None
    assert str(args.sound_offset_args.filename) == "video.mp4"
    assert args.correct_sound_args is None


def test_parse_args_correct_sound():
    """Test parsing the correct-sound subcommand."""
    args = parse_args(["correct-sound", "source.mp4", "target.mp4"])
    assert args.command == "correct-sound"
    assert args.sound_offset_args is None
    assert args.correct_sound_args is not None
    assert str(args.correct_sound_args.input_file) == "source.mp4"
    assert str(args.correct_sound_args.output_file) == "target.mp4"


def test_parse_args_no_args(capsys: pytest.CaptureFixture[str]):
    """Test that parsing with no arguments raises an error."""
    with pytest.raises(SystemExit):
        parse_args([])
    assert "command" in capsys.readouterr().err


def test_parse_args_invalid_command(capsys: pytest.CaptureFixture[str]):
    """Test that parsing an invalid command raises an error."""
    with pytest.raises(SystemExit):
        parse_args(["invalid-command"])
    assert "invalid-command" in capsys.readouterr().err
