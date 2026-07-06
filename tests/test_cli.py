import pytest

from swingtool.cli import build_parser


def test_analyze_defaults():
    args = build_parser().parse_args(["analyze", "clip.mp4"])
    assert args.command == "analyze"
    assert args.device == "cuda"
    assert args.frame_stride == 1
    assert args.max_dim is None


def test_analyze_flags():
    args = build_parser().parse_args(
        ["analyze", "clip.mp4", "--device", "cpu", "--frame-stride", "3",
         "--max-dim", "960"]
    )
    assert args.device == "cpu"
    assert args.frame_stride == 3
    assert args.max_dim == 960


def test_rejects_unknown_device():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyze", "clip.mp4", "--device", "tpu"])
