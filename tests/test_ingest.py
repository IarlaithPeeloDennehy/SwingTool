from pathlib import Path

import cv2
import numpy as np
import pytest

from swingtool.ingest import (
    VideoIngestError,
    VideoReader,
    compute_scale,
    downscale,
    select_indices,
)


def make_video(path: Path, frames: int, width: int = 64, height: int = 48,
               fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    assert writer.isOpened()
    for i in range(frames):
        image = np.full((height, width, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(image)
    writer.release()


class TestComputeScale:
    def test_no_max_dim(self):
        assert compute_scale(1920, 1080, None) == 1.0

    def test_already_small(self):
        assert compute_scale(640, 480, 1080) == 1.0

    def test_landscape(self):
        assert compute_scale(1920, 1080, 960) == pytest.approx(0.5)

    def test_portrait_uses_long_side(self):
        # Portrait phone clip: height is the long side.
        assert compute_scale(1080, 1920, 960) == pytest.approx(0.5)

    def test_coords_map_back(self):
        scale = compute_scale(1080, 1920, 640)
        x_small = 100.0
        assert x_small / scale == pytest.approx(x_small * 1920 / 640)


class TestDownscale:
    def test_identity(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        assert downscale(image, 1.0) is image

    def test_portrait_shape(self):
        image = np.zeros((192, 108, 3), dtype=np.uint8)
        out = downscale(image, 0.5)
        assert out.shape == (96, 54, 3)


class TestSelectIndices:
    def test_stride_one(self):
        assert select_indices(5, 1) == [0, 1, 2, 3, 4]

    def test_stride_three(self):
        assert select_indices(10, 3) == [0, 3, 6, 9]

    def test_invalid_stride(self):
        with pytest.raises(ValueError):
            select_indices(10, 0)


class TestVideoReader:
    def test_missing_file(self, tmp_path):
        with pytest.raises(VideoIngestError, match="not found"):
            with VideoReader(tmp_path / "nope.mp4"):
                pass

    def test_unreadable_file(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"this is not a video")
        with pytest.raises(VideoIngestError):
            with VideoReader(bad):
                pass

    def test_reads_all_frames(self, tmp_path):
        video = tmp_path / "clip.mp4"
        make_video(video, frames=8)
        with VideoReader(video) as reader:
            assert reader.info.width == 64
            assert reader.info.height == 48
            frames = list(reader.frames())
        assert [f.index for f in frames] == list(range(8))
        assert frames[0].timestamp_s == 0.0
        assert frames[5].timestamp_s == pytest.approx(0.5)  # 10 fps

    def test_stride_skips_frames(self, tmp_path):
        video = tmp_path / "clip.mp4"
        make_video(video, frames=8)
        with VideoReader(video, frame_stride=3) as reader:
            indices = [f.index for f in reader.frames()]
        assert indices == [0, 3, 6]

    def test_streaming_one_at_a_time(self, tmp_path):
        video = tmp_path / "clip.mp4"
        make_video(video, frames=4)
        with VideoReader(video) as reader:
            gen = reader.frames()
            first = next(gen)
            assert first.index == 0  # lazily consumable, not preloaded

    def test_invalid_stride(self, tmp_path):
        with pytest.raises(ValueError):
            VideoReader(tmp_path / "x.mp4", frame_stride=0)
