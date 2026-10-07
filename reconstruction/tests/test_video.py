from fractions import Fraction

import av
import numpy as np
import pytest

import nfl_reconstruction.video as video_module
from nfl_reconstruction.video import VideoDecodeError, build_manifest, read_frame


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "tiny.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=Fraction(5, 1))
        stream.width = 16
        stream.height = 12
        stream.pix_fmt = "yuv420p"
        for value in (20, 80, 140):
            image = np.full((12, 16, 3), value, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_manifest_hashes_and_numbers_every_decoded_frame(tiny_video, tmp_path):
    manifest = build_manifest(tiny_video, tmp_path / "frames")

    assert [frame.index for frame in manifest.frames] == [0, 1, 2]
    assert all(len(frame.sha256) == 64 for frame in manifest.frames)
    assert (tmp_path / "frames" / "000000.png").exists()
    assert manifest.video.fps_numerator == 5
    assert manifest.video.fps_denominator == 1


def test_dimension_change_stops_with_frame_number(tiny_video, tmp_path, monkeypatch):
    frames = [
        np.zeros((12, 16, 3), dtype=np.uint8),
        np.zeros((12, 16, 3), dtype=np.uint8),
        np.zeros((10, 16, 3), dtype=np.uint8),
    ]
    monkeypatch.setattr(video_module, "_decode_frames", lambda _: iter(frames))

    with pytest.raises(ValueError, match="frame 2.*dimensions"):
        build_manifest(tiny_video, tmp_path / "frames")


def test_read_frame_returns_rgb_array(tiny_video):
    frame = read_frame(tiny_video, 1)

    assert frame.shape == (12, 16, 3)
    assert frame.dtype == np.uint8
    assert abs(float(frame.mean()) - 80.0) < 5.0


def test_read_frame_reports_missing_index(tiny_video):
    with pytest.raises(VideoDecodeError, match="frame 8"):
        read_frame(tiny_video, 8)
