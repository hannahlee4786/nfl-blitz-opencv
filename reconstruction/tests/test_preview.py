from fractions import Fraction

import av
import numpy as np
import pytest

from nfl_reconstruction.field import NFLField
from nfl_reconstruction.preview import render_preview_frame, write_preview_video
from nfl_reconstruction.schemas import CameraFrame
from tests.fixtures import synthetic_state


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def preview_camera(**updates):
    camera = CameraFrame(
        frame=26,
        state=synthetic_state(),
        confidence=1.0,
        status="reliable",
        homography=IDENTITY,
        line_error_px=0.75,
        source="snapped",
    )
    return camera.model_copy(update=updates)


def test_preview_frame_is_fixed_size_and_preserves_both_panes():
    frame = np.full((101, 157, 3), (30, 100, 40), dtype=np.uint8)
    frame[:, 50:54] = (245, 245, 245)
    mask = np.zeros(frame.shape[:2], dtype=bool)

    rendered = render_preview_frame(
        frame,
        preview_camera(),
        mask,
        NFLField.standard(),
        size=(1280, 720),
        pixels_per_yard=1,
    )

    assert rendered.shape == (720, 1280, 3)
    assert rendered.dtype == np.uint8
    assert rendered[:, :640].mean() > 0
    assert rendered[:, 640:].mean() > 0
    assert rendered[700, 100].max() < 20


def test_preview_frame_uses_warning_treatment_for_missing_camera():
    frame = np.full((101, 157, 3), 50, dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=bool)
    missing = preview_camera(
        confidence=0.0, status="missing", homography=None, source=None
    )

    rendered = render_preview_frame(
        frame, missing, mask, NFLField.standard(), size=(1280, 720)
    )

    warning_pixels = (rendered[..., 0] > 200) & (rendered[..., 1] > 120)
    assert warning_pixels.any()
    assert rendered[100:, 650:].mean() < rendered[100:, :630].mean()


def test_preview_video_preserves_count_dimensions_and_fractional_rate(tmp_path):
    frames = [
        np.full((101, 157, 3), value, dtype=np.uint8)
        for value in (30, 90, 150)
    ]
    cameras = [preview_camera(frame=26 + index) for index in range(3)]
    masks = [np.zeros(frame.shape[:2], dtype=bool) for frame in frames]
    output = tmp_path / "preview.mp4"

    write_preview_video(
        output,
        frames,
        cameras,
        masks,
        NFLField.standard(),
        fps=Fraction(30000, 1001),
        size=(640, 360),
        pixels_per_yard=1,
    )

    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(video=0))
        assert Fraction(stream.average_rate) == Fraction(30000, 1001)
    assert len(decoded) == 3
    assert (decoded[0].width, decoded[0].height) == (640, 360)


def test_preview_video_rejects_length_mismatch_before_writing(tmp_path):
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    output = tmp_path / "partial.mp4"

    with pytest.raises(ValueError, match="same length"):
        write_preview_video(
            output,
            [frame, frame],
            [preview_camera()],
            [np.zeros(frame.shape[:2], dtype=bool)] * 2,
            NFLField.standard(),
            fps=Fraction(30, 1),
        )

    assert not output.exists()
