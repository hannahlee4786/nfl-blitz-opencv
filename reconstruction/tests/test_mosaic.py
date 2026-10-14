import numpy as np
import pytest

from nfl_reconstruction.field import NFLField
from nfl_reconstruction.mosaic import build_mosaic, warp_frame_to_field
from nfl_reconstruction.schemas import CameraFrame
from tests.fixtures import synthetic_state


def aligned_field_frames():
    height, width = 53, 120
    expected = np.zeros((height, width, 3), dtype=np.uint8)
    expected[..., 1] = 100
    expected[:, ::10] = 240
    frames = [expected.copy(), expected.copy()]
    frames[1][20:25, 50:55] = (255, 0, 0)
    masks = [
        np.zeros((height, width), dtype=bool),
        np.zeros((height, width), dtype=bool),
    ]
    masks[1][20:25, 50:55] = True
    homography = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    cameras = [
        CameraFrame(
            frame=index,
            state=synthetic_state(),
            confidence=1.0,
            homography=homography,
        )
        for index in range(2)
    ]
    return frames, cameras, masks, expected


def test_mosaic_rejects_transient_player_pixel():
    frames, cameras, masks, expected = aligned_field_frames()

    result = build_mosaic(
        frames,
        cameras,
        masks,
        NFLField.standard(),
        pixels_per_yard=1,
    )

    assert np.mean(np.abs(result.image.astype(float) - expected.astype(float))) < 2.0
    assert result.sample_count[22, 52] == 1


def test_mosaic_rejects_mask_shape_mismatch():
    frames, cameras, _, _ = aligned_field_frames()

    with pytest.raises(ValueError, match="mask shape"):
        build_mosaic(
            frames,
            cameras,
            [np.zeros((4, 4), bool), np.zeros(frames[1].shape[:2], bool)],
            NFLField.standard(),
        )


def test_mosaic_skips_uncertain_camera():
    frames, cameras, masks, _ = aligned_field_frames()
    cameras[1] = cameras[1].model_copy(
        update={"confidence": 0.1, "status": "uncertain"}
    )

    result = build_mosaic(frames, cameras, masks, NFLField.standard())

    assert result.sample_count.max() == 1


def test_single_frame_warp_maps_identity_field_and_mask():
    frames, cameras, masks, expected = aligned_field_frames()
    masks[0][10, 20] = True

    warped = warp_frame_to_field(
        frames[0], cameras[0], masks[0], NFLField.standard(), pixels_per_yard=1
    )

    assert warped.image.shape == expected.shape
    assert np.array_equal(warped.image[11, 21], expected[11, 21])
    assert not warped.coverage[10, 20]
    assert np.array_equal(warped.image[10, 20], np.zeros(3, dtype=np.uint8))


def test_single_frame_warp_returns_blank_when_homography_is_missing():
    frames, cameras, masks, _ = aligned_field_frames()
    missing = cameras[0].model_copy(
        update={"homography": None, "confidence": 0.0, "status": "missing"}
    )

    warped = warp_frame_to_field(
        frames[0], missing, masks[0], NFLField.standard(), pixels_per_yard=1
    )

    assert warped.image.shape == (53, 120, 3)
    assert not warped.coverage.any()
    assert not warped.image.any()


def test_single_frame_warp_rejects_mask_shape_mismatch():
    frames, cameras, _, _ = aligned_field_frames()

    with pytest.raises(ValueError, match="mask shape"):
        warp_frame_to_field(
            frames[0], cameras[0], np.zeros((4, 4), bool), NFLField.standard()
        )
