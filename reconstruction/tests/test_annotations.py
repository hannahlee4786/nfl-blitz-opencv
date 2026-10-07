import json
from pathlib import Path

import numpy as np
import pytest

from nfl_reconstruction.annotations import (
    anchor_homography,
    load_anchor,
    validate_anchor,
)
from nfl_reconstruction.field import NFLField
from nfl_reconstruction.schemas import AnchorSet, FieldPoint, ImagePoint


def make_anchor(field=None, image=None):
    field = field or [(10, 10), (20, 10), (20, 20), (10, 20)]
    image = image or [(100, 100), (200, 100), (200, 200), (100, 200)]
    return AnchorSet(
        frame=0,
        image_width=864,
        image_height=614,
        field_points=tuple(FieldPoint(x=x, y=y) for x, y in field),
        image_points=tuple(ImagePoint(x=x, y=y) for x, y in image),
    )


def test_anchor_rejects_collinear_field_points():
    anchor = make_anchor(
        field=[(10, 20), (20, 20), (30, 20), (40, 20)],
        image=[(100, 100), (200, 100), (300, 100), (400, 100)],
    )
    with pytest.raises(ValueError, match="non-collinear"):
        validate_anchor(anchor, NFLField.standard(), (864, 614))


def test_anchor_rejects_points_outside_image():
    anchor = make_anchor(
        image=[(100, 100), (200, 100), (300, 200), (900, 300)]
    )
    with pytest.raises(ValueError, match="image bounds"):
        validate_anchor(anchor, NFLField.standard(), (864, 614))


def test_anchor_requires_matching_unique_pairs():
    duplicate = make_anchor(
        field=[(10, 10), (20, 10), (20, 20), (20, 20)]
    )
    with pytest.raises(ValueError, match="unique"):
        validate_anchor(duplicate, NFLField.standard(), (864, 614))


def test_load_anchor_accepts_compact_coordinate_arrays(tmp_path):
    path = tmp_path / "anchor.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame": 0,
                "image_width": 864,
                "image_height": 614,
                "field_points": [[10, 10], [20, 10], [20, 20], [10, 20]],
                "image_points": [[100, 100], [200, 100], [200, 200], [100, 200]],
            }
        )
    )

    anchor = load_anchor(path, (864, 614))

    assert anchor.field_points[0] == FieldPoint(x=10, y=10)


def test_seed_anchor_reprojects_with_less_than_five_pixel_mean_error():
    anchor = load_anchor(
        Path(__file__).parents[1] / "annotations" / "clip3-anchor.json",
        (864, 614),
    )
    homography = anchor_homography(anchor)
    field = np.array([[point.x, point.y] for point in anchor.field_points])
    image = np.array([[point.x, point.y] for point in anchor.image_points])
    homogeneous = np.column_stack([field, np.ones(len(field))])
    projected = (homography @ homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:]

    assert np.linalg.norm(projected - image, axis=1).mean() < 5.0
