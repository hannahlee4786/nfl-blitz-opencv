import numpy as np
import pytest

from nfl_reconstruction.evidence import build_exclusion_mask, field_evidence


def test_exclusion_mask_rejects_dynamic_mask_with_wrong_shape():
    with pytest.raises(ValueError, match="mask shape"):
        build_exclusion_mask(
            (864, 614), [], np.ones((100, 100), dtype=bool)
        )


def test_exclusion_mask_combines_rectangles_and_dynamic_pixels():
    dynamic = np.zeros((10, 20), dtype=bool)
    dynamic[8, 18] = True

    mask = build_exclusion_mask((20, 10), [(0.0, 0.0, 0.5, 0.5)], dynamic)

    assert mask[:5, :10].all()
    assert mask[8, 18]
    assert not mask[7, 17]


def test_field_evidence_removes_excluded_bright_line():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[10, :] = 255
    excluded = np.zeros((20, 30), dtype=bool)
    excluded[:, :15] = True

    evidence = field_evidence(frame, excluded)

    assert evidence.line_likelihood[10, :15].sum() == 0
    assert evidence.line_likelihood[10, 16:].sum() > 0
    assert evidence.distance.shape == (20, 30)

