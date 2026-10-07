import numpy as np
from PIL import Image, ImageDraw
import torch

from nfl_reconstruction.camera import project_field
from nfl_reconstruction.evidence import field_evidence
from nfl_reconstruction.field import NFLField
from nfl_reconstruction.optimize import SequenceFitConfig, fit_camera_sequence
from tests.fixtures import synthetic_state


def render_evidence(state, blank=False):
    image = Image.new("RGB", (432, 307), (30, 100, 65))
    if not blank:
        draw = ImageDraw.Draw(image)
        for start, end in NFLField.standard().marking_segments():
            samples = np.linspace(start, end, 80)
            pixels = project_field(
                torch.tensor(samples, dtype=torch.float64),
                state,
                (432, 307),
            ).detach().numpy()
            draw.line([tuple(point) for point in pixels], fill=(255, 255, 255), width=2)
    frame = np.asarray(image)
    return field_evidence(frame, np.zeros(frame.shape[:2], dtype=bool))


def test_windowed_fit_recovers_smooth_pan():
    states = [synthetic_state(target_x=53.0 + index * 0.4, focal_px=360) for index in range(7)]
    evidence = [render_evidence(state) for state in states]

    result = fit_camera_sequence(
        evidence,
        {0: states[0], 6: states[-1]},
        SequenceFitConfig(window_size=5, overlap=2, steps_per_frame=35),
    )

    targets = np.array([[frame.state.target_x, frame.state.target_y] for frame in result])
    assert np.max(np.abs(np.diff(targets, n=2, axis=0))) < 0.15
    assert np.median([frame.reprojection_error_px for frame in result]) < 2.0


def test_low_confidence_frame_is_preserved_and_not_an_anchor():
    states = [synthetic_state(target_x=54.0 + index * 0.3, focal_px=360) for index in range(7)]
    evidence = [render_evidence(state, blank=index == 3) for index, state in enumerate(states)]

    result = fit_camera_sequence(
        evidence,
        {0: states[0], 6: states[-1]},
        SequenceFitConfig(window_size=5, overlap=2, steps_per_frame=25),
    )

    assert result[3].status == "uncertain"
    assert result[3].confidence < 0.25
    assert result[4].anchor_source != 3
