from fractions import Fraction
import json

import av
import numpy as np
from typer.testing import CliRunner

from nfl_reconstruction import cli
from nfl_reconstruction.schemas import CameraFrame
from tests.fixtures import FOCAL, FRAME_SIZE, blitz_homography, render_field

FRAME_COUNT = 10
ANCHOR_FRAME = 4

# The centers of the painted numbers on the 20, 30, and 40-yard lines.
NUMBER_CENTERS = [(x, y) for x in (30.0, 40.0, 50.0) for y in (15.8, 37.533)]


def clip_cameras():
    """A camera backing up the field while it pans a little."""

    return [
        blitz_homography(
            (83.0 - 0.5 * frame, 26.67 + 0.2 * frame, 15.7, 180.0 - 0.3 * frame, 27.0, 0.0)
        )
        for frame in range(FRAME_COUNT)
    ]


def write_clip(path, homographies):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.width, stream.height = FRAME_SIZE
        stream.pix_fmt = "bgr0"
        for homography in homographies:
            frame = av.VideoFrame.from_ndarray(render_field(homography), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_anchor(path, homography):
    field = np.column_stack([NUMBER_CENTERS, np.ones(len(NUMBER_CENTERS))])
    screen = field @ homography.T
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame": ANCHOR_FRAME,
                "image_width": FRAME_SIZE[0],
                "image_height": FRAME_SIZE[1],
                "field_points": NUMBER_CENTERS,
                "image_points": (screen[:, :2] / screen[:, 2:]).round(1).tolist(),
            }
        )
    )


def test_camera_stage_tracks_solves_and_scores_the_clip(tmp_path):
    cameras = clip_cameras()
    write_clip(tmp_path / "clip.mkv", cameras)
    write_anchor(tmp_path / "anchor.json", cameras[ANCHOR_FRAME])
    config = tmp_path / "clip.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "synthetic-clip",
                "input": "clip.mkv",
                "anchor": "anchor.json",
                "default_frame_range": f"0:{FRAME_COUNT}",
                "hud_rectangles": [],
            }
        )
    )
    run_dir = tmp_path / "run"

    result = CliRunner().invoke(
        cli.app,
        ["run", "--config", str(config), "--run-dir", str(run_dir), "--stages", "prepare,camera,mosaic"],
    )

    assert result.exit_code == 0, result.output
    assert "Lens: fixed" in result.stdout
    records = [
        CameraFrame.model_validate_json(line)
        for line in (run_dir / "camera" / "camera.jsonl").read_text().splitlines()
    ]
    assert [record.frame for record in records] == list(range(FRAME_COUNT))
    assert all(record.status == "reliable" for record in records)
    assert all(record.source == "snapped" for record in records)
    assert max(record.line_error_px for record in records) < 1.5
    assert all(record.state.up_sign == 1 for record in records)
    assert abs(records[0].state.focal_px - FOCAL) < 0.02 * FOCAL
    pose = json.loads((run_dir / "camera" / "camera-pose.json").read_text())
    assert len(pose["frames"]) == FRAME_COUNT
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert evaluation["lost_fraction"] == 0.0
    assert evaluation["active_area"] == [0, FRAME_SIZE[0]]
    assert evaluation["coverage_fraction"] > 0
    preview_path = run_dir / "mosaic" / "reconstruction-preview.mp4"
    assert preview_path.is_file()
    with av.open(str(preview_path)) as container:
        stream = container.streams.video[0]
        preview_frames = list(container.decode(video=0))
        assert Fraction(stream.average_rate) == Fraction(30, 1)
    assert len(preview_frames) == FRAME_COUNT
    assert (preview_frames[0].width, preview_frames[0].height) == (1280, 720)
