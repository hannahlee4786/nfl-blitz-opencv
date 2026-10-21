import json

from typer.testing import CliRunner

from nfl_reconstruction.cli import app


def test_run_help_lists_all_pipeline_stages():
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    for stage in ["prepare", "camera", "mosaic", "detect", "track", "evaluate"]:
        assert stage in result.stdout


def test_dry_run_resolves_clip3_without_writing(tmp_path):
    clip = tmp_path / "cropped_clip_3.mp4"
    clip.write_bytes(b"fixture")
    anchor = tmp_path / "anchor.json"
    anchor.write_text("{}")
    config = tmp_path / "clip3.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": clip.name,
                "anchor": anchor.name,
                "stages": ["prepare", "camera", "mosaic"],
            }
        )
    )
    run_dir = tmp_path / "run"

    result = CliRunner().invoke(
        app,
        ["run", "--config", str(config), "--run-dir", str(run_dir), "--dry-run"],
    )

    assert result.exit_code == 0
    assert "cropped_clip_3.mp4" in result.stdout
    assert "prepare,camera,mosaic" in result.stdout
    assert not run_dir.exists()


def test_clip_reuses_anchor_and_writes_config(tmp_path, monkeypatch):
    from nfl_reconstruction import cli

    video = sorted((cli.RECONSTRUCTION_ROOT.parent / "data" / "input").glob("Clip_4*.mp4"))
    if not video:
        import pytest

        pytest.skip("Clip 4 is not in data/input")
    root = tmp_path / "reconstruction"
    (root / "annotations").mkdir(parents=True)
    anchor = cli.RECONSTRUCTION_ROOT / "annotations" / "clip4-full-anchor.json"
    (root / "annotations" / "clip4-anchor.json").write_text(anchor.read_text())
    monkeypatch.setattr(cli, "RECONSTRUCTION_ROOT", root)
    calls = []
    monkeypatch.setattr(cli, "_run_and_report", lambda *args: calls.append(args))

    result = CliRunner().invoke(app, ["clip", str(video[0]), "--name", "clip4"])

    assert result.exit_code == 0, result.output
    assert "Reusing anchor" in result.stdout
    config = json.loads((root / "configs" / "clip4.json").read_text())
    assert config["default_frame_range"] == "0:260"
    assert (root / "configs" / config["anchor"]).resolve() == (
        root / "annotations" / "clip4-anchor.json"
    )
    assert (root / "configs" / config["input"]).resolve() == video[0].resolve()
    config_path, run_dir, stages, frame_range, input_path = calls[0]
    assert run_dir == root / "runs" / "clip4"
    assert stages == ["prepare", "camera", "mosaic"]


def test_clip_without_frame_picks_anchor_frame_in_browser(tmp_path, monkeypatch):
    import numpy as np

    from nfl_reconstruction import cli

    video = sorted((cli.RECONSTRUCTION_ROOT.parent / "data" / "input").glob("Clip_4*.mp4"))
    if not video:
        import pytest

        pytest.skip("Clip 4 is not in data/input")
    root = tmp_path / "reconstruction"
    monkeypatch.setattr(cli, "RECONSTRUCTION_ROOT", root)
    monkeypatch.setattr(cli, "_run_and_report", lambda *args: None)
    image = np.zeros((1016, 1356, 3), dtype=np.uint8)
    monkeypatch.setattr(cli, "browse_frames", lambda path: (37, image))
    monkeypatch.setattr(cli, "click_image_points", lambda frame, labels: np.zeros((6, 2)))
    monkeypatch.setattr(
        cli, "anchor_field_points", lambda numbers, clicks: np.zeros((6, 2))
    )
    built = []

    def fake_build_anchor(field_points, clicks, frame, size):
        built.append((frame, size))
        return {"frame": frame, "field_points": [[0, 0]] * 6}, 0.5

    monkeypatch.setattr(cli, "build_anchor", fake_build_anchor)

    result = CliRunner().invoke(app, ["clip", str(video[0]), "--name", "clip4"])

    assert result.exit_code == 0, result.output
    assert built == [(37, (1356, 1016))]
    anchor = json.loads((root / "annotations" / "clip4-anchor.json").read_text())
    assert anchor["frame"] == 37


def test_clip_name_from_video_file():
    from pathlib import Path

    from nfl_reconstruction.cli import _clip_name

    assert _clip_name(Path("Clip_5 Goal Line Test.mp4")) == "clip_5-goal-line-test"
