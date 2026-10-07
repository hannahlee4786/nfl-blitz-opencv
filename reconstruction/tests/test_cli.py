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
