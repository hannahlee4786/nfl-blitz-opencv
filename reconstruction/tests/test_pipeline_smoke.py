import json
from pathlib import Path

from nfl_reconstruction.pipeline import run_pipeline


class SpyRunner:
    def __init__(self):
        self.prepare_calls = 0

    def prepare(self, output_dir: Path, config: dict):
        self.prepare_calls += 1
        (output_dir / "prepared.json").write_text(json.dumps({"ok": True}))


def test_pipeline_reuses_stage_with_matching_input_hash(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "run_id": "smoke",
                "video": {
                    "schema_version": 1, "path": "fixture.mp4", "sha256": "a" * 64,
                    "width": 16, "height": 12, "fps_numerator": 5,
                    "fps_denominator": 1, "frame_count": 3
                },
                "stages": ["prepare"]
            }
        )
    )
    runner = SpyRunner()

    first = run_pipeline(config, tmp_path / "run", runner=runner)
    second = run_pipeline(config, tmp_path / "run", runner=runner)

    assert first.stages["prepare"].status == "completed"
    assert second.stages["prepare"].status == "reused"
    assert runner.prepare_calls == 1
