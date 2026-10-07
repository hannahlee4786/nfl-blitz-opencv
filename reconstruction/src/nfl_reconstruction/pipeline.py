"""Atomic and resumable stage orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Protocol

from .schemas import RunManifest, StageRecord, VideoInfo


class StageRunner(Protocol):
    """Object implementing one method per configured pipeline stage."""

    def __getattr__(self, name: str) -> Any: ...


def _stage_hash(config_bytes: bytes, stage: str) -> str:
    digest = hashlib.sha256()
    digest.update(config_bytes)
    digest.update(b"\0")
    digest.update(stage.encode("utf-8"))
    digest.update(b"\0")
    package_dir = Path(__file__).parent
    for source in sorted(package_dir.glob("*.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
    return digest.hexdigest()


def run_pipeline(
    config_path: str | Path,
    run_dir: str | Path,
    stages: list[str] | tuple[str, ...] | None = None,
    *,
    runner: StageRunner,
) -> RunManifest:
    """Run requested stages once and reuse matching atomic outputs."""

    config_path = Path(config_path)
    run_dir = Path(run_dir)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    selected_stages = tuple(stages or config.get("stages", ()))
    video = VideoInfo.model_validate(config["video"])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"

    previous = None
    if manifest_path.exists():
        previous = RunManifest.model_validate_json(manifest_path.read_text())

    records: dict[str, StageRecord] = {}
    for stage in selected_stages:
        input_hash = _stage_hash(config_bytes, stage)
        output_dir = run_dir / stage
        old_record = previous.stages.get(stage) if previous else None
        if (
            old_record is not None
            and old_record.input_hash == input_hash
            and old_record.status in {"completed", "reused"}
            and output_dir.is_dir()
            and any(output_dir.iterdir())
        ):
            records[stage] = old_record.model_copy(update={"status": "reused"})
            continue

        stage_method = getattr(runner, stage)
        temporary = Path(tempfile.mkdtemp(prefix=f".{stage}-", dir=run_dir))
        try:
            stage_method(temporary, config)
            if not any(temporary.iterdir()):
                raise RuntimeError(f"Stage {stage!r} produced no outputs")
            if output_dir.exists():
                shutil.rmtree(output_dir)
            os.replace(temporary, output_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        records[stage] = StageRecord(
            name=stage,
            status="completed",
            input_hash=input_hash,
            output_path=str(output_dir),
        )

    if previous:
        for name, record in previous.stages.items():
            records.setdefault(name, record)

    manifest = RunManifest(run_id=config["run_id"], video=video, stages=records)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(manifest.model_dump_json(indent=2))
    os.replace(temporary_manifest, manifest_path)
    return manifest
