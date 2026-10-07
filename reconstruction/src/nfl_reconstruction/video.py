"""Deterministic video probing, decoding, and frame manifests."""

from __future__ import annotations

import hashlib
import os
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .schemas import FrameRecord, RunManifest, VideoInfo


class VideoDecodeError(RuntimeError):
    def __init__(self, frame_index: int, message: str):
        super().__init__(f"Could not decode frame {frame_index}: {message}")
        self.frame_index = frame_index


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: str | Path) -> VideoInfo:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Video does not exist: {source}")
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            if rate is None:
                raise ValueError("Video does not report a frame rate.")
            rate = Fraction(rate)
            return VideoInfo(
                path=str(source.resolve()),
                sha256=_file_sha256(source),
                width=stream.width,
                height=stream.height,
                fps_numerator=rate.numerator,
                fps_denominator=rate.denominator,
                frame_count=stream.frames,
            )
    except (av.error.FFmpegError, IndexError) as error:
        raise VideoDecodeError(0, str(error)) from error


def _decode_frames(path: str | Path):
    try:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")
    except av.error.FFmpegError as error:
        raise VideoDecodeError(0, str(error)) from error


def read_frame(path: str | Path, frame_index: int) -> NDArray[np.uint8]:
    if frame_index < 0:
        raise VideoDecodeError(frame_index, "frame index cannot be negative")
    for index, frame in enumerate(_decode_frames(path)):
        if index == frame_index:
            return np.asarray(frame, dtype=np.uint8)
    raise VideoDecodeError(frame_index, "video ended before the requested frame")


def _save_png_atomic(frame: NDArray[np.uint8], output_path: Path) -> None:
    temporary_path = output_path.with_suffix(".png.tmp")
    Image.fromarray(frame, mode="RGB").save(temporary_path, format="PNG")
    os.replace(temporary_path, output_path)


def build_manifest(
    path: str | Path,
    frames_dir: Path,
    selected: range | None = None,
) -> RunManifest:
    source = Path(path)
    video = probe_video(source)
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    decoded_count = 0
    for index, frame in enumerate(_decode_frames(source)):
        decoded_count += 1
        if frame.shape != (video.height, video.width, 3):
            raise ValueError(
                f"Decoded frame {index} dimensions {frame.shape[:2]} do not "
                f"match {(video.height, video.width)}."
            )
        if selected is not None and index not in selected:
            continue
        output_path = frames_dir / f"{index:06d}.png"
        _save_png_atomic(frame, output_path)
        records.append(
            FrameRecord(
                index=index,
                path=str(output_path.resolve()),
                sha256=_file_sha256(output_path),
            )
        )
    if decoded_count == 0:
        raise VideoDecodeError(0, "video contains no decodable frames")
    if video.frame_count == 0:
        video = video.model_copy(update={"frame_count": decoded_count})
    return RunManifest(
        run_id=f"{source.stem}-{video.sha256[:12]}",
        video=video,
        frames=tuple(records),
    )

