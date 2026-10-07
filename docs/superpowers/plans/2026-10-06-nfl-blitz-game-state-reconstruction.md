# NFL Blitz Game-State Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Python 3.11 workflow that reconstructs a continuous NFL Blitz clip as a static field, a temporally smooth moving virtual camera, and player tracks in field coordinates.

**Architecture:** A new `reconstruction/` package owns deterministic video preparation, NFL field geometry, differentiable camera fitting, field-mosaic accumulation, detector adapters, track association, projection, visualization, and evaluation. Stages exchange versioned JSON, JSONL, CSV, NPZ, PNG, and MP4 artifacts so expensive stages can be replaced or replayed independently; the existing `src/` package remains untouched.

**Tech Stack:** Python 3.11, NumPy, PyTorch, Kornia, PyAV, Pillow, SciPy, Pydantic 2, Typer, pytest; Ultralytics is an optional training/inference dependency.

**Spec:** `docs/superpowers/specs/2026-10-06-nfl-blitz-game-state-reconstruction-design.md`

## Global Constraints

- The workflow must live outside `src/` and must not import or modify the existing OpenCV tracker.
- The runtime version is Python 3.11.
- The MVP input is one continuous gameplay clip with no cuts or replays.
- Manual initialization and manual correction anchors are allowed.
- The authoritative field is 120 yards long by 53.333 yards wide, including two 10-yard end zones.
- Every camera and player result must carry confidence; uncertain or missing results must remain explicit in output records.
- Unit tests must not download model weights or run full detector training.
- Generated artifacts belong under `reconstruction/runs/<run-id>/` and must not be committed.
- Implementation follows red-green-refactor; each task ends with the full reconstruction test suite passing.

## Review Focus

- Degenerate or mislabeled anchor points must fail validation with an actionable error instead of producing a plausible-looking projection; Task 2 pins this behavior.
- Corrupt, missing, or dimension-changing video frames must be represented explicitly in the manifest or stop preparation cleanly; Task 3 pins this behavior.
- Camera states that place the field behind the camera or outside the image must receive low confidence and must not be used as reliable anchors; Task 4 pins this behavior.
- HUD or dynamic masks with the wrong shape must be rejected before optimization or mosaic accumulation; Tasks 5 and 7 pin this behavior.
- Low-confidence camera frames must not contribute field-space distance to player association or emit unflagged field coordinates; Task 9 pins this behavior.

---

## File Structure

```text
reconstruction/
├── pyproject.toml                         # isolated package, runtime, optional ML extras, pytest config
├── README.md                              # setup, annotation, commands, artifacts, MVP limitations
├── .gitignore                             # environments, model weights, decoded frames, and runs
├── configs/
│   └── clip3.json                         # reproducible Clip 3 paths, masks, windows, thresholds
├── annotations/
│   └── clip3-anchor.json                  # seed anchor converted from the known-good calibration
├── src/nfl_reconstruction/
│   ├── __init__.py                        # package version only
│   ├── schemas.py                         # versioned Pydantic interchange schemas
│   ├── field.py                           # fixed NFL geometry and renderable marking segments
│   ├── video.py                           # PyAV decoding, hashing, manifests, frame selection
│   ├── annotations.py                     # anchor and player-label validation/conversion
│   ├── camera.py                          # differentiable constrained camera and homographies
│   ├── evidence.py                        # line/edge evidence and exclusion-mask validation
│   ├── optimize.py                        # anchor fitting and overlapping-window sequence fitting
│   ├── mosaic.py                          # robust top-down texture accumulation
│   ├── detection.py                       # detector protocol, recorded detections, Ultralytics adapter
│   ├── tracking.py                        # screen/field association and track lifecycle
│   ├── visualization.py                   # diagnostic overlay and static minimap renderers
│   ├── evaluation.py                      # reprojection, detection, fragmentation, and run report metrics
│   ├── pipeline.py                        # resumable stage orchestration and provenance
│   └── cli.py                             # Typer commands
└── tests/
    ├── fixtures.py                        # synthetic field/camera/video/detection fixtures
    ├── test_field.py
    ├── test_schemas.py
    ├── test_video.py
    ├── test_camera.py
    ├── test_evidence.py
    ├── test_optimize.py
    ├── test_mosaic.py
    ├── test_detection.py
    ├── test_tracking.py
    ├── test_visualization.py
    ├── test_evaluation.py
    └── test_pipeline_smoke.py
```

### Task 1: Isolated package, schemas, and NFL field model

**Files:**
- Create: `reconstruction/pyproject.toml`
- Create: `reconstruction/.gitignore`
- Create: `reconstruction/src/nfl_reconstruction/__init__.py`
- Create: `reconstruction/src/nfl_reconstruction/schemas.py`
- Create: `reconstruction/src/nfl_reconstruction/field.py`
- Create: `reconstruction/tests/test_schemas.py`
- Create: `reconstruction/tests/test_field.py`

**Interfaces:**
- Consumes: no earlier task interfaces.
- Produces: `FieldPoint`, `ImagePoint`, `AnchorSet`, `CameraState`, `CameraFrame`, `DetectionRecord`, `TrackRecord`, `RunManifest`; `NFLField.standard()`; `NFLField.marking_segments()`; `NFLField.contains(points)`.

- [ ] **Step 1: Write failing schema and field tests**

```python
def test_standard_field_contains_regulation_points():
    field = NFLField.standard()
    assert field.length_yards == 120.0
    assert field.width_yards == pytest.approx(53.333)
    assert field.contains(np.array([[0.0, 0.0], [120.0, 53.333]])).all()
    assert not field.contains(np.array([[-0.01, 20.0], [60.0, 54.0]])).any()


def test_camera_frame_requires_explicit_confidence():
    with pytest.raises(ValidationError, match="confidence"):
        CameraFrame.model_validate({"schema_version": 1, "frame": 0, "state": {}})
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_schemas.py tests/test_field.py -v`

Expected: collection fails because `nfl_reconstruction.schemas` and `nfl_reconstruction.field` do not exist.

- [ ] **Step 3: Add the package metadata and minimal typed models**

```toml
[project]
name = "nfl-blitz-reconstruction"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "av>=13,<14", "kornia>=0.7,<0.9", "numpy>=1.26,<3",
  "pillow>=10,<13", "pydantic>=2.8,<3", "scipy>=1.13,<2",
  "torch>=2.3,<3", "typer>=0.12,<1"
]

[project.optional-dependencies]
detector = ["ultralytics>=8.3,<9"]
test = ["pytest>=8,<9"]

[project.scripts]
nfl-reconstruct = "nfl_reconstruction.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Implement immutable Pydantic records with `schema_version: Literal[1]`, finite numeric validation, frame indices greater than or equal to zero, and confidence constrained to `[0, 1]`. Implement the regulation field dimensions, goal lines, 10-yard lines, hash rows at `23.58` and `29.75`, sidelines, and bounds checks.

- [ ] **Step 4: Run the task tests and the full reconstruction suite**

Run: `cd reconstruction && python3.11 -m pytest tests/test_schemas.py tests/test_field.py -v && python3.11 -m pytest -q`

Expected: both commands pass with no warnings.

- [ ] **Step 5: Commit the isolated foundation**

```bash
git add reconstruction/pyproject.toml reconstruction/.gitignore reconstruction/src reconstruction/tests/test_schemas.py reconstruction/tests/test_field.py
git commit -m "feat: add reconstruction field and interchange schemas"
```

### Task 2: Anchor annotation validation and Clip 3 seed

**Files:**
- Create: `reconstruction/src/nfl_reconstruction/annotations.py`
- Create: `reconstruction/tests/test_annotations.py`
- Create: `reconstruction/annotations/clip3-anchor.json`

**Interfaces:**
- Consumes: `AnchorSet`, `FieldPoint`, and `ImagePoint` from `schemas.py`; `NFLField.contains()`.
- Produces: `load_anchor(path: Path, image_size: tuple[int, int]) -> AnchorSet`; `validate_anchor(anchor: AnchorSet, field: NFLField, image_size: tuple[int, int]) -> None`; `anchor_homography(anchor: AnchorSet) -> np.ndarray` mapping field yards to image pixels.

- [ ] **Step 1: Write failing tests for valid and degenerate anchors**

```python
def test_anchor_rejects_collinear_field_points(tmp_path):
    anchor = make_anchor(
        field=[(10, 20), (20, 20), (30, 20), (40, 20)],
        image=[(100, 100), (200, 100), (300, 100), (400, 100)],
    )
    with pytest.raises(ValueError, match="non-collinear"):
        validate_anchor(anchor, NFLField.standard(), (864, 614))


def test_anchor_rejects_points_outside_image():
    anchor = make_anchor(image=[(100, 100), (200, 100), (300, 200), (900, 300)])
    with pytest.raises(ValueError, match="image bounds"):
        validate_anchor(anchor, NFLField.standard(), (864, 614))
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_annotations.py -v`

Expected: FAIL because the annotation module is missing.

- [ ] **Step 3: Implement validation and seed the known calibration**

Use normalized Direct Linear Transform with an SVD rank check and require at least four unique, non-collinear point pairs. The checked-in Clip 3 anchor records frame `26`, image size `864x614`, source `../cropped_clip_3.mp4`, and the six known correspondences already stored in `data/output/cropped_clip_3/initial_homography.npz`:

```json
{
  "schema_version": 1,
  "frame": 26,
  "image_width": 864,
  "image_height": 614,
  "field_points": [[50.0,23.58],[50.0,29.75],[55.0,23.58],[55.0,29.75],[60.0,23.58],[60.0,29.75]],
  "image_points": [[254.0,101.0],[533.0,101.0],[233.0,202.0],[530.0,197.0],[193.0,335.0],[519.0,334.0]]
}
```

- [ ] **Step 4: Verify numeric recovery and full suite**

Run: `cd reconstruction && python3.11 -m pytest tests/test_annotations.py -v && python3.11 -m pytest -q`

Expected: the recovered homography reprojects the seed points with mean error below 5 pixels; the full suite passes.

- [ ] **Step 5: Commit anchor support**

```bash
git add reconstruction/src/nfl_reconstruction/annotations.py reconstruction/tests/test_annotations.py reconstruction/annotations/clip3-anchor.json
git commit -m "feat: validate manual camera anchors"
```

### Task 3: Deterministic video manifest and frame extraction

**Files:**
- Create: `reconstruction/src/nfl_reconstruction/video.py`
- Create: `reconstruction/tests/test_video.py`
- Modify: `reconstruction/src/nfl_reconstruction/schemas.py`
- Create: `reconstruction/configs/clip3.json`

**Interfaces:**
- Consumes: `RunManifest` and `FrameRecord` schemas.
- Produces: `probe_video(path: Path) -> VideoInfo`; `build_manifest(path: Path, frames_dir: Path, selected: range | None = None) -> RunManifest`; `read_frame(path: Path, frame_index: int) -> np.ndarray` in RGB order.

- [ ] **Step 1: Write failing tests with a generated three-frame fixture**

```python
def test_manifest_hashes_and_numbers_every_decoded_frame(tiny_video, tmp_path):
    manifest = build_manifest(tiny_video, tmp_path / "frames")
    assert [frame.index for frame in manifest.frames] == [0, 1, 2]
    assert all(len(frame.sha256) == 64 for frame in manifest.frames)
    assert (tmp_path / "frames" / "000000.png").exists()


def test_dimension_change_stops_with_frame_number(changing_size_video, tmp_path):
    with pytest.raises(ValueError, match="frame 2.*dimensions"):
        build_manifest(changing_size_video, tmp_path / "frames")
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_video.py -v`

Expected: FAIL because video preparation is not implemented.

- [ ] **Step 3: Implement PyAV decoding and the Clip 3 configuration**

Decode on the video stream time base, convert to RGB24, number frames by decode order, and atomically save PNGs. Record source SHA-256, FPS as numerator/denominator, dimensions, frame count, per-frame checksum, and decode errors. A missing or corrupt requested frame raises `VideoDecodeError(frame_index, message)`.

`configs/clip3.json` must point to `../../cropped_clip_3.mp4`, the checked-in anchor, `window_size: 45`, `window_overlap: 15`, and normalized HUD rectangles for the upper-left scoreboard and bottom-center banner.

- [ ] **Step 4: Verify focused tests and full suite**

Run: `cd reconstruction && python3.11 -m pytest tests/test_video.py -v && python3.11 -m pytest -q`

Expected: all tests pass and generated fixtures decode deterministically.

- [ ] **Step 5: Commit deterministic preparation**

```bash
git add reconstruction/src/nfl_reconstruction/video.py reconstruction/src/nfl_reconstruction/schemas.py reconstruction/tests/test_video.py reconstruction/configs/clip3.json
git commit -m "feat: add deterministic clip preparation"
```

### Task 4: Differentiable virtual-camera model

**Files:**
- Create: `reconstruction/src/nfl_reconstruction/camera.py`
- Create: `reconstruction/tests/test_camera.py`
- Modify: `reconstruction/tests/fixtures.py`

**Interfaces:**
- Consumes: `CameraState`; field points shaped `(..., 2)` in yards.
- Produces: `camera_matrix(state, image_size) -> torch.Tensor`; `world_to_camera(state) -> torch.Tensor`; `field_homography(state, image_size) -> torch.Tensor`; `project_field(points, state, image_size) -> torch.Tensor`; `unproject_image(points, homography) -> torch.Tensor`; `camera_visibility(state, field, image_size) -> float`.

- [ ] **Step 1: Write failing synthetic round-trip and invalid-view tests**

```python
def test_project_then_unproject_recovers_field_points(synthetic_camera):
    points = torch.tensor([[50.0, 20.0], [55.0, 30.0], [60.0, 25.0]])
    H = field_homography(synthetic_camera, (864, 614))
    pixels = project_field(points, synthetic_camera, (864, 614))
    recovered = unproject_image(pixels, H)
    torch.testing.assert_close(recovered, points, atol=1e-3, rtol=1e-4)


def test_camera_looking_away_has_zero_visibility():
    state = synthetic_state(yaw_degrees=180.0, pitch_degrees=20.0)
    assert camera_visibility(state, NFLField.standard(), (864, 614)) == 0.0
```

- [ ] **Step 2: Run the camera tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_camera.py -v`

Expected: FAIL because camera projection functions are absent.

- [ ] **Step 3: Implement the constrained pinhole camera**

Construct the camera center from target `(x, y, 0)`, distance, yaw, and pitch; build an orthonormal look-at rotation; use pixel focal length and centered principal point for intrinsics; allow roll but initialize and regularize it to zero. Derive the plane homography from columns 0, 1, and 3 of `K[R|t]`. Reject non-finite state values, non-positive focal length/distance, and singular homographies.

- [ ] **Step 4: Verify analytic behavior and full suite**

Run: `cd reconstruction && python3.11 -m pytest tests/test_camera.py -v && python3.11 -m pytest -q`

Expected: round-trip error is below `1e-3` yards and invalid views report zero confidence.

- [ ] **Step 5: Commit camera geometry**

```bash
git add reconstruction/src/nfl_reconstruction/camera.py reconstruction/tests/test_camera.py reconstruction/tests/fixtures.py
git commit -m "feat: add differentiable virtual camera"
```

### Task 5: Field evidence and single-frame camera fitting

**Files:**
- Create: `reconstruction/src/nfl_reconstruction/evidence.py`
- Create: `reconstruction/src/nfl_reconstruction/optimize.py`
- Create: `reconstruction/tests/test_evidence.py`
- Create: `reconstruction/tests/test_optimize.py`

**Interfaces:**
- Consumes: decoded RGB frames, normalized HUD rectangles, optional boolean dynamic masks, `AnchorSet`, and camera functions from Task 4.
- Produces: `build_exclusion_mask(image_size, rectangles, dynamic_mask=None) -> np.ndarray`; `field_evidence(frame, exclusion_mask) -> EvidenceMap`; `fit_anchor_camera(anchor, initial, image_size, steps=500) -> CameraFit`; `camera_confidence(losses, visibility) -> float`.

- [ ] **Step 1: Write failing mask, synthetic fitting, and invalid-view tests**

```python
def test_exclusion_mask_rejects_dynamic_mask_with_wrong_shape():
    with pytest.raises(ValueError, match="mask shape"):
        build_exclusion_mask((864, 614), [], np.ones((100, 100), dtype=bool))


def test_anchor_fit_recovers_perturbed_camera(synthetic_anchor, synthetic_camera):
    start = dataclasses.replace(synthetic_camera, target_x=synthetic_camera.target_x + 4, focal_px=700)
    result = fit_anchor_camera(synthetic_anchor, start, (864, 614), steps=300)
    assert result.mean_reprojection_error_px < 1.0
    assert result.confidence > 0.9
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_evidence.py tests/test_optimize.py -v`

Expected: FAIL because evidence and fitting functions do not exist.

- [ ] **Step 3: Implement evidence extraction and anchor optimization**

Generate grayscale Sobel magnitude, bright low-saturation line likelihood, and a Euclidean distance transform. Apply exclusion masks before normalization. Fit camera parameters with Adam against Huber reprojection loss plus weak roll and parameter-bound penalties. Retain the best finite state, report individual loss terms, and set confidence to zero if visibility is below `0.05` or the final homography is singular.

- [ ] **Step 4: Verify recovery and full suite**

Run: `cd reconstruction && python3.11 -m pytest tests/test_evidence.py tests/test_optimize.py -v && python3.11 -m pytest -q`

Expected: synthetic camera recovery passes; malformed masks and invalid cameras fail clearly.

- [ ] **Step 5: Commit single-frame fitting**

```bash
git add reconstruction/src/nfl_reconstruction/evidence.py reconstruction/src/nfl_reconstruction/optimize.py reconstruction/tests/test_evidence.py reconstruction/tests/test_optimize.py
git commit -m "feat: fit virtual camera from field evidence"
```

### Task 6: Overlapping-window temporal camera recovery

**Files:**
- Modify: `reconstruction/src/nfl_reconstruction/optimize.py`
- Modify: `reconstruction/src/nfl_reconstruction/schemas.py`
- Create: `reconstruction/tests/test_sequence_optimize.py`

**Interfaces:**
- Consumes: `Sequence[EvidenceMap]`, anchor states keyed by frame, window size, overlap, and confidence threshold.
- Produces: `fit_camera_sequence(evidence, anchors, config) -> list[CameraFrame]`; `merge_window_results(windows, overlap) -> list[CameraFrame]`.

- [ ] **Step 1: Write failing temporal and confidence tests**

```python
def test_windowed_fit_recovers_smooth_pan(synthetic_pan_sequence):
    result = fit_camera_sequence(**synthetic_pan_sequence)
    targets = np.array([[frame.state.target_x, frame.state.target_y] for frame in result])
    assert np.max(np.abs(np.diff(targets, n=2, axis=0))) < 0.15
    assert np.median([frame.reprojection_error_px for frame in result]) < 2.0


def test_low_confidence_frame_is_preserved_and_not_an_anchor(blank_middle_sequence):
    result = fit_camera_sequence(**blank_middle_sequence)
    assert result[5].status == "uncertain"
    assert result[5].confidence < 0.25
    assert result[6].anchor_source != 5
```

- [ ] **Step 2: Run the sequence tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_sequence_optimize.py -v`

Expected: FAIL because sequence fitting is missing.

- [ ] **Step 3: Implement windowed joint optimization**

Optimize data loss per frame, first-difference velocity loss, second-difference acceleration loss, and anchor loss. Use 45-frame windows with 15-frame overlap by default. Blend overlapping finite estimates by normalized confidence, but never promote an uncertain estimate to anchor status. Emit one record per input frame with status `reliable`, `uncertain`, or `missing`.

- [ ] **Step 4: Verify smooth recovery and full suite**

Run: `cd reconstruction && python3.11 -m pytest tests/test_sequence_optimize.py -v && python3.11 -m pytest -q`

Expected: synthetic pan remains smooth, blank evidence is explicitly uncertain, and all tests pass.

- [ ] **Step 5: Commit temporal recovery**

```bash
git add reconstruction/src/nfl_reconstruction/optimize.py reconstruction/src/nfl_reconstruction/schemas.py reconstruction/tests/test_sequence_optimize.py
git commit -m "feat: recover camera motion in temporal windows"
```

### Task 7: Robust static-field mosaic

**Files:**
- Create: `reconstruction/src/nfl_reconstruction/mosaic.py`
- Create: `reconstruction/tests/test_mosaic.py`

**Interfaces:**
- Consumes: RGB frames, matching `CameraFrame` records, optional dynamic masks, `NFLField`, pixels-per-yard, and confidence threshold.
- Produces: `build_mosaic(frames, camera_frames, dynamic_masks, field, pixels_per_yard=10) -> MosaicResult`; `MosaicResult.image`, `coverage`, and `sample_count` arrays.

- [ ] **Step 1: Write failing robust-aggregation and shape tests**

```python
def test_mosaic_rejects_transient_player_pixel(aligned_field_frames):
    frames, cameras, masks, expected = aligned_field_frames
    result = build_mosaic(frames, cameras, masks, NFLField.standard(), pixels_per_yard=4)
    assert np.mean(np.abs(result.image.astype(float) - expected.astype(float))) < 2.0


def test_mosaic_rejects_mask_shape_mismatch(aligned_field_frames):
    frames, cameras, _, _ = aligned_field_frames
    with pytest.raises(ValueError, match="mask shape"):
        build_mosaic(frames, cameras, [np.zeros((4, 4), bool)], NFLField.standard())
```

- [ ] **Step 2: Run mosaic tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_mosaic.py -v`

Expected: FAIL because mosaic accumulation is absent.

- [ ] **Step 3: Implement confidence-gated robust accumulation**

Warp frames into a 120-by-53.333-yard canvas with Kornia, discard excluded and out-of-bounds pixels, and aggregate samples by median in batches. Frames below the configured camera confidence do not contribute. Save coverage and sample-count maps so holes remain visible.

- [ ] **Step 4: Verify aggregation and full suite**

Run: `cd reconstruction && python3.11 -m pytest tests/test_mosaic.py -v && python3.11 -m pytest -q`

Expected: transient foreground is suppressed and invalid masks are rejected.

- [ ] **Step 5: Commit mosaic generation**

```bash
git add reconstruction/src/nfl_reconstruction/mosaic.py reconstruction/tests/test_mosaic.py
git commit -m "feat: build robust static field mosaic"
```

### Task 8: Player labels and replaceable detector adapters

**Files:**
- Create: `reconstruction/src/nfl_reconstruction/detection.py`
- Create: `reconstruction/tests/test_detection.py`
- Modify: `reconstruction/src/nfl_reconstruction/annotations.py`
- Modify: `reconstruction/pyproject.toml`

**Interfaces:**
- Consumes: RGB frames and YOLO-format labels with team classes `offense`, `defense`, and `unknown`.
- Produces: `PlayerDetector.detect(frame, frame_index) -> list[DetectionRecord]`; `RecordedDetector(path)`; `UltralyticsDetector(weights, confidence)`; `export_yolo_dataset(labels, frames_dir, output_dir) -> DatasetManifest`.

- [ ] **Step 1: Write failing adapter and annotation tests**

```python
def test_recorded_detector_returns_ground_contact_midpoint(tmp_path):
    path = write_detection_fixture(tmp_path, bbox=[100, 50, 140, 130], team="offense")
    detection = RecordedDetector(path).detect(blank_frame(), 7)[0]
    assert detection.ground_x == pytest.approx(120.0)
    assert detection.ground_y == pytest.approx(130.0)


def test_export_rejects_label_outside_frame(tmp_path):
    labels = [player_label(frame=0, bbox=(-1, 10, 20, 30))]
    with pytest.raises(ValueError, match="frame bounds"):
        export_yolo_dataset(labels, tmp_path / "frames", tmp_path / "dataset")
```

- [ ] **Step 2: Run detector tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_detection.py -v`

Expected: FAIL because detector adapters do not exist.

- [ ] **Step 3: Implement recorded inference, label export, and optional YOLO inference**

Define a `Protocol` for detectors. Store recorded detections as JSONL, validate unique detection IDs within each frame, and default the ground point to bottom-center when no explicit feet annotation exists. Import `ultralytics` only inside `UltralyticsDetector.__init__` and raise a message naming `pip install -e '.[detector]'` when unavailable. Export deterministic train/validation splits by frame index modulo five and record the source checksums.

- [ ] **Step 4: Verify adapters without downloading weights**

Run: `cd reconstruction && python3.11 -m pytest tests/test_detection.py -v && python3.11 -m pytest -q`

Expected: tests use `RecordedDetector`; no network request or model download occurs.

- [ ] **Step 5: Commit detector boundaries**

```bash
git add reconstruction/src/nfl_reconstruction/detection.py reconstruction/src/nfl_reconstruction/annotations.py reconstruction/tests/test_detection.py reconstruction/pyproject.toml
git commit -m "feat: add player detector adapters"
```

### Task 9: Player association and field projection

**Files:**
- Create: `reconstruction/src/nfl_reconstruction/tracking.py`
- Create: `reconstruction/tests/test_tracking.py`

**Interfaces:**
- Consumes: per-frame `DetectionRecord` lists, matching `CameraFrame` records, field model, and association thresholds.
- Produces: `track_players(detections, cameras, field, config) -> list[TrackRecord]`; `project_detection(detection, camera, field) -> ProjectedDetection`.

- [ ] **Step 1: Write failing occlusion, team, bounds, and confidence tests**

```python
def test_track_survives_two_frame_occlusion(linear_player_sequence):
    tracks = track_players(*linear_player_sequence, config=TrackingConfig(max_staleness=3))
    visible_ids = {record.track_id for record in tracks if record.detection_id is not None}
    assert len(visible_ids) == 1


def test_low_camera_confidence_disables_field_projection(detection, uncertain_camera):
    projected = project_detection(detection, uncertain_camera, NFLField.standard())
    assert projected.field_x is None
    assert projected.field_y is None
    assert projected.projection_status == "camera_uncertain"


def test_team_mismatch_prevents_association(two_team_crossing):
    tracks = track_players(*two_team_crossing, config=TrackingConfig())
    assert offense_track_ids(tracks).isdisjoint(defense_track_ids(tracks))
```

- [ ] **Step 2: Run tracking tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_tracking.py -v`

Expected: FAIL because player projection and association are absent.

- [ ] **Step 3: Implement confidence-aware assignment**

Use constant-velocity Kalman state in image coordinates, Hungarian assignment, IoU cost, image-center distance, team mismatch as an infinite cost, and field-space distance only when both frames have reliable camera estimates. Keep unmatched tracks for `max_staleness` frames. Emit explicit prediction-only rows during short occlusions. Flag out-of-field projections rather than clamping them.

- [ ] **Step 4: Verify association and full suite**

Run: `cd reconstruction && python3.11 -m pytest tests/test_tracking.py -v && python3.11 -m pytest -q`

Expected: IDs survive short occlusion, never cross teams, and uncertain cameras suppress field coordinates.

- [ ] **Step 5: Commit player reconstruction**

```bash
git add reconstruction/src/nfl_reconstruction/tracking.py reconstruction/tests/test_tracking.py
git commit -m "feat: track players in static field coordinates"
```

### Task 10: Diagnostics, evaluation, and resumable pipeline

**Files:**
- Create: `reconstruction/src/nfl_reconstruction/visualization.py`
- Create: `reconstruction/src/nfl_reconstruction/evaluation.py`
- Create: `reconstruction/src/nfl_reconstruction/pipeline.py`
- Create: `reconstruction/tests/test_visualization.py`
- Create: `reconstruction/tests/test_evaluation.py`
- Create: `reconstruction/tests/test_pipeline_smoke.py`

**Interfaces:**
- Consumes: run config, manifest, anchor, camera JSONL, mosaic NPZ, detections JSONL, and tracks CSV.
- Produces: `render_overlay(...)`; `render_minimap(...)`; `evaluate_run(...) -> EvaluationReport`; `run_pipeline(config_path, run_dir, stages=None) -> RunManifest`.

- [ ] **Step 1: Write failing metric and resumability tests**

```python
def test_evaluation_reports_reprojection_detection_and_fragmentation(eval_fixture):
    report = evaluate_run(**eval_fixture)
    assert report.median_reprojection_error_px == pytest.approx(2.5)
    assert report.detection_recall == pytest.approx(0.9)
    assert report.track_fragments == 1


def test_pipeline_reuses_stage_with_matching_input_hash(smoke_config, tmp_path, spy_runner):
    first = run_pipeline(smoke_config, tmp_path / "run")
    second = run_pipeline(smoke_config, tmp_path / "run")
    assert second.stages["prepare"].status == "reused"
    assert spy_runner.prepare_calls == 1
```

- [ ] **Step 2: Run diagnostics tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_visualization.py tests/test_evaluation.py tests/test_pipeline_smoke.py -v`

Expected: FAIL because visualization, evaluation, and orchestration are missing.

- [ ] **Step 3: Implement diagnostics and atomic stage orchestration**

Render overlays with Pillow using distinct reliable/uncertain colors, source-frame coordinates, projected field segments, boxes, IDs, and confidence. Render a regulation minimap with camera viewport and trajectory tails. Compute median and p95 reprojection error, camera second-difference smoothness, annotated detection recall, out-of-bounds rate, and track fragment count. Each stage writes to a temporary directory, validates outputs, then atomically renames it and records input/config/code hashes.

- [ ] **Step 4: Verify diagnostic outputs and full suite**

Run: `cd reconstruction && python3.11 -m pytest tests/test_visualization.py tests/test_evaluation.py tests/test_pipeline_smoke.py -v && python3.11 -m pytest -q`

Expected: expected PNG dimensions and colors match numeric fixtures; unchanged stages are reused; all tests pass.

- [ ] **Step 5: Commit orchestration and evaluation**

```bash
git add reconstruction/src/nfl_reconstruction/visualization.py reconstruction/src/nfl_reconstruction/evaluation.py reconstruction/src/nfl_reconstruction/pipeline.py reconstruction/tests/test_visualization.py reconstruction/tests/test_evaluation.py reconstruction/tests/test_pipeline_smoke.py
git commit -m "feat: add reconstruction diagnostics and evaluation"
```

### Task 11: CLI, Clip 3 smoke flow, and operator documentation

**Files:**
- Create: `reconstruction/src/nfl_reconstruction/cli.py`
- Create: `reconstruction/tests/test_cli.py`
- Create: `reconstruction/README.md`
- Modify: `reconstruction/configs/clip3.json`

**Interfaces:**
- Consumes: every pipeline interface from Tasks 1–10.
- Produces commands `prepare`, `fit-camera`, `build-mosaic`, `export-labels`, `detect`, `track`, `evaluate`, and `run`; documented Clip 3 workflow.

- [ ] **Step 1: Write failing CLI help and dry-run tests**

```python
def test_run_help_lists_all_pipeline_stages(cli_runner):
    result = cli_runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    for stage in ["prepare", "camera", "mosaic", "detect", "track", "evaluate"]:
        assert stage in result.stdout


def test_dry_run_resolves_clip3_without_writing(cli_runner, clip3_config, tmp_path):
    result = cli_runner.invoke(app, ["run", "--config", str(clip3_config), "--run-dir", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "cropped_clip_3.mp4" in result.stdout
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `cd reconstruction && python3.11 -m pytest tests/test_cli.py -v`

Expected: FAIL because the CLI is absent.

- [ ] **Step 3: Implement commands and exact operating instructions**

Use Typer with explicit config and run-directory arguments. `run --dry-run` validates paths and prints resolved stages without creating output. Document these commands:

```bash
cd reconstruction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
nfl-reconstruct run --config configs/clip3.json --run-dir runs/clip3-mvp --stages prepare,camera,mosaic
nfl-reconstruct export-labels --run-dir runs/clip3-mvp --output annotations/clip3-players
python -m pip install -e '.[detector]'
nfl-reconstruct detect --run-dir runs/clip3-mvp --weights weights/clip3-best.pt
nfl-reconstruct track --run-dir runs/clip3-mvp
nfl-reconstruct evaluate --run-dir runs/clip3-mvp
```

Document that the detector acceptance criterion cannot be measured until the 50–100-frame player annotation set exists, while camera/mosaic stages are independently runnable from the checked-in anchor.

- [ ] **Step 4: Run all automated verification and the bounded Clip 3 smoke test**

Run: `cd reconstruction && python3.11 -m pytest -q`

Run: `cd reconstruction && .venv/bin/nfl-reconstruct run --config configs/clip3.json --run-dir /tmp/nfl-blitz-clip3-smoke --stages prepare,camera,mosaic --frame-range 26:36`

Expected: unit suite passes; smoke run writes a manifest, ten camera records, a mosaic with non-zero coverage, a diagnostic overlay, and an evaluation report. Median seed-anchor reprojection error is below 5 pixels. Any unmet visual-stability criterion is reported explicitly rather than converted into a passing assertion.

- [ ] **Step 5: Commit the operator flow**

```bash
git add reconstruction/src/nfl_reconstruction/cli.py reconstruction/tests/test_cli.py reconstruction/README.md reconstruction/configs/clip3.json
git commit -m "docs: add reproducible Clip 3 reconstruction flow"
```

### Task 12: Final verification and acceptance report

**Files:**
- Modify only files required to correct failures found by the commands below.
- Create: `reconstruction/docs/clip3-mvp-results.md`

**Interfaces:**
- Consumes: completed implementation and `/tmp/nfl-blitz-clip3-smoke` outputs.
- Produces: a factual results report that separates automated passes, measured metrics, manual checks, and blocked detector criteria.

- [ ] **Step 1: Run the complete suite from a clean isolated environment**

```bash
cd reconstruction
python3.11 -m venv /tmp/nfl-blitz-reconstruction-verify
/tmp/nfl-blitz-reconstruction-verify/bin/python -m pip install -e '.[test]'
/tmp/nfl-blitz-reconstruction-verify/bin/python -m pytest -q
```

Expected: zero failures and zero unexpected warnings.

- [ ] **Step 2: Run the Clip 3 camera and mosaic workflow from scratch**

```bash
cd reconstruction
/tmp/nfl-blitz-reconstruction-verify/bin/nfl-reconstruct run \
  --config configs/clip3.json \
  --run-dir /tmp/nfl-blitz-clip3-final \
  --stages prepare,camera,mosaic \
  --frame-range 26:86
```

Expected: exit code zero with explicit counts for reliable, uncertain, and missing camera frames.

- [ ] **Step 3: Inspect generated artifacts and record measured results**

Read `/tmp/nfl-blitz-clip3-final/evaluation.json`, confirm output schemas validate, and visually inspect the overlay and minimap. Write `reconstruction/docs/clip3-mvp-results.md` with exact commands, package versions, frame range, reprojection metrics, confidence counts, mosaic coverage, visible field-swimming assessment, and the detector/tracking criteria that remain blocked pending player labels and weights.

- [ ] **Step 4: Run repository checks and inspect the final diff**

Run: `cd reconstruction && /tmp/nfl-blitz-reconstruction-verify/bin/python -m pytest -q`

Run: `git diff --check && git status --short && git log --oneline --decorate -15`

Expected: tests pass, diff check is clean, only intended files are changed, and no generated frames, weights, environments, or run outputs are tracked.

- [ ] **Step 5: Commit the verified results**

```bash
git add reconstruction/docs/clip3-mvp-results.md
git commit -m "docs: report Clip 3 reconstruction results"
```

## Self-Review Record

- **Spec coverage:** Every design component is owned by Tasks 1–11; Task 12 verifies and reports the acceptance state. Full detector accuracy remains explicitly data-gated rather than falsely claimed.
- **Placeholder scan:** The plan contains no deferred implementation placeholders. Detector training is an operator workflow with exact inputs and commands, not an automated unit-test dependency.
- **Type consistency:** Camera records, detection records, track records, manifests, and evaluation reports originate in Task 1 and are consumed unchanged by later tasks.
- **Review focus:** Degenerate anchors, corrupt/dimension-changing video, invalid camera views, mask mismatch, and low-confidence field projection each have an explicit test in their owning task.
