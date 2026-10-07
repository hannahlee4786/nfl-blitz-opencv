# NFL Blitz Game-State Reconstruction Design

**Date:** 2026-10-06  
**Status:** Approved in conversation; pending written-spec review

## Objective

Build an experimental workflow that reconstructs an NFL Blitz gameplay clip in a fixed football-field coordinate system. The workflow must estimate the moving virtual camera, detect and track players, and project player ground positions onto a static top-down field.

The MVP targets short, continuous clips such as `Clip_3 Easiest QB Test.mp4`. It may require manual initialization and occasional manual correction. Handling broadcast cuts, replays, arbitrary games, and fully automatic initialization is outside the MVP.

## Isolation

The experiment will be independent of the existing tracker:

- It will live outside the current `src/` package.
- It will use a separate Python 3.11 environment and dependency file.
- It will have its own commands, tests, configuration, annotations, and output directory.
- It will not import or modify the existing OpenCV tracking implementation.
- Existing input videos may be read in place; generated artifacts will be written under the experiment's output directory.

The workflow may use FFmpeg or PyAV for video decoding, PyTorch and Kornia for differentiable geometry, a compact YOLO-family detector, and a small tracking implementation.

## Pipeline

### 1. Prepare the clip

The preparation stage will:

1. Decode the source clip into deterministically numbered frames.
2. Record video dimensions, frame rate, and frame count in a manifest.
3. Apply permanent exclusion masks for HUD and other screen-space graphics.
4. Collect a manually annotated anchor frame containing correspondences between visible image landmarks and known football-field coordinates.
5. Validate that the anchor correspondences produce a usable initial projection.

Preparation artifacts will be reusable by every later stage.

### 2. Recover the camera

The field is modeled as a fixed plane with known NFL dimensions and markings. Rather than chaining arbitrary frame-to-frame homographies, the workflow will estimate an interpretable virtual-camera state for every frame.

The per-frame state contains:

- Camera target position on the field
- Camera height or distance
- Yaw and pitch
- Focal length or zoom
- Optional roll constrained near zero

The initial state is fitted from the manually annotated anchor. Later states are optimized using a combination of:

- Alignment between rendered field markings and observed yard lines, hash marks, sidelines, and end-zone boundaries
- Alignment against stable field texture accumulated into a field mosaic
- Reprojection error at manually annotated landmarks
- Temporal smoothness of camera parameters
- Penalties for physically implausible camera changes

HUD regions and moving-object masks receive zero weight in the camera objective.

Camera states are optimized in overlapping windows of approximately 30–60 frames. Windows are anchored to high-confidence frames rather than being propagated indefinitely. This limits cumulative drift while preserving continuous camera motion.

When confidence falls below a configured threshold, the workflow retains the last reliable estimate, marks the affected frames as uncertain, and accepts a manual correction anchor. It must never silently present a low-confidence estimate as reliable.

Each frame exports both the camera parameters and the corresponding field-to-image homography. Either representation can therefore be used by visualization and player projection stages.

### 3. Build a stable field representation

Frames with confident camera estimates contribute their static, unmasked pixels to a top-down field mosaic. Robust temporal aggregation suppresses players, shadows, indicators, and transient effects.

The mosaic serves two purposes:

- It provides denser alignment evidence than painted field markings alone.
- It creates a visual static-field reconstruction for inspection.

The known geometric field template remains the authoritative coordinate system. The mosaic supplements it but does not define field dimensions.

### 4. Detect and track players

Player detection will be trained for NFL Blitz imagery rather than relying on models trained only on real sports footage. The MVP annotation set should contain approximately 50–100 representative frames and record:

- Player bounding box
- Team class
- Ground-contact point, approximated by the midpoint between the feet
- Exclusion labels for non-player overlays such as the ball-carrier marker when necessary

A compact detector produces bounding boxes, team classes, ground-contact estimates, and confidence scores. A ByteTrack-style association stage links detections across frames.

Association uses both image-space motion and projected field-space motion. Field-space evidence is ignored whenever camera confidence is insufficient. Short occlusions may preserve an identity; extended uncertainty creates a new track rather than forcing an incorrect match.

For each accepted detection, the inverse frame homography maps the ground-contact point into static field coordinates.

### 5. Visualize and evaluate

The workflow produces:

- A static top-down field image or video with player positions and trajectories
- A diagnostic source-video overlay with projected field markings, player detections, track IDs, and confidence indicators
- A per-frame camera-state file
- A per-detection and per-track tabular file
- A machine-readable evaluation report

The primary player record contains:

```text
frame, track_id, team, confidence,
screen_x, screen_y, field_x, field_y,
camera_confidence
```

## Component boundaries

The experiment is divided into replaceable components:

- **Video source:** deterministic frame decoding and metadata
- **Annotation store:** anchor landmarks, HUD masks, player labels, and correction anchors
- **Field model:** NFL geometry, markings, and coordinate conversions
- **Camera model:** conversion between camera parameters and homographies
- **Camera optimizer:** per-window fitting, confidence, and correction handling
- **Mosaic builder:** robust accumulation of static field texture
- **Player detector:** game-specific detections and team classification
- **Track associator:** screen- and field-space identity continuity
- **Projection layer:** image ground points to field coordinates
- **Visualization:** diagnostic overlay and static minimap
- **Evaluation:** camera, detection, and tracking metrics

Components exchange versioned files with documented schemas. A stage can be rerun without rerunning earlier successful stages when its inputs have not changed.

## Error handling and observability

Every processing stage will fail with an actionable message when required inputs are missing or malformed. Long-running stages will write progress and timing information.

Camera and player results carry explicit confidence values. The visualizations will distinguish reliable, uncertain, and missing results. Failed or uncertain frames remain in output records so gaps cannot be mistaken for successful processing.

Model checkpoints, configuration, input hashes, and software versions will be recorded with each run to make results reproducible.

## Testing strategy

Development will follow test-driven implementation. The test suite will include:

- Unit tests for field coordinates and projection math
- Synthetic camera tests with known parameters and homographies
- Optimizer tests that recover perturbed synthetic camera states
- Manifest, annotation, and output-schema validation tests
- Mosaic tests that reject moving foreground pixels
- Tracking tests covering short occlusion, team mismatch, and low camera confidence
- A deterministic Clip 3 smoke test over a small frame range
- Golden-image or numeric regression tests for selected annotated frames

The full player detector training run is not part of the fast unit suite. Tests will use small fixtures or recorded detector outputs where model inference is not the behavior under test.

## MVP acceptance criteria

The MVP is accepted when:

- Median landmark reprojection error is below 5 pixels on the manually checked evaluation frames.
- The projected static field shows no conspicuous swimming in the diagnostic overlay.
- At least 90% of clearly visible players are detected on the annotated evaluation subset.
- Projected player positions remain within valid field bounds unless explicitly flagged.
- Track identities survive ordinary short player overlaps in the evaluation clip.
- Camera loss, correction frames, missing detections, and fragmented tracks are visible in both records and diagnostic output.
- A new user can reproduce the Clip 3 smoke test from documented commands in the isolated environment.

## Non-goals

The MVP does not include:

- SportsLabKit integration
- Reuse or replacement of the existing `src/` tracker
- Fully automatic initial calibration
- Broadcast cut or replay detection
- Generalization to real football broadcasts or other games
- Jersey-number recognition
- Ball tracking
- Real-time processing
- Guaranteed identity recovery after long off-screen absences

## Risks and mitigations

### Sparse field evidence

Some frames may contain few distinctive markings. Windowed temporal optimization, texture mosaics, and manual correction anchors reduce this risk.

### Stylized and animated field texture

Texture may contain lighting changes or transient effects. Only high-confidence camera frames contribute to the mosaic, and robust aggregation rejects inconsistent pixels.

### Player and HUD occlusion

Permanent HUD masks and dynamic player masks exclude foreground pixels from camera fitting.

### Insufficient training labels

The detector starts with a deliberately small annotation set. Evaluation will determine whether more labels are needed; the camera workflow remains independently usable.

### Optimization ambiguity

The known field geometry, constrained virtual-camera parameterization, anchor frames, and temporal priors reduce the number of equivalent but incorrect solutions.

## Delivery sequence

Implementation will proceed in vertical slices:

1. Isolated package, configuration, frame manifest, and field model
2. Synthetic camera model and projection verification
3. Manual anchor format and single-frame camera fitting
4. Windowed camera recovery and diagnostic overlay
5. Field mosaic
6. Player annotation format, detector interface, and recorded-output path
7. Player association and field projection
8. End-to-end Clip 3 smoke test and evaluation report

Each slice must have passing tests and a usable diagnostic artifact before the next slice begins.
