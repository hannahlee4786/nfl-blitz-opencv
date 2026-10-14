
"""Persistent NFL Blitz player tracking with controlled track creation."""

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


INVALID_COST = 1e6
MAX_COLOR_DISTANCE = np.sqrt(3 * 255 ** 2)
TEAMS = ("red", "blue")


def bbox_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class Track:
    """One player tracked in field coordinates with a Kalman filter."""

    def __init__(
        self,
        detection,
        process_noise,
        measurement_noise,
        velocity_decay,
        history_length,
        team_decay=0.9,
        team_switch_ratio=1.5,
    ):
        x, y = detection.field_xy

        self.id = None
        self.state = np.array(
            [x, y, 0.0, 0.0], dtype=np.float64
        )
        self.covariance = np.diag([1.0, 1.0, 4.0, 4.0])

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.quality_total = float(detection.quality)
        self.updates = 1

        self.color = np.asarray(
            detection.color, dtype=np.float64
        )
        self.bbox = tuple(detection.bbox)
        self.last_foot = detection.foot

        self.history = deque(
            [(float(x), float(y))],
            maxlen=history_length,
        )

        self.team = None
        self.team_scores = np.zeros(2)
        self.team_decay = team_decay
        self.team_switch_ratio = team_switch_ratio
        self._vote_team(detection)

        self.transition = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, velocity_decay, 0],
            [0, 0, 0, velocity_decay],
        ], dtype=np.float64)

        self.process_noise = np.diag([
            process_noise,
            process_noise,
            process_noise * 2,
            process_noise * 2,
        ])

        self.measurement_noise = (
            np.eye(2) * measurement_noise ** 2
        )

        self.observation = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

    @property
    def position(self):
        return self.state[:2]

    @property
    def matched(self):
        return self.time_since_update == 0

    @property
    def mean_quality(self):
        return self.quality_total / max(self.updates, 1)

    def _vote_team(self, detection):
        if detection.team not in TEAMS:
            return

        self.team_scores *= self.team_decay
        self.team_scores += [
            detection.red_fraction,
            detection.blue_fraction,
        ]

        best = int(np.argmax(self.team_scores))
        best_team = TEAMS[best]

        if self.team is None:
            self.team = best_team
        elif best_team != self.team:
            current = self.team_scores[TEAMS.index(self.team)]

            if (
                self.team_scores[best]
                > self.team_switch_ratio * current
            ):
                self.team = best_team

    def predict(self):
        self.state = self.transition @ self.state

        self.covariance = (
            self.transition
            @ self.covariance
            @ self.transition.T
            + self.process_noise
        )

        self.time_since_update += 1
        self.age += 1

    def update(self, detection):
        measurement = np.asarray(
            detection.field_xy, dtype=np.float64
        )

        innovation = (
            measurement - self.observation @ self.state
        )

        s = (
            self.observation
            @ self.covariance
            @ self.observation.T
            + self.measurement_noise
        )

        gain = (
            self.covariance
            @ self.observation.T
            @ np.linalg.inv(s)
        )

        self.state = self.state + gain @ innovation

        identity = np.eye(4)
        residual = identity - gain @ self.observation

        self.covariance = (
            residual @ self.covariance @ residual.T
            + gain @ self.measurement_noise @ gain.T
        )

        self.hits += 1
        self.time_since_update = 0
        self.quality_total += float(detection.quality)
        self.updates += 1

        self.color = (
            0.85 * self.color
            + 0.15 * np.asarray(detection.color)
        )

        self._vote_team(detection)

        self.bbox = tuple(detection.bbox)
        self.last_foot = detection.foot
        self.history.append(tuple(self.position))


class PlayerTracker:
    """Kalman/Hungarian tracker with edge-only new-track spawning."""

    def __init__(
        self,
        max_age=45,
        min_hits=5,
        gate_yards=3.0,
        gate_growth=0.4,
        max_gate_yards=8.0,
        color_weight=0.35,
        process_noise=0.05,
        measurement_noise=0.7,
        velocity_decay=0.95,
        history_length=40,
        min_quality=0.2,
        max_players=14,
        team_decay=0.9,
        team_switch_ratio=1.5,
        team_mismatch_cost=0.75,
        iou_weight=0.2,
        max_match_cost=1.5,
        tentative_max_age=5,
        min_detection_quality=0.10,
        strict_team_matching=True,
        entry_margin_fraction=0.08,
        initialization_frames=5,
        max_initial_players=14,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.gate_yards = gate_yards
        self.gate_growth = gate_growth
        self.max_gate_yards = max_gate_yards
        self.color_weight = color_weight
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.velocity_decay = velocity_decay
        self.history_length = history_length
        self.min_quality = min_quality
        self.max_players = max_players

        self.team_decay = team_decay
        self.team_switch_ratio = team_switch_ratio
        self.team_mismatch_cost = team_mismatch_cost
        self.iou_weight = iou_weight
        self.max_match_cost = max_match_cost
        self.tentative_max_age = tentative_max_age
        self.min_detection_quality = min_detection_quality
        self.strict_team_matching = strict_team_matching

        # Controlled initialization and spawning.
        self.entry_margin_fraction = entry_margin_fraction
        self.initialization_frames = initialization_frames
        self.max_initial_players = max_initial_players

        self.frame_count = 0
        self.frame_width = None
        self.frame_height = None

        self.tracks = []
        self.next_id = 1

    def _confirm(self, track):
        if (
            track.id is None
            and track.hits >= self.min_hits
            and track.mean_quality >= self.min_quality
        ):
            track.id = self.next_id
            self.next_id += 1

    
    def _can_create_track(self, detection):
        """
        Initialize players during the startup window.

        After initialization, unmatched detections cannot
        create new player identities.
        """
        return self.frame_count <= self.initialization_frames


    def _associate(self, detections):
        nt = len(self.tracks)
        nd = len(detections)

        if nt == 0 or nd == 0:
            return (
                [],
                list(range(nt)),
                list(range(nd)),
            )

        cost = np.full(
            (nt, nd),
            INVALID_COST,
            dtype=np.float64,
        )

        for i, track in enumerate(self.tracks):
            gate = min(
                self.gate_yards
                + self.gate_growth * track.time_since_update,
                self.max_gate_yards,
            )

            for j, detection in enumerate(detections):
                distance = np.linalg.norm(
                    track.position - detection.field_xy
                )

                if distance > gate:
                    continue

                mismatch = (
                    track.team is not None
                    and detection.team is not None
                    and track.team != detection.team
                )

                if (
                    mismatch
                    and self.strict_team_matching
                    and detection.jersey_strong
                ):
                    continue

                color_distance = (
                    np.linalg.norm(
                        track.color
                        - np.asarray(detection.color)
                    )
                    / MAX_COLOR_DISTANCE
                )

                overlap = bbox_iou(
                    track.bbox,
                    detection.bbox,
                )

                iou_factor = (
                    self.iou_weight
                    if track.time_since_update <= 1
                    else 0.0
                )

                mismatch_penalty = (
                    self.team_mismatch_cost
                    if mismatch
                    else 0.0
                )

                cost[i, j] = (
                    distance / max(gate, 1e-6)
                    + self.color_weight * color_distance
                    + iou_factor * (1.0 - overlap)
                    + mismatch_penalty
                )

        allowed = cost < min(
            INVALID_COST,
            self.max_match_cost,
        )

        # Dummy columns allow tracks to remain unmatched.
        assignment_cost = np.full(
            (nt, nd + nt),
            max(self.max_match_cost, 1e-6),
            dtype=np.float64,
        )

        assignment_cost[:, :nd] = np.where(
            allowed,
            cost,
            INVALID_COST,
        )

        rows, columns = linear_sum_assignment(
            assignment_cost
        )

        matches = [
            (int(i), int(j))
            for i, j in zip(rows, columns)
            if j < nd and allowed[i, j]
        ]

        used_tracks = {i for i, _ in matches}
        used_detections = {j for _, j in matches}

        return (
            matches,
            [
                i for i in range(nt)
                if i not in used_tracks
            ],
            [
                j for j in range(nd)
                if j not in used_detections
            ],
        )

    def update(self, detections, frame_shape=None):
        """
        Advance the tracker one frame.

        frame_shape is required for edge-based spawning.
        """
        self.frame_count += 1

        if frame_shape is not None:
            self.frame_height, self.frame_width = (
                frame_shape[:2]
            )

        detections = [
            d for d in detections
            if d.field_xy is not None
            and np.all(np.isfinite(d.field_xy))
            and float(d.quality) >= self.min_detection_quality
        ]

        for track in self.tracks:
            track.predict()

        matches, _, unmatched = self._associate(
            detections
        )

        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            track.update(detections[detection_index])
            self._confirm(track)

        # Prefer high-quality unmatched detections when
        # choosing initial player candidates.
        unmatched = sorted(
            unmatched,
            key=lambda index: detections[index].quality,
            reverse=True,
        )

        for detection_index in unmatched:
            detection = detections[detection_index]

            if not detection.jersey_strong:
                continue

            if not self._can_create_track(detection):
                continue

            # Do not spawn another track around an existing player.
            if self._overlaps_existing_track(detection):
                continue

            # Limit tentative and confirmed tracks together.
            if self.max_players and len(self.tracks) >= self.max_players:
                continue

            if (
                self.frame_count <= self.initialization_frames
                and len(self.tracks) >= self.max_initial_players
            ):
                continue

            track = Track(
                detection,
                self.process_noise,
                self.measurement_noise,
                self.velocity_decay,
                self.history_length,
                self.team_decay,
                self.team_switch_ratio,
            )

            self._confirm(track)
            self.tracks.append(track)

        # Confirmed tracks persist through occlusions.
        # Tentative tracks have a shorter persistence period.
        self.tracks = [
            track for track in self.tracks
            if (
                track.time_since_update <= self.max_age
                if track.id is not None
                else track.time_since_update
                <= self.tentative_max_age
            )
        ]

        return self.confirmed_tracks()

    def confirmed_tracks(self):
        confirmed = [
            track for track in self.tracks
            if track.id is not None
        ]

        if self.max_players and len(confirmed) > self.max_players:
            confirmed.sort(
                key=lambda t: (
                    t.matched,
                    t.hits * t.mean_quality,
                ),
                reverse=True,
            )
            confirmed = confirmed[:self.max_players]

        return sorted(
            confirmed,
            key=lambda track: track.id,
        )

    
    def _overlaps_existing_track(self, detection):
        """
        Prevent a new track from being created when its detection
        likely belongs to an existing player.

        This only controls track creation. Existing tracks can
        still match and update normally.
        """
        for track in self.tracks:
            # Do not compare players from different known teams.
            if (
                track.team is not None
                and detection.team is not None
                and track.team != detection.team
            ):
                continue

            # Compare bounding box overlap.
            overlap = bbox_iou(
                track.bbox,
                detection.bbox,
            )

            if overlap >= 0.20:
                return True

            # Also compare ground positions in field coordinates.
            if detection.field_xy is not None:
                distance = np.linalg.norm(
                    track.position - detection.field_xy
                )

                if distance < 1.0:
                    return True

        return False

