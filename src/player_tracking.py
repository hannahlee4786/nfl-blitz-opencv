# Multi-player tracking in field coordinates (yards).
# Tracking in field space cancels the camera motion, because the field
# homography already removes it.
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

INVALID_COST = 1e6
MAX_COLOR_DISTANCE = 441.67  # sqrt(3 * 255^2)


class Track:
    """One player: a constant-velocity Kalman filter on (x, y) yards."""

    def __init__(
        self,
        detection,
        process_noise,
        measurement_noise,
        velocity_decay,
        history_length,
    ):
        x, y = detection.field_xy

        self.id = None  # assigned once the track is confirmed
        self.state = np.array([x, y, 0.0, 0.0])
        self.covariance = np.diag([1.0, 1.0, 4.0, 4.0])

        self.hits = 1
        self.time_since_update = 0

        self.color = np.array(detection.color, dtype=np.float64)
        self.bbox = detection.bbox
        self.last_foot = detection.foot
        self.history = deque([(x, y)], maxlen=history_length)

        self.transition = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, velocity_decay, 0],
                [0, 0, 0, velocity_decay],
            ],
            dtype=np.float64,
        )

        self.process_noise = np.diag(
            [process_noise, process_noise,
             process_noise * 2, process_noise * 2]
        )

        self.measurement_noise = np.eye(2) * measurement_noise ** 2
        self.observation = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float64,
        )

    @property
    def position(self):
        return self.state[:2]

    @property
    def matched(self):
        """True if a detection updated this track on the current frame."""
        return self.time_since_update == 0

    def predict(self):
        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T
            + self.process_noise
        )
        self.time_since_update += 1

    def update(self, detection):
        measurement = np.asarray(detection.field_xy, dtype=np.float64)

        innovation = measurement - self.observation @ self.state

        innovation_covariance = (
            self.observation @ self.covariance @ self.observation.T
            + self.measurement_noise
        )

        gain = (
            self.covariance
            @ self.observation.T
            @ np.linalg.inv(innovation_covariance)
        )

        self.state = self.state + gain @ innovation

        self.covariance = (
            np.eye(4) - gain @ self.observation
        ) @ self.covariance

        self.hits += 1
        self.time_since_update = 0

        # Slowly follow the jersey color; it helps tell players apart.
        self.color = 0.8 * self.color + 0.2 * np.array(detection.color)

        self.bbox = detection.bbox
        self.last_foot = detection.foot
        self.history.append((self.state[0], self.state[1]))


class PlayerTracker:
    def __init__(
        self,
        max_age=30,
        min_hits=3,
        gate_yards=2.0,
        gate_growth=0.3,
        max_gate_yards=8.0,
        color_weight=0.5,
        process_noise=0.05,
        measurement_noise=0.7,
        velocity_decay=0.95,
        history_length=40,
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

        self.tracks = []
        self.next_id = 1

    def _confirm(self, track):
        if track.id is None and track.hits >= self.min_hits:
            track.id = self.next_id
            self.next_id += 1

    def _associate(self, detections):
        track_count = len(self.tracks)
        detection_count = len(detections)

        if track_count == 0 or detection_count == 0:
            return [], list(range(track_count)), list(range(detection_count))

        predicted = np.array([t.position for t in self.tracks])
        measured = np.array([d.field_xy for d in detections])

        distance = np.linalg.norm(
            predicted[:, None, :] - measured[None, :, :],
            axis=2,
        )

        # The longer a track has been missed, the farther it may have moved.
        gates = np.array(
            [
                min(
                    self.gate_yards + self.gate_growth * t.time_since_update,
                    self.max_gate_yards,
                )
                for t in self.tracks
            ]
        )[:, None]

        track_colors = np.array([t.color for t in self.tracks])
        detection_colors = np.array([d.color for d in detections])

        color_distance = np.linalg.norm(
            track_colors[:, None, :] - detection_colors[None, :, :],
            axis=2,
        ) / MAX_COLOR_DISTANCE

        cost = distance / gates + self.color_weight * color_distance
        cost[distance > gates] = INVALID_COST

        rows, columns = linear_sum_assignment(cost)

        matches = [
            (row, column)
            for row, column in zip(rows, columns)
            if cost[row, column] < INVALID_COST
        ]

        matched_tracks = {row for row, _ in matches}
        matched_detections = {column for _, column in matches}

        return (
            matches,
            [i for i in range(track_count) if i not in matched_tracks],
            [i for i in range(detection_count)
             if i not in matched_detections],
        )

    def update(self, detections):
        """
        detections: list of Detection with field_xy set. Pass an empty
        list to let every track coast (for example when the field
        homography is not trustworthy).

        Returns the confirmed tracks, matched or coasting.
        """

        for track in self.tracks:
            track.predict()

        matches, _, unmatched_detections = self._associate(detections)

        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            track.update(detections[detection_index])
            self._confirm(track)

        for detection_index in unmatched_detections:
            track = Track(
                detections[detection_index],
                self.process_noise,
                self.measurement_noise,
                self.velocity_decay,
                self.history_length,
            )

            self._confirm(track)
            self.tracks.append(track)

        kept = []

        for track in self.tracks:
            # A tentative track that is missed once is noise.
            if track.id is None and track.time_since_update > 0:
                continue

            if track.time_since_update > self.max_age:
                continue

            kept.append(track)

        self.tracks = kept

        return self.confirmed_tracks()

    def confirmed_tracks(self):
        return [t for t in self.tracks if t.id is not None]