"""Regulation NFL field geometry in yard coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class NFLField:
    length_yards: float
    width_yards: float
    upper_hash_y: float
    lower_hash_y: float

    @classmethod
    def standard(cls) -> "NFLField":
        return cls(
            length_yards=120.0,
            width_yards=53.333,
            upper_hash_y=23.58,
            lower_hash_y=29.75,
        )

    def contains(self, points: ArrayLike) -> NDArray[np.bool_]:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)
        if coordinates.shape[-1] != 2:
            raise ValueError("Field points must have shape (..., 2).")
        return (
            (coordinates[:, 0] >= 0.0)
            & (coordinates[:, 0] <= self.length_yards)
            & (coordinates[:, 1] >= 0.0)
            & (coordinates[:, 1] <= self.width_yards)
        )

    def marking_segments(self) -> tuple[Segment, ...]:
        verticals = tuple(
            ((float(x), 0.0), (float(x), self.width_yards))
            for x in range(10, 111, 10)
        )
        return (
            *verticals,
            ((0.0, 0.0), (self.length_yards, 0.0)),
            (
                (0.0, self.width_yards),
                (self.length_yards, self.width_yards),
            ),
            ((10.0, self.upper_hash_y), (110.0, self.upper_hash_y)),
            ((10.0, self.lower_hash_y), (110.0, self.lower_hash_y)),
        )
