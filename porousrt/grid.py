"""Uniform Cartesian grids used by the pore-scale models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Uniform two-dimensional Cartesian grid."""

    nx: int
    ny: int
    length: float
    height: float

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("grid cell counts must be positive")
        if self.length <= 0.0 or self.height <= 0.0:
            raise ValueError("grid dimensions must be positive")

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dy(self) -> float:
        return self.height / self.ny

    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y(self) -> np.ndarray:
        return (np.arange(self.ny) + 0.5) * self.dy

    @property
    def X(self) -> np.ndarray:
        return np.meshgrid(self.x, self.y)[0]

    @property
    def Y(self) -> np.ndarray:
        return np.meshgrid(self.x, self.y)[1]
