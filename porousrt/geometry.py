"""Diffuse-interface geometries for pore-scale reactive models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .grid import Grid


def staggered_square_centers(grid: Grid) -> list[tuple[float, float]]:
    """Return five staggered columns containing thirteen square grains."""

    x_columns = np.linspace(0.19 * grid.length, 0.83 * grid.length, 5)
    y_three = (0.20 * grid.height, 0.50 * grid.height, 0.80 * grid.height)
    y_two = (0.35 * grid.height, 0.65 * grid.height)
    centers: list[tuple[float, float]] = []
    for column, x_center in enumerate(x_columns):
        y_centers = y_three if column % 2 == 0 else y_two
        centers.extend((float(x_center), float(y_center)) for y_center in y_centers)
    return centers


def _square_signed_distance(
    x: np.ndarray,
    y: np.ndarray,
    x_center: float,
    y_center: float,
    side: float,
) -> np.ndarray:
    """Signed distance to an axis-aligned square (negative inside)."""

    qx = np.abs(x - x_center) - 0.5 * side
    qy = np.abs(y - y_center) - 0.5 * side
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside


def build_staggered_square_porosity(
    grid: Grid,
    minimum_porosity: float = 1.0e-3,
    square_side: float | None = None,
    interface_width_cells: float = 1.15,
    centers: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Map staggered square grains to a diffuse porosity field.

    Fluid cells have porosity one and grain interiors use ``minimum_porosity``.
    A cubic smoothstep across the interface avoids grid-scale discontinuities.
    """

    if not 0.0 < minimum_porosity < 1.0:
        raise ValueError("minimum_porosity must be between zero and one")
    if interface_width_cells <= 0.0:
        raise ValueError("interface_width_cells must be positive")
    side = square_side if square_side is not None else 0.18 * grid.height
    if side <= 0.0:
        raise ValueError("square_side must be positive")

    grain_centers = (
        list(centers) if centers is not None else staggered_square_centers(grid)
    )
    width = interface_width_cells * max(grid.dx, grid.dy)
    porosity = np.ones((grid.ny, grid.nx), dtype=float)
    for x_center, y_center in grain_centers:
        distance = _square_signed_distance(
            grid.X, grid.Y, x_center, y_center, side
        )
        transition = np.clip(0.5 + distance / (2.0 * width), 0.0, 1.0)
        transition = transition * transition * (3.0 - 2.0 * transition)
        local_porosity = minimum_porosity + (1.0 - minimum_porosity) * transition
        porosity = np.minimum(porosity, local_porosity)
    return porosity
