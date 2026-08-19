from __future__ import annotations

import unittest

import numpy as np

from cases.carnallite_replacement_binary.geometry import (
    build_gstools_binary_porosity,
)
from cases.carnallite_replacement_binary.model import (
    BinaryCarnalliteBrineConfig,
)
from porousrt.grid import Grid


class GSToolsBinaryGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = Grid(nx=48, ny=20, length=1.2e-3, height=5.0e-4)
        self.options = {
            "minimum_porosity": 1.0e-3,
            "solid_fraction": 13.0 / 60.0,
            "correlation_length_x": 1.5e-4,
            "correlation_length_y": 7.5e-5,
            "random_seed": 2025,
            "boundary_buffer_cells": 2,
        }

    def test_field_is_binary_reproducible_and_has_requested_fraction(self) -> None:
        first = build_gstools_binary_porosity(self.grid, **self.options)
        second = build_gstools_binary_porosity(self.grid, **self.options)

        self.assertEqual(first.shape, (self.grid.ny, self.grid.nx))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.unique(first), [1.0e-3, 1.0])

        solid_cells = int(np.count_nonzero(first == 1.0e-3))
        expected = round(self.options["solid_fraction"] * first.size)
        self.assertEqual(solid_cells, expected)

    def test_boundary_buffer_is_fluid_and_seed_changes_interior(self) -> None:
        first = build_gstools_binary_porosity(self.grid, **self.options)
        changed = dict(self.options, random_seed=7)
        second = build_gstools_binary_porosity(self.grid, **changed)

        buffer = self.options["boundary_buffer_cells"]
        self.assertTrue(np.all(first[:buffer, :] == 1.0))
        self.assertTrue(np.all(first[-buffer:, :] == 1.0))
        self.assertTrue(np.all(first[:, :buffer] == 1.0))
        self.assertTrue(np.all(first[:, -buffer:] == 1.0))
        self.assertFalse(np.array_equal(first, second))

    def test_case_configuration_dispatches_to_binary_geometry(self) -> None:
        config = BinaryCarnalliteBrineConfig(
            nx=self.grid.nx,
            ny=self.grid.ny,
            random_seed=11,
            make_video=False,
            make_gif=False,
        )
        config.validate()
        self.assertEqual(config.frame_interval, 1)
        porosity = config.build_initial_porosity()
        np.testing.assert_array_equal(np.unique(porosity), [1.0e-3, 1.0])


if __name__ == "__main__":
    unittest.main()
