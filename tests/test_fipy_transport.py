import unittest

import numpy as np

from porousrt import Grid
from porousrt.transport import (
    ComponentBoundaryConditions,
    ConservativeTransportSolver,
)


class FiPyTransportTests(unittest.TestCase):
    def test_uniform_solution_is_preserved(self) -> None:
        grid = Grid(nx=8, ny=4, length=8.0e-4, height=4.0e-4)
        solver = ConservativeTransportSolver(grid)
        concentrations = np.empty((3, grid.ny, grid.nx))
        concentrations[0] = 1.2
        concentrations[1] = 0.3
        concentrations[2] = -2.0e-8
        porosity = np.full((grid.ny, grid.nx), 0.73)
        u = np.full((grid.ny, grid.nx + 1), 2.0e-5)
        v = np.zeros((grid.ny + 1, grid.nx))

        updated = solver.advance(
            concentrations,
            porosity,
            u,
            v,
            dt=0.5,
            molecular_diffusivity_m2_s=1.0e-9,
            inlet_concentrations=concentrations[:, 0, 0],
        )

        np.testing.assert_allclose(updated, concentrations, rtol=0.0, atol=2.0e-12)

    def test_upwind_step_remains_bounded(self) -> None:
        grid = Grid(nx=12, ny=3, length=1.2e-3, height=3.0e-4)
        solver = ConservativeTransportSolver(grid)
        concentrations = np.zeros((1, grid.ny, grid.nx))
        porosity = np.ones((grid.ny, grid.nx))
        u = np.full((grid.ny, grid.nx + 1), 4.0e-5)
        v = np.zeros((grid.ny + 1, grid.nx))

        updated = solver.advance(
            concentrations,
            porosity,
            u,
            v,
            dt=1.0,
            molecular_diffusivity_m2_s=1.0e-9,
            inlet_concentrations=np.array([1.0]),
        )

        self.assertGreater(float(np.max(updated)), 0.0)
        self.assertGreaterEqual(float(np.min(updated)), -1.0e-12)
        self.assertLessEqual(float(np.max(updated)), 1.0 + 1.0e-12)
        self.assertGreater(float(np.mean(updated[:, :, 0])), float(np.mean(updated[:, :, -1])))

    def test_distinct_left_and_right_inlet_compositions(self) -> None:
        grid = Grid(nx=10, ny=5, length=1.0e-3, height=5.0e-4)
        solver = ConservativeTransportSolver(
            grid,
            inlet_sides=("left", "right"),
            outlet_sides=("bottom",),
        )
        concentrations = np.zeros((2, grid.ny, grid.nx))
        porosity = np.ones((grid.ny, grid.nx))
        u = np.zeros((grid.ny, grid.nx + 1))
        v = np.zeros((grid.ny + 1, grid.nx))

        updated = solver.advance(
            concentrations,
            porosity,
            u,
            v,
            dt=2.0,
            molecular_diffusivity_m2_s=1.0e-9,
            inlet_concentrations=ComponentBoundaryConditions(
                left=np.array([1.0, 0.0]),
                right=np.array([0.0, 2.0]),
            ),
        )

        self.assertGreater(float(np.mean(updated[0, :, 0])), 0.0)
        self.assertGreater(
            float(np.mean(updated[0, :, 0])),
            float(np.mean(updated[0, :, -1])),
        )
        self.assertGreater(float(np.mean(updated[1, :, -1])), 0.0)
        self.assertGreater(
            float(np.mean(updated[1, :, -1])),
            float(np.mean(updated[1, :, 0])),
        )


if __name__ == "__main__":
    unittest.main()
