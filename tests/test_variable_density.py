from __future__ import annotations

import unittest

import numpy as np

from porousrt import Grid, build_staggered_square_porosity
from porousrt.flow import solve_variable_density_brinkman_flow
from porousrt.boundary_flow import (
    RectangularBoundaryFlowCondition,
    RectangularFlowBoundaryConditions,
    solve_rectangular_variable_density_brinkman_flow,
)


class VariableDensityNumericsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = Grid(nx=32, ny=16, length=6.0e-3, height=3.0e-3)
        self.porosity = build_staggered_square_porosity(
            self.grid,
            minimum_porosity=1.0e-3,
            square_side=4.3e-4,
            interface_width_cells=1.15,
        )

    def _flow(
        self, porosity: np.ndarray, density: np.ndarray, viscosity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        return solve_variable_density_brinkman_flow(
            porosity,
            density,
            viscosity,
            self.grid,
            inlet_velocity_m_s=2.0e-5,
            inlet_density_kg_m3=float(np.mean(density[:, 0])),
            minimum_porosity=1.0e-3,
            permeability_parameter_m2=1.0e-15,
            drag_cap_nondimensional=2.0e7,
            gravity_m_s2=(0.0, -9.80665),
        )

    def test_uniform_density_mass_balance_and_solid_penalty(self) -> None:
        density = np.full_like(self.porosity, 1197.0)
        viscosity = np.full_like(self.porosity, 1.76e-3)
        _u, _v, _p, diagnostics = self._flow(
            self.porosity, density, viscosity
        )
        self.assertLess(diagnostics["max_relative_mass_balance_error"], 1.0e-7)
        self.assertLess(
            diagnostics["mean_solid_speed_m_s"],
            1.0e-2 * diagnostics["mean_fluid_speed_m_s"],
        )

    def test_horizontal_density_gradient_drives_vertical_recircultion(self) -> None:
        porosity = np.ones((self.grid.ny, self.grid.nx))
        density = 1137.0 + 60.0 * self.grid.X / self.grid.length
        viscosity = 1.36e-3 + 0.40e-3 * self.grid.X / self.grid.length
        _u, _v, _p, diagnostics = self._flow(porosity, density, viscosity)
        self.assertGreater(diagnostics["max_vertical_speed_m_s"], 1.0e-10)
        self.assertLess(diagnostics["max_relative_mass_balance_error"], 1.0e-7)

    def test_finite_width_inlet_profile_preserves_total_injection(self) -> None:
        porosity = np.ones((self.grid.ny, self.grid.nx))
        density = np.full_like(porosity, 1197.0)
        viscosity = np.full_like(porosity, 1.76e-3)
        profile = np.exp(
            -0.5 * ((self.grid.y / self.grid.height - 0.45) / 0.08) ** 2
        )
        u, _v, _p, diagnostics = solve_variable_density_brinkman_flow(
            porosity,
            density,
            viscosity,
            self.grid,
            inlet_velocity_m_s=2.0e-5,
            inlet_density_kg_m3=1197.0,
            inlet_velocity_profile=profile,
            minimum_porosity=1.0e-3,
            permeability_parameter_m2=1.0e-15,
            drag_cap_nondimensional=2.0e7,
            gravity_m_s2=(0.0, 0.0),
        )
        np.testing.assert_allclose(u[:, 0], 2.0e-5 * profile / np.mean(profile))
        self.assertAlmostEqual(float(np.mean(u[:, 0])), 2.0e-5, places=14)
        self.assertGreater(diagnostics["maximum_inlet_velocity_m_s"], 2.0e-5)
        self.assertLess(diagnostics["max_relative_mass_balance_error"], 1.0e-7)

    def test_two_side_inlets_are_balanced_by_bottom_outlet(self) -> None:
        grid = Grid(nx=18, ny=12, length=1.8e-3, height=1.2e-3)
        porosity = np.ones((grid.ny, grid.nx))
        density = np.full_like(porosity, 1000.0)
        viscosity = np.full_like(porosity, 1.0e-3)
        speed = 2.0e-5
        boundaries = RectangularFlowBoundaryConditions(
            left=RectangularBoundaryFlowCondition(
                kind="velocity",
                normal_velocity_m_s=-speed,
                inflow_density_kg_m3=1000.0,
            ),
            right=RectangularBoundaryFlowCondition(
                kind="velocity",
                normal_velocity_m_s=-speed,
                inflow_density_kg_m3=1000.0,
            ),
            bottom=RectangularBoundaryFlowCondition(
                kind="mass_compatible_outlet"
            ),
        )

        u, v, _pressure, diagnostics = (
            solve_rectangular_variable_density_brinkman_flow(
                porosity,
                density,
                viscosity,
                grid,
                boundaries=boundaries,
                minimum_porosity=1.0e-3,
                permeability_parameter_m2=1.0e-15,
                drag_cap_nondimensional=2.0e7,
                gravity_m_s2=(0.0, 0.0),
            )
        )

        np.testing.assert_allclose(u[:, 0], speed)
        np.testing.assert_allclose(u[:, -1], -speed)
        self.assertTrue(np.all(v[0, :] < 0.0))
        np.testing.assert_allclose(v[-1, :], 0.0)
        expected_outlet_speed = 2.0 * grid.height / grid.length * speed
        self.assertAlmostEqual(
            diagnostics["mass_compatible_outlet_normal_velocity_m_s"],
            expected_outlet_speed,
            places=13,
        )
        self.assertLess(diagnostics["max_relative_mass_balance_error"], 1.0e-7)
        self.assertAlmostEqual(
            diagnostics["net_outward_boundary_mass_flux_kg_m_s"],
            diagnostics["integrated_mass_source_kg_m_s"],
            places=13,
        )

if __name__ == "__main__":
    unittest.main()
