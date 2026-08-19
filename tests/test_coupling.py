from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from porousrt.coupling import SequentialCoupler


@dataclass
class _ChemicalState:
    concentrations: np.ndarray
    porosity: np.ndarray
    density_kg_m3: np.ndarray
    viscosity_pa_s: np.ndarray
    solid_mass_release_rate_kg_m3_s: np.ndarray


class _Chemistry:
    def __init__(self) -> None:
        shape = (1, 2)
        self.initial_state = _ChemicalState(
            concentrations=np.zeros((1, *shape)),
            porosity=np.full(shape, 0.5),
            density_kg_m3=np.full(shape, 1000.0),
            viscosity_pa_s=np.full(shape, 1.0e-3),
            solid_mass_release_rate_kg_m3_s=np.zeros(shape),
        )
        self.inlet_concentrations = np.array([1.0])

    def react(
        self, concentrations: np.ndarray, porosity: np.ndarray, dt: float
    ) -> _ChemicalState:
        new_porosity = porosity + 0.05
        return _ChemicalState(
            concentrations=concentrations,
            porosity=new_porosity,
            density_kg_m3=np.full_like(new_porosity, 1000.0),
            viscosity_pa_s=np.full_like(new_porosity, 1.0e-3),
            solid_mass_release_rate_kg_m3_s=np.full_like(new_porosity, 2.0),
        )


class _Transport:
    def advance(
        self,
        concentrations: np.ndarray,
        porosity: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        *,
        dt: float,
        molecular_diffusivity_m2_s: float,
        inlet_concentrations: np.ndarray,
    ) -> np.ndarray:
        return concentrations + dt * inlet_concentrations[:, None, None]


class _Flow:
    def __init__(self) -> None:
        self.sources: list[np.ndarray] = []

    def solve(
        self,
        porosity: np.ndarray,
        density_kg_m3: np.ndarray,
        viscosity_pa_s: np.ndarray,
        mass_balance_source_kg_m3_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        self.sources.append(mass_balance_source_kg_m3_s.copy())
        ny, nx = porosity.shape
        return (
            np.ones((ny, nx + 1)),
            np.zeros((ny + 1, nx)),
            np.zeros((ny, nx)),
            {"maximum_relative_mass_balance_error": 0.0},
        )


class SequentialCouplerTests(unittest.TestCase):
    def _coupler(self, flow_interval: int = 2) -> tuple[SequentialCoupler, _Flow]:
        flow = _Flow()
        coupler = SequentialCoupler(
            chemistry=_Chemistry(),
            transport=_Transport(),
            flow=flow,
            dt=1.0,
            molecular_diffusivity_m2_s=1.0e-9,
            flow_interval=flow_interval,
        )
        return coupler, flow

    def test_multirate_cadence_and_mass_source(self) -> None:
        coupler, flow = self._coupler(flow_interval=2)
        state = coupler.initialize()
        state = coupler.advance(state, step_index=1)
        self.assertEqual(coupler.flow_solves, 1)
        self.assertTrue(coupler.has_pending_flow_feedback)

        state = coupler.advance(state, step_index=2)
        self.assertEqual(coupler.flow_solves, 2)
        self.assertFalse(coupler.has_pending_flow_feedback)
        self.assertEqual(state.time, 2.0)
        np.testing.assert_allclose(state.chemistry.concentrations, 2.0)
        # Integrated mineral release is 4 kg/m3; bulk fluid mass increased by
        # 100 kg/m3 over two seconds, hence (4 - 100) / 2 = -48 kg/m3/s.
        np.testing.assert_allclose(flow.sources[-1], -48.0)

    def test_refresh_flow_commits_pending_feedback_without_advancing_time(self) -> None:
        coupler, _flow = self._coupler(flow_interval=5)
        state = coupler.initialize()
        state = coupler.advance(state, step_index=1)
        refreshed = coupler.refresh_flow(state)
        self.assertEqual(refreshed.time, 1.0)
        self.assertEqual(coupler.flow_solves, 2)
        self.assertFalse(coupler.has_pending_flow_feedback)
        np.testing.assert_allclose(refreshed.mass_source_kg_m3_s, -48.0)


if __name__ == "__main__":
    unittest.main()
