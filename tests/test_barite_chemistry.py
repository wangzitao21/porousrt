from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cases.two_inlet_barite_precipitation.chemistry import (
    BariteChemistryConfig,
    BaritePhreeqcRM,
)


@unittest.skipUnless(
    importlib.util.find_spec("phreeqcrm") is not None,
    "phreeqcrm is not installed in this Python environment",
)
class BaritePhreeqcRMTests(unittest.TestCase):
    def _mixed_inlet(self, chemistry: BaritePhreeqcRM) -> np.ndarray:
        mixed = 0.5 * (
            chemistry.left_inlet_concentrations + chemistry.right_inlet_concentrations
        )
        return mixed[:, None, None]

    def test_equilibrium_mode_precipitates_barite_from_mixed_inlets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with BaritePhreeqcRM(
                np.ones((1, 1)),
                BariteChemistryConfig(thread_count=1),
                Path(directory),
            ) as chemistry:
                initial = chemistry.initial_state
                reacted = chemistry.react(
                    self._mixed_inlet(chemistry), initial.porosity, dt=1.0
                )

                self.assertGreater(float(reacted.barite_moles[0, 0]), 0.0)
                self.assertLess(float(reacted.porosity[0, 0]), 1.0)
                self.assertLessEqual(
                    abs(float(reacted.barite_saturation_index[0, 0])), 1.0e-7
                )
                self.assertLess(
                    max(reacted.element_relative_residuals.values()), 1.0e-8
                )

    def test_kinetic_mode_is_time_step_dependent_and_not_instant_equilibrium(
        self,
    ) -> None:
        equilibrium_moles = 0.0
        kinetic_moles = 0.0
        kinetic_si = 0.0
        with tempfile.TemporaryDirectory() as directory:
            with BaritePhreeqcRM(
                np.ones((1, 1)),
                BariteChemistryConfig(thread_count=1, reaction_mode="equilibrium"),
                Path(directory) / "equilibrium",
            ) as chemistry:
                state = chemistry.react(
                    self._mixed_inlet(chemistry),
                    chemistry.initial_state.porosity,
                    dt=1.0,
                )
                equilibrium_moles = float(state.barite_moles[0, 0])

            with BaritePhreeqcRM(
                np.ones((1, 1)),
                BariteChemistryConfig(thread_count=1, reaction_mode="kinetic"),
                Path(directory) / "kinetic",
            ) as chemistry:
                state = chemistry.react(
                    self._mixed_inlet(chemistry),
                    chemistry.initial_state.porosity,
                    dt=1.0,
                )
                kinetic_moles = float(state.barite_moles[0, 0])
                kinetic_si = float(state.barite_saturation_index[0, 0])
                self.assertLess(max(state.element_relative_residuals.values()), 1.0e-8)

        self.assertGreater(kinetic_moles, 0.0)
        self.assertLess(kinetic_moles, equilibrium_moles)
        self.assertGreater(kinetic_si, 0.0)


if __name__ == "__main__":
    unittest.main()
