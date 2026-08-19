from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cases.halite_dissolution.chemistry import (
    HaliteChemistryConfig,
    HalitePhreeqcRM,
)


@unittest.skipUnless(
    importlib.util.find_spec("phreeqcrm") is not None,
    "phreeqcrm is not installed in this Python environment",
)
class HalitePhreeqcRMTests(unittest.TestCase):
    def test_undersaturated_brine_dissolves_only_available_halite(self) -> None:
        initial_porosity = np.array([[1.0, 0.5, 1.0e-3]])
        config = HaliteChemistryConfig(thread_count=1)

        with tempfile.TemporaryDirectory() as directory:
            artifact_directory = Path(directory)
            with HalitePhreeqcRM(
                initial_porosity, config, artifact_directory
            ) as chemistry:
                initial = chemistry.initial_state
                transported = np.broadcast_to(
                    chemistry.inlet_concentrations[:, None, None],
                    initial.concentrations.shape,
                ).copy()
                reacted = chemistry.react(transported, initial.porosity, dt=1.0)
                self.assertTrue((artifact_directory / "halite_dissolution.pqi").is_file())
                self.assertTrue(
                    (artifact_directory / "phreeqcrm_initialize.yaml").is_file()
                )

                self.assertAlmostEqual(float(initial.halite_moles[0, 0]), 0.0)
                self.assertAlmostEqual(float(reacted.halite_moles[0, 0]), 0.0)
                self.assertLess(
                    float(reacted.halite_moles[0, 1]),
                    float(initial.halite_moles[0, 1]),
                )
                self.assertGreater(
                    float(reacted.porosity[0, 1]), float(initial.porosity[0, 1])
                )
                self.assertLess(
                    float(reacted.density_kg_m3[0, 0]),
                    float(initial.density_kg_m3[0, 0]),
                )
                # The tiny pore-water volume in the near-solid cell permits only a
                # correspondingly tiny equilibrium dissolution increment.
                interface_loss = (
                    initial.halite_moles[0, 1] - reacted.halite_moles[0, 1]
                )
                core_loss = initial.halite_moles[0, 2] - reacted.halite_moles[0, 2]
                self.assertLess(float(core_loss), 0.01 * float(interface_loss))
                self.assertLess(reacted.chemistry_na_relative_residual, 1.0e-11)
                self.assertLess(reacted.chemistry_cl_relative_residual, 1.0e-11)
                self.assertLess(
                    float(np.max(np.abs(reacted.phreeqc_percent_error))), 1.0e-3
                )


if __name__ == "__main__":
    unittest.main()
