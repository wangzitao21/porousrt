import tempfile
import unittest
from pathlib import Path

import numpy as np

from cases.carnallite_replacement.chemistry import (
    EvaporiteChemistryConfig,
    EvaporitePhreeqcRM,
    MINERAL_MOLAR_VOLUMES_L_MOL,
)


class EvaporitePhreeqcRMTests(unittest.TestCase):
    def test_reversible_precipitation_respects_minimum_pore_volume(self) -> None:
        porosity = np.array([[1.0e-3]])
        phases = np.zeros((3, 1, 1))
        phases[2, 0, 0] = 1.0 - porosity[0, 0]
        config = EvaporiteChemistryConfig(
            thread_count=1,
            reaction_mode="reversible_kinetic",
            formation_na_molality=1.0e-6,
            formation_k_molality=0.01,
            formation_mg_molality=5.0,
            injected_na_molality=5.5,
            injected_k_molality=0.0,
            injected_mg_molality=0.0,
            injected_br_molality=0.02,
            saturate_injected_halite=True,
            equilibrate_formation_evaporites=True,
            kinetic_rate_constant_mol_m2_s=4.0e-2,
            sylvite_rate_constant_mol_m2_s=4.0e-3,
            halite_rate_constant_mol_m2_s=2.0e-4,
            kinetic_nucleation_inventory_mol_l=1.0e-2,
        )
        with tempfile.TemporaryDirectory() as directory:
            with EvaporitePhreeqcRM(
                porosity,
                config,
                Path(directory),
                initial_phase_volume_fractions=phases,
            ) as chemistry:
                initial = chemistry.initial_state
                supersaturated = initial.concentrations.copy()
                supersaturated[chemistry.component_index["na"]] += 1.0
                supersaturated[chemistry.component_index["cl"]] += 1.0
                reacted = chemistry.react(
                    supersaturated,
                    initial.porosity,
                    100.0,
                )

                self.assertGreater(
                    float(reacted.saturation_indices["halite"][0, 0]),
                    0.0,
                )
                self.assertLessEqual(
                    float(reacted.phase_volume_fractions[2, 0, 0]),
                    1.0 - config.minimum_porosity + 1.0e-12,
                )
                np.testing.assert_allclose(
                    reacted.porosity
                    + np.sum(reacted.phase_volume_fractions, axis=0),
                    1.0,
                    atol=2.0e-12,
                )

    def test_reversible_kinetics_unifies_all_mineral_inventories(self) -> None:
        porosity = np.array([[0.3, 0.3, 0.3, 1.0]])
        phases = np.zeros((3, 1, 4))
        phases[0, 0, 0] = 0.7
        phases[1, 0, 1] = 0.7
        phases[2, 0, 2] = 0.7
        config = EvaporiteChemistryConfig(
            thread_count=1,
            reaction_mode="reversible_kinetic",
            formation_na_molality=1.0e-6,
            formation_k_molality=0.01,
            formation_mg_molality=5.0,
            injected_na_molality=5.5,
            injected_k_molality=0.0,
            injected_mg_molality=0.0,
            injected_br_molality=0.02,
            saturate_injected_halite=True,
            equilibrate_formation_evaporites=True,
            kinetic_rate_constant_mol_m2_s=2.0e-2,
            sylvite_rate_constant_mol_m2_s=4.0e-3,
            halite_rate_constant_mol_m2_s=3.0e-3,
            kinetic_nucleation_inventory_mol_l=1.0e-2,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            with EvaporitePhreeqcRM(
                porosity,
                config,
                artifact_dir,
                initial_phase_volume_fractions=phases,
            ) as chemistry:
                initial = chemistry.initial_state
                np.testing.assert_allclose(
                    initial.phase_moles,
                    initial.kinetic_phase_moles,
                    atol=1.0e-12,
                )
                np.testing.assert_allclose(
                    initial.equilibrium_phase_moles,
                    0.0,
                    atol=1.0e-12,
                )
                self.assertGreater(float(initial.phase_moles[0, 0, 0]), 0.0)
                self.assertGreater(float(initial.phase_moles[1, 0, 1]), 0.0)
                self.assertGreater(float(initial.phase_moles[2, 0, 2]), 0.0)
                self.assertLess(float(np.max(initial.phase_moles[:, 0, 3])), 1.0e-12)

                mixed = (
                    0.9 * initial.concentrations
                    + 0.1 * chemistry.inlet_concentrations[:, None, None]
                )
                reacted = chemistry.react(mixed, initial.porosity, 2.0)
                self.assertLess(
                    float(reacted.phase_moles[0, 0, 0]),
                    float(initial.phase_moles[0, 0, 0]),
                )
                np.testing.assert_allclose(
                    reacted.equilibrium_phase_moles,
                    0.0,
                    atol=1.0e-12,
                )

                pqi = (artifact_dir / "carnallite_replacement.pqi").read_text(
                    encoding="utf-8"
                )
                self.assertIn("Carnallite_reversible", pqi)
                self.assertIn("Sylvite_reversible", pqi)
                self.assertIn("Halite_reversible", pqi)
                self.assertIn("KINETICS 4 Unified reversible minerals", pqi)
                self.assertNotIn("Secondary salts", pqi)

    def test_mixed_primary_minerals_secondary_salts_and_inert_skeleton(self) -> None:
        porosity = np.array([[0.3, 0.3, 0.3, 0.3, 1.0]])
        phases = np.zeros((3, 1, 5))
        phases[0, 0, 0] = 0.7
        phases[1, 0, 1] = 0.7
        phases[2, 0, 2] = 0.7
        inert = np.zeros((1, 5))
        inert[0, 3] = 0.7
        config = EvaporiteChemistryConfig(
            thread_count=1,
            reaction_mode="kinetic",
            formation_na_molality=1.0e-6,
            formation_k_molality=0.01,
            formation_mg_molality=5.0,
            injected_na_molality=5.5,
            injected_k_molality=0.0,
            injected_mg_molality=0.0,
            injected_br_molality=0.02,
            saturate_injected_halite=True,
            equilibrate_formation_evaporites=True,
            kinetic_sylvite=True,
            kinetic_rate_constant_mol_m2_s=1.0e-2,
            sylvite_rate_constant_mol_m2_s=6.0e-3,
        )
        with tempfile.TemporaryDirectory() as directory:
            with EvaporitePhreeqcRM(
                porosity,
                config,
                Path(directory),
                initial_phase_volume_fractions=phases,
                inert_solid_fraction=inert,
            ) as chemistry:
                initial = chemistry.initial_state
                self.assertGreater(float(initial.kinetic_phase_moles[0, 0, 0]), 0.0)
                self.assertGreater(float(initial.kinetic_phase_moles[1, 0, 1]), 0.0)
                self.assertGreater(float(initial.equilibrium_phase_moles[2, 0, 2]), 0.0)
                initial_halite = float(np.sum(initial.phase_moles[2]))

                mixed = (
                    0.9 * initial.concentrations
                    + 0.1 * chemistry.inlet_concentrations[:, None, None]
                )
                reacted = chemistry.react(mixed, initial.porosity, 2.0)

                self.assertLess(
                    float(reacted.kinetic_phase_moles[0, 0, 0]),
                    float(initial.kinetic_phase_moles[0, 0, 0]),
                )
                self.assertLess(
                    float(reacted.kinetic_phase_moles[1, 0, 1]),
                    float(initial.kinetic_phase_moles[1, 0, 1]),
                )
                self.assertGreater(float(np.sum(reacted.equilibrium_phase_moles[1])), 0.0)
                self.assertGreater(float(np.sum(reacted.phase_moles[2])), initial_halite)
                np.testing.assert_array_equal(reacted.inert_solid_fraction, inert)
                self.assertAlmostEqual(float(reacted.porosity[0, 3]), 0.3)

    def test_carnallite_replacement_phases_are_transient(self) -> None:
        # Five moles of carnallite in the one-litre representative cell.
        porosity = np.array([[1.0 - 5.0 * MINERAL_MOLAR_VOLUMES_L_MOL[0]]])
        config = EvaporiteChemistryConfig(thread_count=1)
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            with EvaporitePhreeqcRM(porosity, config, artifact_dir) as chemistry:
                state = chemistry.initial_state
                self.assertAlmostEqual(float(state.phase_moles[0, 0, 0]), 5.0, places=8)
                self.assertLess(float(np.max(state.phase_moles[1:])), 1.0e-12)
                self.assertTrue((artifact_dir / "carnallite_replacement.pqi").is_file())
                self.assertTrue((artifact_dir / "phreeqcrm_initialize.yaml").is_file())

                peak_sylvite = 0.0
                peak_halite = 0.0
                maximum_residual = 0.0
                fraction = 0.035
                for _ in range(400):
                    mixed = (
                        (1.0 - fraction) * state.concentrations
                        + fraction * chemistry.inlet_concentrations[:, None, None]
                    )
                    state = chemistry.react(mixed, state.porosity, 1.0)
                    peak_sylvite = max(peak_sylvite, float(np.sum(state.phase_moles[1])))
                    peak_halite = max(peak_halite, float(np.sum(state.phase_moles[2])))
                    maximum_residual = max(
                        maximum_residual,
                        max(state.element_relative_residuals.values()),
                    )

                self.assertGreater(peak_sylvite, 3.0)
                self.assertGreater(peak_halite, 0.1)
                self.assertLess(float(np.sum(state.phase_moles)), 1.0e-7)
                self.assertLess(maximum_residual, 2.0e-8)
                np.testing.assert_allclose(
                    1.0 - state.porosity,
                    np.sum(state.phase_volume_fractions, axis=0),
                    atol=2.0e-13,
                )

    def test_kinetic_carnallite_is_rate_limited_and_bromide_is_conservative(self) -> None:
        initial_moles = 1.0
        porosity = np.array(
            [[1.0 - initial_moles * MINERAL_MOLAR_VOLUMES_L_MOL[0]]]
        )
        config = EvaporiteChemistryConfig(
            thread_count=1,
            reaction_mode="kinetic",
            injected_na_molality=5.5,
            injected_k_molality=0.0,
            injected_mg_molality=0.0,
            injected_br_molality=0.01,
            saturate_injected_halite=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            with EvaporitePhreeqcRM(porosity, config, artifact_dir) as chemistry:
                initial = chemistry.initial_state
                self.assertAlmostEqual(
                    float(initial.phase_moles[0, 0, 0]), initial_moles, places=8
                )
                reacted = chemistry.react(
                    np.broadcast_to(
                        chemistry.inlet_concentrations[:, None, None],
                        initial.concentrations.shape,
                    ).copy(),
                    initial.porosity,
                    1.0,
                )

                remaining = float(reacted.phase_moles[0, 0, 0])
                self.assertLess(remaining, initial_moles)
                self.assertGreater(remaining, 0.9 * initial_moles)
                self.assertLess(
                    reacted.element_relative_residuals["br"], 1.0e-8
                )
                pqi = (artifact_dir / "carnallite_replacement.pqi").read_text(
                    encoding="utf-8"
                )
                self.assertIn("KINETICS 1", pqi)
                self.assertIn("Halite-saturated injected brine", pqi)


if __name__ == "__main__":
    unittest.main()
