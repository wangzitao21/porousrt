from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cases.carnallite_wormholing.model import CarnalliteWormholeConfig
from cases.carnallite_wormholing_equilibrium.chemistry import EvaporitePhreeqcRM
from cases.carnallite_wormholing_equilibrium.model import (
    EquilibriumCarnalliteWormholeConfig,
)


class EquilibriumCarnalliteWormholeTests(unittest.TestCase):
    def test_defaults_only_change_identity_reaction_mode_and_duration(self) -> None:
        baseline = CarnalliteWormholeConfig(make_video=False, make_gif=False)
        variant = EquilibriumCarnalliteWormholeConfig(
            make_video=False,
            make_gif=False,
        )

        baseline_values = asdict(baseline)
        variant_values = asdict(variant)
        for identity_field in (
            "case_slug",
            "scenario_label",
            "output_dir",
            "reaction_mode",
            "total_time",
        ):
            baseline_values.pop(identity_field)
            variant_values.pop(identity_field)

        self.assertEqual(variant_values, baseline_values)
        self.assertEqual(variant.total_time, 2.0 * baseline.total_time)
        self.assertEqual(variant.n_steps, 600)
        self.assertEqual(baseline.chemistry_config.reaction_mode, "reversible_kinetic")
        self.assertEqual(variant.chemistry_config.reaction_mode, "equilibrium")
        variant.validate()

    def test_initial_geometry_is_exactly_the_baseline_geometry(self) -> None:
        baseline = CarnalliteWormholeConfig(
            make_video=False,
            make_gif=False,
        ).build_microstructure()
        variant = EquilibriumCarnalliteWormholeConfig(
            make_video=False,
            make_gif=False,
        ).build_microstructure()

        np.testing.assert_array_equal(variant.porosity, baseline.porosity)
        np.testing.assert_array_equal(
            variant.phase_volume_fractions,
            baseline.phase_volume_fractions,
        )
        np.testing.assert_array_equal(
            variant.inert_solid_fraction,
            baseline.inert_solid_fraction,
        )
        np.testing.assert_array_equal(variant.solid_labels, baseline.solid_labels)

    def test_non_equilibrium_mode_is_rejected(self) -> None:
        config = EquilibriumCarnalliteWormholeConfig(
            reaction_mode="reversible_kinetic",  # type: ignore[arg-type]
            make_video=False,
            make_gif=False,
        )
        with self.assertRaisesRegex(ValueError, "requires reaction_mode='equilibrium'"):
            config.validate()

    def test_phreeqc_input_contains_only_equilibrium_mineral_reactants(self) -> None:
        porosity = np.array([[0.3, 0.3, 0.3, 1.0]])
        phases = np.zeros((3, 1, 4))
        phases[0, 0, 0] = 0.7
        phases[1, 0, 1] = 0.7
        phases[2, 0, 2] = 0.7
        config = EquilibriumCarnalliteWormholeConfig(
            chemistry_threads=1,
            make_video=False,
            make_gif=False,
        ).chemistry_config

        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            with EvaporitePhreeqcRM(
                porosity,
                config,
                artifact_dir,
                initial_phase_volume_fractions=phases,
            ) as chemistry:
                initial = chemistry.initial_state
                np.testing.assert_allclose(initial.kinetic_phase_moles, 0.0)
                np.testing.assert_allclose(
                    initial.phase_moles,
                    initial.equilibrium_phase_moles,
                )

            pqi = (artifact_dir / "carnallite_replacement.pqi").read_text(
                encoding="utf-8"
            )
            self.assertIn("EQUILIBRIUM_PHASES 1 Primary Carnallite", pqi)
            self.assertIn("EQUILIBRIUM_PHASES 2 Primary Sylvite", pqi)
            self.assertIn("EQUILIBRIUM_PHASES 3 Primary Halite", pqi)
            self.assertIn("EQUILIBRIUM_PHASES 4 Initially open pore", pqi)
            self.assertNotIn("\nRATES\n", pqi)
            self.assertNotIn("\nKINETICS ", pqi)

    def test_equilibrium_precipitation_stops_at_closed_pore_volume(self) -> None:
        config = EquilibriumCarnalliteWormholeConfig(
            chemistry_threads=1,
            make_video=False,
            make_gif=False,
        ).chemistry_config
        porosity = np.array([[config.minimum_porosity]])
        phases = np.zeros((3, 1, 1))
        phases[2, 0, 0] = 1.0 - config.minimum_porosity

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
                self.assertGreaterEqual(
                    float(reacted.porosity[0, 0]),
                    config.minimum_porosity,
                )
                self.assertLessEqual(
                    float(np.sum(reacted.phase_volume_fractions[:, 0, 0])),
                    1.0 - config.minimum_porosity + 1.0e-12,
                )


if __name__ == "__main__":
    unittest.main()
