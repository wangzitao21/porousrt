from __future__ import annotations

import unittest

import numpy as np

from cases.carnallite_wormholing.geometry import HALITE, SYLVITE
from cases.carnallite_wormholing.model import CarnalliteWormholeConfig
from cases.carnallite_wormholing_no_primary_sylvite.chemistry import MINERAL_NAMES
from cases.carnallite_wormholing_no_primary_sylvite.model import (
    NoPrimarySylviteWormholeConfig,
)


class NoPrimarySylviteWormholeTests(unittest.TestCase):
    def test_defaults_encode_only_the_two_requested_changes(self) -> None:
        baseline = CarnalliteWormholeConfig(make_video=False, make_gif=False)
        variant = NoPrimarySylviteWormholeConfig(
            make_video=False,
            make_gif=False,
        )

        self.assertEqual(variant.sylvite_solid_fraction, 0.0)
        self.assertEqual(
            variant.halite_solid_fraction,
            baseline.sylvite_solid_fraction + baseline.halite_solid_fraction,
        )
        self.assertEqual(
            variant.kinetic_rate_constant_mol_m2_s,
            2.0 * baseline.kinetic_rate_constant_mol_m2_s,
        )
        self.assertEqual(
            variant.sylvite_rate_constant_mol_m2_s,
            baseline.sylvite_rate_constant_mol_m2_s,
        )
        self.assertEqual(
            variant.halite_rate_constant_mol_m2_s,
            baseline.halite_rate_constant_mol_m2_s,
        )
        self.assertEqual(MINERAL_NAMES, ("Carnallite", "Sylvite", "Halite"))
        self.assertEqual(variant.chemistry_config.reaction_mode, "reversible_kinetic")
        self.assertGreater(
            variant.chemistry_config.kinetic_nucleation_inventory_mol_l,
            0.0,
        )
        variant.validate()

    def test_geometry_exactly_relabels_baseline_sylvite_as_halite(self) -> None:
        baseline = CarnalliteWormholeConfig(
            make_video=False,
            make_gif=False,
        ).build_microstructure()
        variant = NoPrimarySylviteWormholeConfig(
            make_video=False,
            make_gif=False,
        ).build_microstructure()

        np.testing.assert_array_equal(variant.porosity, baseline.porosity)
        np.testing.assert_array_equal(
            variant.inert_solid_fraction,
            baseline.inert_solid_fraction,
        )
        np.testing.assert_array_equal(
            variant.phase_volume_fractions[0],
            baseline.phase_volume_fractions[0],
        )
        np.testing.assert_array_equal(variant.phase_volume_fractions[1], 0.0)
        np.testing.assert_array_equal(
            variant.phase_volume_fractions[2],
            baseline.phase_volume_fractions[1]
            + baseline.phase_volume_fractions[2],
        )

        expected_labels = baseline.solid_labels.copy()
        expected_labels[expected_labels == SYLVITE] = HALITE
        np.testing.assert_array_equal(variant.solid_labels, expected_labels)

    def test_nonzero_primary_sylvite_is_rejected(self) -> None:
        config = NoPrimarySylviteWormholeConfig(
            sylvite_solid_fraction=0.01,
            halite_solid_fraction=0.31,
            make_video=False,
            make_gif=False,
        )
        with self.assertRaisesRegex(ValueError, "primary Sylvite fraction"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
