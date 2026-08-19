from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import generate_binary_structure, label

from cases.carnallite_wormholing.geometry import (
    build_mixed_mineral_porous_aquifer,
    build_seeded_carnallite_ore_porosity,
    build_staggered_carnallite_microstructure,
)
from cases.carnallite_wormholing.model import (
    CarnalliteWormholeConfig,
    ReactionLedger,
    _advance_reaction_ledger,
    _connected_emergent_channel,
    _dissolution_diagnostics,
    _flow_localization,
)
from porousrt.grid import Grid


class CarnalliteWormholeGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = Grid(nx=64, ny=24, length=1.6e-3, height=6.0e-4)

    def test_microstructure_is_reproducible_resolved_and_connected(self) -> None:
        first = build_staggered_carnallite_microstructure(self.grid, random_seed=7)
        second = build_staggered_carnallite_microstructure(self.grid, random_seed=7)
        changed = build_staggered_carnallite_microstructure(self.grid, random_seed=8)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, changed))
        self.assertEqual(first.shape, (self.grid.ny, self.grid.nx))
        self.assertAlmostEqual(float(np.min(first)), 1.0e-3)
        self.assertAlmostEqual(float(np.max(first)), 1.0)
        self.assertGreater(float(np.mean(1.0 - first)), 0.45)
        self.assertLess(float(np.mean(1.0 - first)), 0.75)
        self.assertTrue(np.any((first > 1.0e-3) & (first < 1.0)))

        pore_labels, _ = label(first > 0.5)
        spanning = (set(pore_labels[:, 0]) & set(pore_labels[:, -1])) - {0}
        self.assertTrue(spanning)

    def test_inlet_and_outlet_buffers_are_open(self) -> None:
        porosity = build_staggered_carnallite_microstructure(
            self.grid, inlet_outlet_buffer_cells=3
        )
        np.testing.assert_allclose(porosity[:, :3], 1.0)
        np.testing.assert_allclose(porosity[:, -3:], 1.0)

    def test_seeded_ore_has_three_competing_resolved_flaws(self) -> None:
        porosity = build_seeded_carnallite_ore_porosity(
            self.grid,
            random_seed=2026,
            inlet_outlet_buffer_cells=3,
        )
        self.assertGreater(float(np.mean(1.0 - porosity)), 0.65)
        self.assertTrue(np.any((porosity > 1.0e-3) & (porosity < 1.0)))
        pore_labels, _ = label(
            porosity > 0.5,
            structure=generate_binary_structure(2, 2),
        )
        spanning = (set(pore_labels[:, 0]) & set(pore_labels[:, -1])) - {0}
        self.assertTrue(spanning)

    def test_default_aquifer_is_a_connected_mixed_mineral_pore_network(self) -> None:
        config = CarnalliteWormholeConfig(
            geometry_kind="porous_aquifer",
            make_video=False,
            make_gif=False,
        )
        first = config.build_microstructure()
        second = build_mixed_mineral_porous_aquifer(
            config.grid,
            pitch_x=config.pitch_x,
            pitch_y=config.pitch_y,
            radius_fraction=config.radius_fraction,
            radius_jitter=config.radius_jitter,
            center_jitter=config.center_jitter,
            interface_width_cells=config.interface_width_cells,
            random_seed=config.random_seed,
            inlet_outlet_buffer_cells=config.inlet_outlet_buffer_cells,
            weak_seed_fraction=config.weak_seed_fraction,
        )

        first.validate()
        np.testing.assert_array_equal(first.porosity, second.porosity)
        closure = (
            first.porosity
            + first.inert_solid_fraction
            + np.sum(first.phase_volume_fractions, axis=0)
        )
        np.testing.assert_allclose(closure, 1.0, atol=2.0e-12)
        self.assertTrue(np.all(np.sum(first.phase_volume_fractions, axis=(1, 2)) > 0))
        self.assertGreater(float(np.sum(first.inert_solid_fraction)), 0.0)

        pore_labels, _ = label(
            first.porosity > 0.55,
            structure=generate_binary_structure(2, 2),
        )
        spanning = (set(pore_labels[:, 0]) & set(pore_labels[:, -1])) - {0}
        self.assertTrue(spanning)
        self.assertFalse(np.any(np.all(first.porosity > 0.55, axis=1)))

    def test_configuration_selects_kinetics_saturated_halite_and_tracer(self) -> None:
        config = CarnalliteWormholeConfig(
            nx=64,
            ny=24,
            geometry_kind="porous_aquifer",
            dt=1.0,
            total_time=10.0,
            make_video=False,
            make_gif=False,
        )
        config.validate()
        chemistry = config.chemistry_config
        self.assertEqual(config.geometry_kind, "porous_aquifer")
        self.assertEqual(config.frame_interval, 1)
        self.assertEqual(chemistry.reaction_mode, "reversible_kinetic")
        self.assertTrue(chemistry.saturate_injected_halite)
        self.assertTrue(chemistry.equilibrate_formation_evaporites)
        self.assertGreater(
            chemistry.kinetic_rate_constant_mol_m2_s,
            chemistry.sylvite_rate_constant_mol_m2_s,
        )
        self.assertGreater(chemistry.halite_rate_constant_mol_m2_s, 0.0)
        self.assertGreater(chemistry.kinetic_nucleation_inventory_mol_l, 0.0)
        self.assertGreater(chemistry.injected_br_molality, 0.0)

    def test_default_case_runs_300_seconds_and_samples_each_second(self) -> None:
        config = CarnalliteWormholeConfig(make_video=False, make_gif=False)
        config.validate()
        self.assertEqual(config.geometry_kind, "porous_aquifer")
        microstructure = config.build_microstructure()
        self.assertTrue(
            np.all(np.sum(microstructure.phase_volume_fractions, axis=(1, 2)) > 0)
        )
        self.assertGreater(float(np.sum(microstructure.inert_solid_fraction)), 0.0)
        self.assertEqual(config.inlet_profile_kind, "uniform")
        np.testing.assert_allclose(config.build_inlet_velocity_profile(), 1.0)
        self.assertEqual(config.dt, 1.0)
        self.assertEqual(config.total_time, 300.0)
        self.assertEqual(config.n_steps, 300)
        self.assertEqual(config.frame_interval, 1)
        self.assertGreater(
            config.kinetic_rate_constant_mol_m2_s,
            config.sylvite_rate_constant_mol_m2_s,
        )
        self.assertGreater(
            config.sylvite_rate_constant_mol_m2_s,
            config.halite_rate_constant_mol_m2_s,
        )


class CarnalliteWormholeDiagnosticsTests(unittest.TestCase):
    def test_reaction_ledger_keeps_transient_precipitation(self) -> None:
        initial = np.array([[[2.0]], [[0.0]], [[0.0]]])
        ledger = ReactionLedger.initialize(initial)
        _advance_reaction_ledger(
            ledger,
            np.array([[[1.5]], [[0.4]], [[0.1]]]),
        )
        _advance_reaction_ledger(
            ledger,
            np.array([[[1.7]], [[0.1]], [[0.05]]]),
        )

        np.testing.assert_allclose(
            ledger.cumulative_precipitated_moles[:, 0, 0],
            (0.2, 0.4, 0.1),
        )
        np.testing.assert_allclose(
            ledger.cumulative_dissolved_moles[:, 0, 0],
            (0.5, 0.3, 0.05),
        )

    def test_connected_channel_requires_reaction_or_flow_amplification(self) -> None:
        config = CarnalliteWormholeConfig(
            nx=10,
            ny=6,
            length=1.0,
            height=0.6,
            inlet_outlet_buffer_cells=1,
            inlet_velocity=1.0,
        )
        initial_porosity = np.full((6, 10), 0.70)
        initial_speed = np.ones((6, 10))
        mask, reach, spans = _connected_emergent_channel(
            initial_porosity,
            initial_porosity.copy(),
            initial_speed,
            initial_speed.copy(),
            config,
        )
        self.assertFalse(np.any(mask))
        self.assertEqual(reach, 0.0)
        self.assertFalse(spans)

        opened = initial_porosity.copy()
        opened[2, :] += 0.04
        accelerated = initial_speed.copy()
        accelerated[2, :] *= 2.0
        mask, reach, spans = _connected_emergent_channel(
            initial_porosity,
            opened,
            initial_speed,
            accelerated,
            config,
        )
        self.assertTrue(np.any(mask))
        self.assertEqual(reach, 1.0)
        self.assertTrue(spans)

    def test_flow_localization_distinguishes_uniform_and_focused_flux(self) -> None:
        uniform = np.ones((10, 5))
        width, fastest, index = _flow_localization(uniform, 2)
        self.assertAlmostEqual(width, 1.0)
        self.assertAlmostEqual(fastest, 0.2)
        self.assertAlmostEqual(index, 0.0)

        focused = np.zeros((10, 5))
        focused[0, :] = 1.0
        width, fastest, index = _flow_localization(focused, 2)
        self.assertAlmostEqual(width, 0.1)
        self.assertAlmostEqual(fastest, 1.0)
        self.assertAlmostEqual(index, 0.9)

    def test_dissolution_metrics_capture_localized_sweep(self) -> None:
        grid = Grid(nx=10, ny=4, length=1.0, height=0.4)
        initial = np.ones((4, 10))
        current = initial.copy()
        current[0, :5] = 0.0
        local, contact, sweep, top_share, reach = _dissolution_diagnostics(
            initial, current, grid
        )
        self.assertTrue(np.allclose(local[0, :5], 1.0))
        self.assertAlmostEqual(contact, 0.125)
        self.assertAlmostEqual(sweep, 0.125)
        self.assertGreater(top_share, 0.0)
        self.assertAlmostEqual(reach, 0.45)


if __name__ == "__main__":
    unittest.main()
