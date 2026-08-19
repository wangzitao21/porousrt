from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cases.two_inlet_barite_precipitation.model import (
    TwoInletBariteConfig,
    run_two_inlet_barite_simulation,
)


@unittest.skipUnless(
    importlib.util.find_spec("phreeqcrm") is not None,
    "phreeqcrm is not installed in this Python environment",
)
class TwoInletBariteCaseTests(unittest.TestCase):
    def test_smoke_run_saves_initial_and_every_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            config = TwoInletBariteConfig(
                nx=8,
                ny=8,
                length=1.6e-4,
                height=1.6e-4,
                left_inlet_velocity=2.0e-5,
                right_inlet_velocity=2.0e-5,
                dt=1.0,
                total_time=2.0,
                flow_interval=2,
                frame_interval=2,
                gravity_y=0.0,
                chemistry_threads=1,
                output_dir=str(output),
                make_video=False,
                make_gif=False,
            )

            final_state, history = run_two_inlet_barite_simulation(config)

            self.assertEqual(len(history), 3)
            self.assertAlmostEqual(final_state.time, 2.0)
            state_files = sorted((output / "states").glob("state_*.npz"))
            self.assertEqual(len(state_files), 3)
            self.assertEqual(
                [path.name for path in state_files],
                [
                    "state_000000.npz",
                    "state_000001.npz",
                    "state_000002.npz",
                ],
            )
            manifest = json.loads(
                (output / "state_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["saved_every_time_step"])
            self.assertEqual(len(manifest["states"]), 3)
            with np.load(state_files[-1]) as fields:
                self.assertEqual(fields["porosity"].shape, (8, 8))
                self.assertEqual(fields["u_m_s"].shape, (8, 9))
                self.assertEqual(fields["v_m_s"].shape, (9, 8))
                self.assertGreater(float(np.sum(fields["barite_moles"])), 0.0)
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertTrue((output / "final_fields.npz").is_file())


if __name__ == "__main__":
    unittest.main()
