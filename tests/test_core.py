from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from porousrt.artifacts import TimeLevelArchive
from porousrt.geometry import build_staggered_square_porosity
from porousrt.grid import Grid
from porousrt.media import make_mp4, resolve_animation_fps


class GridAndGeometryTests(unittest.TestCase):
    def test_grid_rejects_nonpositive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            Grid(nx=0, ny=4, length=1.0, height=1.0)
        with self.assertRaises(ValueError):
            Grid(nx=4, ny=4, length=-1.0, height=1.0)

    def test_square_geometry_contains_fluid_solid_and_interface(self) -> None:
        grid = Grid(nx=60, ny=24, length=6.0e-3, height=2.4e-3)
        porosity = build_staggered_square_porosity(grid)
        self.assertEqual(porosity.shape, (grid.ny, grid.nx))
        self.assertTrue(np.any(porosity < 0.01))
        self.assertTrue(np.any((porosity > 0.01) & (porosity < 0.99)))
        self.assertTrue(np.any(porosity > 0.99))


class MediaTests(unittest.TestCase):
    def test_requested_animation_duration_sets_playback_rate(self) -> None:
        self.assertAlmostEqual(resolve_animation_fps(400, 8.0, 40.0), 10.0)

    @patch("porousrt.media.subprocess.run")
    @patch("porousrt.media.shutil.which", return_value="ffmpeg")
    def test_mp4_encoder_pads_odd_frame_dimensions(
        self, _which: object, run: object
    ) -> None:
        make_mp4(Path("frames"), Path("movie.mp4"), fps=10.0)
        command = run.call_args.args[0]
        self.assertIn("pad=ceil(iw/2)*2:ceil(ih/2)*2", command)
        run.assert_called_once_with(command, check=True)


class ArtifactTests(unittest.TestCase):
    def test_time_level_archive_saves_each_step_and_refreshes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = TimeLevelArchive(
                root / "states",
                manifest_path=root / "state_manifest.json",
                metadata={"saved_every_time_step": True},
            )
            first = archive.save(0, 0.0, {"field": np.zeros((2, 3))})
            second = archive.save(1, 0.5, {"field": np.ones((2, 3))})

            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            with np.load(second) as values:
                self.assertEqual(int(values["step_index"]), 1)
                self.assertAlmostEqual(float(values["time_s"]), 0.5)
                np.testing.assert_array_equal(values["field"], np.ones((2, 3)))
            manifest = json.loads(
                (root / "state_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["includes_initial_state"])
            self.assertTrue(manifest["saved_every_time_step"])
            self.assertEqual(len(manifest["states"]), 2)
            with self.assertRaises(ValueError):
                archive.save(1, 1.0, {"field": np.ones(1)})


if __name__ == "__main__":
    unittest.main()
