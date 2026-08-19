"""Animation helpers shared by the example cases."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def resolve_animation_fps(
    frame_count: int,
    fps: float,
    duration_seconds: float | None,
) -> float:
    """Return playback rate, optionally targeting an exact duration."""

    if frame_count <= 0:
        raise ValueError("animation requires at least one frame")
    if duration_seconds is not None:
        if duration_seconds <= 0.0:
            raise ValueError("animation duration must be positive")
        return frame_count / duration_seconds
    if fps <= 0.0:
        raise ValueError("animation fps must be positive")
    return float(fps)


def make_mp4(frames_dir: Path, output: Path, fps: float) -> None:
    """Encode numbered PNG frames as an H.264 MP4."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found; cannot create MP4")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def make_gif(
    frames_dir: Path,
    output: Path,
    source_fps: float,
    maximum_fps: float = 12.0,
) -> None:
    """Encode a palette-optimized GIF, downsampling only dense frame sets."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found; cannot create GIF")
    if source_fps <= 0.0:
        raise ValueError("source fps must be positive")
    if maximum_fps <= 0.0:
        raise ValueError("GIF fps must be positive")
    output_fps = min(source_fps, maximum_fps)
    filter_graph = (
        f"fps={output_fps:.12g},scale=960:-2:flags=lanczos,"
        "split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(source_fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-filter_complex",
        filter_graph,
        "-loop",
        "0",
        str(output),
    ]
    subprocess.run(command, check=True)
