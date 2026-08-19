"""Reusable, restart-friendly artifact writers for coupled simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class TimeLevelRecord:
    """One persisted time level in a :class:`TimeLevelArchive`."""

    step: int
    time_s: float
    file: str


class TimeLevelArchive:
    """Save a complete compressed NPZ and refresh a manifest at every step.

    The class is chemistry- and case-independent: callers provide a mapping of
    named arrays or scalar values.  ``step_index`` and ``time_s`` are inserted
    consistently by the archive.  Updating the JSON manifest after each NPZ
    means a partially completed run still has an accurate list of durable time
    levels.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        manifest_path: str | Path | None = None,
        prefix: str = "state",
        digits: int = 6,
        clear_existing: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not prefix or any(character in prefix for character in "\\/:"):
            raise ValueError("archive prefix must be a simple nonempty filename stem")
        if digits < 1:
            raise ValueError("archive digit count must be positive")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else self.directory / "manifest.json"
        )
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.digits = int(digits)
        self.metadata = dict(metadata or {})
        existing = sorted(self.directory.glob(f"{self.prefix}_*.npz"))
        if existing and not clear_existing:
            raise FileExistsError(
                f"time-level archive already contains {len(existing)} file(s): "
                f"{self.directory}"
            )
        if clear_existing:
            for path in existing:
                path.unlink()
            if self.manifest_path.is_file():
                self.manifest_path.unlink()
            temporary_manifest = self.manifest_path.with_suffix(
                self.manifest_path.suffix + ".tmp"
            )
            if temporary_manifest.is_file():
                temporary_manifest.unlink()
        self._records: list[TimeLevelRecord] = []

    @property
    def records(self) -> tuple[TimeLevelRecord, ...]:
        return tuple(self._records)

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.manifest_path.parent).as_posix()
        except ValueError:
            return str(path)

    def _write_manifest(self) -> None:
        payload = {
            "format": "compressed NumPy NPZ",
            "includes_initial_state": bool(
                self._records
                and self._records[0].step == 0
                and np.isclose(self._records[0].time_s, 0.0)
            ),
            **self.metadata,
            "states": [asdict(record) for record in self._records],
        }
        temporary_path = self.manifest_path.with_suffix(
            self.manifest_path.suffix + ".tmp"
        )
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary_path.replace(self.manifest_path)

    def save(
        self,
        step: int,
        time_s: float,
        fields: Mapping[str, Any],
    ) -> Path:
        """Persist one strictly increasing step and return its absolute path."""

        if step < 0:
            raise ValueError("archive step must be nonnegative")
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("archive time must be finite and nonnegative")
        if self._records and step <= self._records[-1].step:
            raise ValueError("archive steps must be strictly increasing")
        if self._records and time_s < self._records[-1].time_s:
            raise ValueError("archive times must be nondecreasing")
        reserved = {"step_index", "time_s"} & set(fields)
        if reserved:
            raise ValueError(
                "archive fields use reserved name(s): " + ", ".join(sorted(reserved))
            )
        path = self.directory / f"{self.prefix}_{step:0{self.digits}d}.npz"
        np.savez_compressed(
            path,
            step_index=np.array(step, dtype=np.int64),
            time_s=np.array(float(time_s)),
            **fields,
        )
        self._records.append(
            TimeLevelRecord(
                step=int(step),
                time_s=float(time_s),
                file=self._display_path(path),
            )
        )
        self._write_manifest()
        return path
