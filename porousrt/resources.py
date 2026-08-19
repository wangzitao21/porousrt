"""Packaged thermodynamic resources used by the reference cases."""

from __future__ import annotations

from pathlib import Path


def default_pitzer_database() -> Path:
    """Return the vendored Pitzer database path."""

    database = Path(__file__).resolve().parent / "data" / "pitzer.dat"
    if not database.is_file():
        raise FileNotFoundError(f"vendored Pitzer database not found: {database}")
    return database
